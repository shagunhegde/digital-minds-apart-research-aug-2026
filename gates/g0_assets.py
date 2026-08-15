"""GATE G0: assets and environment.

Emits a report and stops. It judges nothing -- no PASS, no FAIL, no "as
expected". A human reads the numbers and decides whether Phase 1 proceeds.

Runs on the GPU box. Needs the model (55.6 GB bf16) and the lens (6.6 GB
resident fp32) in memory at once, so it is the first real check that the
tier is what the sprint assumed.

    python gates/g0_assets.py [--layers-stride N] [--out artifacts/g0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import jlens
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import stats  # noqa: E402

RULE = "=" * 78


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def gpu_query() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip() or "nvidia-smi returned nothing"
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi unavailable: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers-stride", type=int, default=1,
                    help="subsample layers for the per-layer sweeps")
    ap.add_argument("--n-tokens", type=int, default=1000,
                    help="random tokens for the identity cross-check")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "g0")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    cfg = yaml.safe_load((ROOT / "configs" / "sprint.yaml").read_text())
    torch.manual_seed(cfg["seed"])

    # ---------------------------------------------------------------- load
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer

    lens_path = Path(hf_hub_download(
        repo_id=cfg["lens"]["repo"], filename=cfg["lens"]["filename"],
        revision=cfg["lens"]["revision"],
    ))
    t_lens = time.time()
    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    raw_j = checkpoint["J"]
    stored_dtype = next(iter(raw_j.values())).dtype
    lens = jlens.JacobianLens(
        jacobians=raw_j, n_prompts=checkpoint["n_prompts"],
        d_model=checkpoint["d_model"],
    )
    lens_load_s = time.time() - t_lens

    tok = AutoTokenizer.from_pretrained(
        cfg["model"]["repo"], revision=cfg["model"]["revision"])
    hf_config = AutoConfig.from_pretrained(
        cfg["model"]["repo"], revision=cfg["model"]["revision"])

    loader_used = None
    hf_model = None
    import transformers
    for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            hf_model = cls.from_pretrained(
                cfg["model"]["repo"], revision=cfg["model"]["revision"],
                dtype=torch.bfloat16, device_map="auto",
                attn_implementation="sdpa",
            )
            loader_used = name
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[load] {name} failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    if hf_model is None:
        raise RuntimeError("no transformers auto-class loaded this checkpoint")

    model = jlens.from_hf(hf_model, tok)
    model_load_s = time.time() - t_start

    # ------------------------------------------------------------ measures
    source_layers = list(lens.source_layers)
    swept = source_layers[:: args.layers_stride]

    # invariant: no NaN/Inf, no degenerate rows
    n_nan = n_inf = n_zero_rows = 0
    row_norm_rows = []
    for layer in source_layers:
        j = lens.jacobians[layer]
        n_nan += int(torch.isnan(j).sum())
        n_inf += int(torch.isinf(j).sum())
        norms = j.norm(dim=1)
        n_zero_rows += int((norms == 0).sum())
        row_norm_rows.append((layer, stats.median_iqr(norms.numpy())))

    # invariant: determinism
    probe = "The currency used in the country shaped like a boot is"
    ids = model.encode(probe)
    with torch.inference_mode():
        logits_a = model.unembed(_final_residual(model, ids)).float()
        logits_b = model.unembed(_final_residual(model, ids)).float()
    determinism_delta = float((logits_a - logits_b).abs().max())

    # cost: peak VRAM after a 128-token forward
    torch.cuda.reset_peak_memory_stats()
    long_ids = torch.randint(0, hf_config.get_text_config().vocab_size,
                             (1, 128), device=model.input_device)
    with torch.inference_mode():
        _final_residual(model, long_ids)
    peak_bytes = torch.cuda.max_memory_allocated()
    total_bytes = torch.cuda.get_device_properties(0).total_memory

    # cross-check: W_U J_l -> W_U as l -> final.
    # HFLensModel keeps the unembedding private; the lens matrix we want to
    # characterise is W_U J_l, whose row t is W_U[t] @ J_l.
    unembed_w = model._lm_head.weight  # [vocab, d_model]
    rng = np.random.default_rng(cfg["seed"])
    token_ids = torch.as_tensor(
        rng.choice(unembed_w.shape[0], size=args.n_tokens, replace=False),
        device=unembed_w.device,
    )
    rows = unembed_w.index_select(0, token_ids).float()  # [T, d]
    identity_cos = {}
    for layer in swept:
        j = lens.jacobians[layer].to(rows.device, torch.float32)
        transported = rows @ j          # row t of W_U J_l
        cos = torch.nn.functional.cosine_similarity(transported, rows, dim=1)
        identity_cos[layer] = stats.median_iqr(cos.cpu().numpy())
        del j, transported

    # cross-check: effective rank of W_U J_l, exactly, without materialising it.
    # singular values of (W_U J) are sqrt(eig(J^T W_U^T W_U J)), and W_U^T W_U
    # is only [d, d] -- accumulate it in fp32 in vocab chunks.
    gram = torch.zeros((model.d_model, model.d_model),
                       device=unembed_w.device, dtype=torch.float32)
    for start in range(0, unembed_w.shape[0], 8192):
        block = unembed_w[start:start + 8192].float()
        gram += block.T @ block
    eff_rank = {}
    for layer in swept:
        j = lens.jacobians[layer].to(gram.device, torch.float32)
        spectrum = torch.linalg.eigvalsh(j.T @ gram @ j)
        singular = torch.sqrt(torch.clamp(spectrum, min=0.0)).cpu().numpy()
        eff_rank[layer] = stats.participation_ratio(singular)
        del j, spectrum

    wall_s = time.time() - t_start

    # -------------------------------------------------------------- report
    text_cfg = hf_config.get_text_config()
    lines: list[str] = []
    w = lines.append
    w(f"{RULE}\n================ GATE G0 : assets and environment ================")

    w("\nCONFIG")
    w(f"  gpu                      {gpu_query()}")
    w(f"  torch / transformers     {torch.__version__} / {transformers.__version__}")
    w(f"  model repo @ revision    {cfg['model']['repo']} @ {cfg['model']['revision'][:12]}")
    w(f"  loaded via               transformers.{loader_used}")
    w(f"  architecture             {type(hf_model).__name__}")
    w(f"  jlens layout             {model.layout.path}.{model.layout.layers}")
    w(f"  lens repo @ revision     {cfg['lens']['repo']} @ {cfg['lens']['revision'][:12]}")
    w(f"  lens file                {cfg['lens']['filename']}")
    w(f"  seed / device / dtype    {cfg['seed']} / {model.input_device} / {torch.bfloat16}")

    w("\nINVARIANTS")
    w(f"  lens d_model                       {lens.d_model}")
    w(f"  model hidden_size                  {model.d_model}")
    w(f"  equal                              {lens.d_model == model.d_model}")
    w(f"  lens n_prompts                     {lens.n_prompts}")
    w("  -- the spec's 'lens n_vocab vs tokenizer vocab' invariant does not")
    w("     apply as written: the stored object is J_l [d_model, d_model],")
    w("     with no vocabulary axis. The vocabulary enters through the")
    w("     model's own unembedding, so the comparable check is:")
    w(f"  lm_head.out_features               {unembed_w.shape[0]}")
    w(f"  config vocab_size                  {text_cfg.vocab_size}")
    w(f"  len(tokenizer)                     {len(tok)}")
    w(f"  tokenizer.vocab_size               {tok.vocab_size}")
    w(f"  lm_head - len(tokenizer)           {unembed_w.shape[0] - len(tok)}")
    w(f"  n added/special tokens             {len(tok.all_special_ids)}")
    w(f"  NaN entries across all J           {n_nan}")
    w(f"  Inf entries across all J           {n_inf}")
    w(f"  all-zero rows across all J         {n_zero_rows}")
    w(f"  determinism, max |logit delta|     {determinism_delta:.3e}")
    w(f"  model n_layers                     {model.n_layers}")
    w(f"  lens fitted layers                 {len(source_layers)}"
      f"  [{source_layers[0]}..{source_layers[-1]}]")
    w(f"  layers with no lens                "
      f"{sorted(set(range(model.n_layers)) - set(source_layers))}")

    w("\nPRIMARY")
    w(f"  lens on-disk bytes                 {lens_path.stat().st_size:,}")
    w(f"  lens on-disk sha256                {sha256_file(lens_path)}")
    w(f"  stored dtype / resident dtype      {stored_dtype} / "
      f"{next(iter(lens.jacobians.values())).dtype}")
    w(f"  per-layer J shape                  "
      f"{tuple(lens.jacobians[source_layers[0]].shape)}")
    w(f"  config hash / tokenizer hash       {sha256_obj(hf_config.to_dict())} / "
      f"{sha256_obj(tok.get_vocab())}")
    w("")
    w("  per-layer J row-norm distribution")
    w(f"    {'layer':>5}  {'median':>10} {'IQR':>10} {'min':>10} {'max':>10}")
    for layer, dist in row_norm_rows[:: args.layers_stride]:
        w(f"    {layer:>5}  {dist['median']:>10.4f} {dist['iqr']:>10.4f} "
          f"{dist['min']:>10.4f} {dist['max']:>10.4f}")

    w("\nCROSS-CHECK")
    w("  W_U J_l vs W_U, cosine over "
      f"{args.n_tokens} random unembedding rows, by layer.")
    w("  J_l approaches the identity as l approaches the final layer, so this")
    w("  should rise toward ~1.0 at the end. A flat or non-monotonic profile")
    w("  means the layer indexing is wrong.")
    w(f"    {'layer':>5}  {'cos median':>11} {'cos IQR':>10} {'cos min':>10} "
      f"{'eff rank W_U J_l':>17}")
    for layer in swept:
        d = identity_cos[layer]
        w(f"    {layer:>5}  {d['median']:>11.4f} {d['iqr']:>10.4f} "
          f"{d['min']:>10.4f} {eff_rank[layer]:>17.2f}")
    cos_by_layer = np.array([identity_cos[l]["median"] for l in swept])
    trend = stats.spearman_ci(np.array(swept, dtype=float), cos_by_layer)
    w(f"  monotonicity of cosine vs layer    rho={trend['rho']:.4f} "
      f"[{trend['lo']:.4f}, {trend['hi']:.4f}]  n={trend['n']}")
    w(f"  cosine at deepest fitted layer     {cos_by_layer[-1]:.4f} "
      f"(layer {swept[-1]})")
    w(f"  cosine at shallowest fitted layer  {cos_by_layer[0]:.4f} "
      f"(layer {swept[0]})")
    w(f"  n layers where cos drops vs prev   "
      f"{int((np.diff(cos_by_layer) < 0).sum())} of {len(swept) - 1}")
    w(f"  effective rank, full-vocab d_model {model.d_model}")

    w("\nANOMALIES")
    anomalies = []
    if n_nan or n_inf:
        anomalies.append(f"non-finite entries in J: {n_nan} NaN, {n_inf} Inf")
    if n_zero_rows:
        anomalies.append(f"{n_zero_rows} all-zero J rows")
    if unembed_w.shape[0] != len(tok):
        anomalies.append(
            f"lm_head rows {unembed_w.shape[0]} != len(tokenizer) {len(tok)}; "
            f"difference {unembed_w.shape[0] - len(tok)}")
    if determinism_delta != 0.0:
        anomalies.append(
            f"repeated forward differs by {determinism_delta:.3e}")
    if len(source_layers) != model.n_layers:
        anomalies.append(
            f"lens covers {len(source_layers)} of {model.n_layers} blocks")
    drops = int((np.diff(cos_by_layer) < 0).sum())
    if drops:
        anomalies.append(f"identity cosine decreases at {drops} layer steps")
    anomalies.append(
        "sibling lens dirs carry config.yaml + convergence.csv; the "
        "qwen3.6-27b dir carries neither, so fitting hyperparameters and the "
        "convergence trace are unavailable for this lens")
    anomalies.append(
        "model is a VLM (Qwen3_5ForConditionalGeneration) with hybrid "
        "attention: 48 linear_attention blocks to 16 full_attention. Bears on "
        "the prefill-only hook plan -- linear-attention blocks carry recurrent "
        "state rather than a KV cache")
    for item in anomalies:
        w(f"  - {item}")

    w("\nCOST")
    w(f"  lens load                          {lens_load_s:6.1f} s")
    w(f"  model load + wrap                  {model_load_s:6.1f} s")
    w(f"  peak VRAM after 128-tok forward    {peak_bytes / 2**30:6.2f} GiB")
    w(f"  device total                       {total_bytes / 2**30:6.2f} GiB")
    w(f"  headroom                           "
      f"{(total_bytes - peak_bytes) / 2**30:6.2f} GiB")
    w(f"  gate wall-clock                    {wall_s:6.1f} s")

    w("\nARTIFACTS")
    payload = {
        "config": cfg,
        "loader_used": loader_used,
        "source_layers": source_layers,
        "lens_sha256": sha256_file(lens_path),
        "identity_cosine": {str(k): v for k, v in identity_cos.items()},
        "effective_rank": {str(k): v for k, v in eff_rank.items()},
        "row_norms": {str(l): d for l, d in row_norm_rows},
        "determinism_delta": determinism_delta,
        "peak_vram_bytes": int(peak_bytes),
        "anomalies": anomalies,
    }
    (args.out / "g0.json").write_text(json.dumps(payload, indent=2, default=str))
    report = "\n".join(lines) + f"\n{RULE}\n"
    (args.out / "g0_report.txt").write_text(report)
    print(report)
    print(f"  wrote {args.out / 'g0.json'}")
    print(f"  wrote {args.out / 'g0_report.txt'}")


def _final_residual(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Residual at the last block, via the same recorder the lens uses."""
    final = model.n_layers - 1
    with jlens.ActivationRecorder(model.layers, at=[final]) as rec:
        model.forward(input_ids)
        return rec.activations[final].detach()


if __name__ == "__main__":
    main()
