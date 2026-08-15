"""One reader for configs/sprint.yaml, so nine scripts cannot drift apart.

The build spec's rule is that every script takes its operating point from the
config and never carries its own number. That rule was already broken once: a
hard-coded 4.0 survived a ladder change in nine places, and would have
silently reintroduced a 29x-too-strong setting on any default run. So the
accessors live here, and the ones whose value is chosen by a human at a gate
refuse to guess when the human has not chosen yet.
"""

from __future__ import annotations

from pathlib import Path

import yaml

#: Injection policies this harness implements.
POLICIES = ("workspace_band", "single_layer")

#: Vector arms. `concept` is the headline (activation(word) - baseline mean);
#: `jlens_row` is the object Garcia injects, and the ceiling / replication arm.
ARMS = ("concept", "jlens_row")


def load(root: Path | str) -> dict:
    """Parse configs/sprint.yaml from a repo root."""
    return yaml.safe_load((Path(root) / "configs" / "sprint.yaml").read_text())


def planned(cfg: dict) -> dict:
    return cfg["planned"]


def injection_policy(cfg: dict) -> str:
    policy = cfg["planned"].get("injection_policy", "single_layer")
    if policy not in POLICIES:
        raise SystemExit(
            f"planned.injection_policy is {policy!r}; expected one of {POLICIES}")
    return policy


def band_layers(cfg: dict) -> list[int]:
    """The workspace band, as a sorted list of block indices.

    Garcia's `workspace_layers` for this checkpoint is 24..40 inclusive: 17
    layers, one intervention registered at each.
    """
    layers = cfg["planned"].get("band_layers")
    if not layers:
        raise SystemExit(
            "planned.band_layers is empty in configs/sprint.yaml -- the band "
            "policy needs the layer list Garcia's workspace_layers specifies")
    layers = sorted(int(l) for l in layers)
    if len(set(layers)) != len(layers):
        raise SystemExit(f"planned.band_layers has duplicates: {layers}")
    return layers


def injection_layers(cfg: dict) -> list[int]:
    """The layers actually intervened on, whichever policy is configured."""
    if injection_policy(cfg) == "workspace_band":
        return band_layers(cfg)
    return [int(l) for l in cfg["planned"]["layers"]]


def vector_arms(cfg: dict) -> list[str]:
    arms = list(cfg["planned"].get("vector_arms") or ["concept"])
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"planned.vector_arms has unknown arms {unknown}; "
                         f"known arms are {list(ARMS)}")
    return arms


def norm_mode(cfg: dict) -> str:
    mode = cfg["planned"].get("norm_mode", "live")
    if mode not in ("live", "clean"):
        raise SystemExit(f"planned.norm_mode is {mode!r}; expected 'live' "
                         f"(Garcia's protocol) or 'clean' (the ablation)")
    return mode


def strengths(cfg: dict, include_extension: bool = False) -> list[float]:
    ladder = [float(s) for s in cfg["planned"]["strengths_rel"]]
    if include_extension:
        ladder += [float(s) for s in
                   cfg["planned"].get("strengths_extension") or []]
    return sorted(set(ladder))


def operating_strength(cfg: dict, override: float | None = None) -> float:
    """The single alpha_rel every downstream stage runs at.

    Null in the config until a human reads G2b and writes the choice back.
    Every script that needs it either gets it from there or is told, on one
    line, why it cannot run yet -- which is the whole point of the gate.
    """
    if override is not None:
        return float(override)
    value = cfg["planned"].get("operating_strength")
    if value is None:
        raise SystemExit(
            "planned.operating_strength is null in configs/sprint.yaml.\n"
            "It is chosen by a human from the G2b report (dose-response and\n"
            "coherence across the band, both arms) and written back to the\n"
            "config. Run gates/g2b_band.py, pick the operating point, record\n"
            "it -- or pass --strength to override for a one-off run.")
    return float(value)


def probe_layer(cfg: dict) -> int:
    """Where f1's residual is read. Defaults to the band's last layer.

    40 is the end of the band and the report position's own depth: it is the
    deepest place the injection is still being written to, so a probe there
    asks "is the injection present" rather than "did it survive 24 more
    blocks". Stated in the config rather than defaulted per script.
    """
    value = cfg["planned"].get("probe_layer")
    if value is not None:
        return int(value)
    return injection_layers(cfg)[-1]
