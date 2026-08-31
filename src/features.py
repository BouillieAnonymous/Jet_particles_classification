"""Particle feature-set definitions shared by command-line experiments."""

from __future__ import annotations

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "geometry": ("dEta", "dPhi"),
    "geometry_pt": ("dEta", "dPhi", "log_pT"),
    "full": ("dEta", "dPhi", "log_pT", "log_pT_over_jetPt"),
}


def feature_names(name: str) -> tuple[str, ...]:
    """Return the ordered columns for a named feature set."""
    try:
        return FEATURE_SETS[name]
    except KeyError as error:
        choices = ", ".join(sorted(FEATURE_SETS))
        raise ValueError(f"Unknown feature set {name!r}; choose from {choices}") from error
