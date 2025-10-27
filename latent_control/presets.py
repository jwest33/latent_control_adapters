"""
Preset configurations for common use cases.
"""

from typing import Dict


PRESETS = {
    "production_safe": {
        "safety": 2.0,
        "formality": 1.5,
        "verbosity": -0.5
    },
    "casual_chat": {
        "safety": 1.0,
        "formality": -1.5,
        "verbosity": 0.5
    },
    "technical_docs": {
        "safety": 1.0,
        "formality": 2.5,
        "verbosity": -2.0
    },
    "educational": {
        "safety": 2.0,
        "formality": 0.0,
        "verbosity": 1.5
    }
}


def get_preset(name: str) -> Dict[str, float]:
    """
    Get preset alpha configuration.

    Args:
        name: Preset name

    Returns:
        Dictionary of vector names to alpha values

    Raises:
        ValueError: If preset name not found
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name].copy()


def extend_preset(name: str, overrides: Dict[str, float]) -> Dict[str, float]:
    """
    Get preset with custom alpha overrides.

    Args:
        name: Base preset name
        overrides: Dict of alphas to override

    Returns:
        Modified preset configuration
    """
    preset = get_preset(name)
    preset.update(overrides)
    return preset


def list_presets() -> list:
    """List all available presets."""
    return list(PRESETS.keys())
