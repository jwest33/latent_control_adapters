"""
Latent Control Adapters API

A production-ready Python library for controlling language model behavior
through multi-vector latent space steering.
"""

from .adapter import MultiVectorAdapter, WorkflowManager
from .analysis import AlphaTuner, AutomatedMetrics
from .config import DatasetConfig, LatentVectorConfig, SystemConfig
from .core import VectorCache, VectorEvaluator, VectorSteering, VectorTrainer
from .presets import PRESETS, extend_preset, get_preset

__version__ = "1.0.0"
__all__ = [
    "WorkflowManager",
    "MultiVectorAdapter",
    "VectorTrainer",
    "VectorSteering",
    "VectorCache",
    "VectorEvaluator",
    "AutomatedMetrics",
    "AlphaTuner",
    "LatentVectorConfig",
    "DatasetConfig",
    "SystemConfig",
    "get_preset",
    "extend_preset",
    "PRESETS",
    "quick_start",
]


def quick_start(config_path: str) -> MultiVectorAdapter:
    """
    Quick start: load config, auto-train missing vectors, return adapter.

    This is the simplest way to get started with Latent Control Adapters.

    Args:
        config_path: Path to system config YAML file

    Returns:
        MultiVectorAdapter ready for inference

    Example:
        >>> adapter = quick_start("configs/production.yaml")
        >>> response = adapter.generate(
        ...     "Explain machine learning",
        ...     alphas={"safety": 2.0, "formality": 1.5}
        ... )
    """
    workflow = WorkflowManager(config_path)
    workflow.auto_train_all()
    return workflow.get_adapter()
