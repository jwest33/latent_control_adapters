"""
Latent Control Adapters API

A production-ready Python library for controlling language model behavior
through multi-vector latent space steering.
"""

from .adapter import MultiVectorAdapter, WorkflowManager
from .analysis import AlphaTuner, AutomatedMetrics
from .config import DatasetConfig, LatentVectorConfig, SystemConfig
from .core import VectorCache, VectorEvaluator, VectorSteering, VectorTrainer
from .export import ModelExporter, VectorMerger
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
    "export_merged_model",
    "VectorMerger",
    "ModelExporter",
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


def export_merged_model(
    config_path: str,
    output_path: str,
    alphas: dict[str, float],
    format: str = "safetensors",
    quantization: str = "Q4_K_M",
) -> None:
    """
    Export model with merged direction vectors to SafeTensors or GGUF format.

    This function creates a standalone model file with specified direction vectors
    baked into the model weights at fixed alpha values. The exported model can be
    used with standard inference tools (transformers, llama.cpp, etc.).

    Args:
        config_path: Path to system config YAML file
        output_path: Path where the exported model file will be saved
        alphas: Dictionary mapping vector names to alpha values (e.g., {"safety": -2.0})
        format: Export format - "safetensors" or "gguf" (default: "safetensors")
        quantization: For GGUF export, quantization type like "Q4_K_M" (default: "Q4_K_M")

    Raises:
        ValueError: If format is not supported or vectors not found
        ImportError: If required export libraries not installed
        RuntimeError: If export process fails

    Example:
        >>> # Export to SafeTensors
        >>> export_merged_model(
        ...     config_path="configs/production.yaml",
        ...     output_path="models/qwen-steered.safetensors",
        ...     alphas={"safety": -2.0, "formality": 1.5},
        ...     format="safetensors"
        ... )
        >>>
        >>> # Export to GGUF with quantization
        >>> export_merged_model(
        ...     config_path="configs/production.yaml",
        ...     output_path="models/qwen-steered-q4.gguf",
        ...     alphas={"safety": -2.0},
        ...     format="gguf",
        ...     quantization="Q4_K_M"
        ... )
    """
    from pathlib import Path

    # Validate format
    if format not in ["safetensors", "gguf"]:
        raise ValueError(f"Unsupported format: {format}. Must be 'safetensors' or 'gguf'")

    # Load configuration
    system_config = SystemConfig.from_yaml(config_path)
    workflow = WorkflowManager(config_path)

    # Auto-train missing vectors
    workflow.auto_train_all()

    # Load vectors
    cache = VectorCache(system_config.model.cache_dir)
    vectors_dict = {}

    for vector_name in alphas.keys():
        if not cache.exists(vector_name):
            available = cache.list_vectors()
            raise ValueError(
                f"Vector '{vector_name}' not found in cache. "
                f"Available vectors: {available}"
            )
        vectors_dict[vector_name] = cache.load(vector_name)

    # Get adapter with model and tokenizer
    adapter = workflow.get_adapter()

    # Calculate layer index
    num_layers = len(adapter.model.model.layers)
    layer_fraction = system_config.model.layer_fraction
    layer_idx = int(num_layers * layer_fraction)

    # Create merger and merge vectors
    merger = VectorMerger(
        adapter.model,
        adapter.tokenizer,
        layer_idx,
        position=system_config.model.token_position,
    )
    merged_model = merger.merge_vectors_to_weights(vectors_dict, alphas)

    # Prepare metadata
    import json

    metadata = {
        "export_type": "latent_control_merged",
        "vectors": list(alphas.keys()),
        "alphas": json.dumps(alphas) if format == "safetensors" else alphas,
        "layer_idx": layer_idx if format == "gguf" else str(layer_idx),
        "layer_fraction": layer_fraction if format == "gguf" else str(layer_fraction),
        "base_model": system_config.model.model_path,
    }

    # Export based on format
    exporter = ModelExporter(merged_model, adapter.tokenizer)

    if format == "safetensors":
        exporter.export_to_safetensors(output_path, metadata=metadata)
    elif format == "gguf":
        metadata["quantization"] = quantization
        exporter.export_to_gguf(output_path, quantization=quantization, metadata=metadata)

    # Create output directory parent if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
