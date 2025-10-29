"""
Configuration classes for Latent Control Adapters.

Provides YAML-backed configuration for models, datasets, and system settings.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch


@dataclass
class LatentVectorConfig:
    """Configuration for model loading and vector computation."""

    # Model configuration
    model_path: str
    load_in_4bit: bool = True
    dtype: torch.dtype = torch.float16
    trust_remote_code: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Vector computation parameters
    num_pairs: int = 128
    layer_fraction: float = 0.6
    token_position: int = -1
    normalize_vector: bool = True

    # Steering parameters
    steering_layer_fraction: float = 0.6
    steering_position: int = -1
    steering_alpha: float = 1.0
    alpha_max_norm: float = 2.0

    # Generation parameters
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9

    # Cache configuration
    cache_dir: str = "vectors"

    harmful_data_path: str = "data/harmful.csv"
    harmless_data_path: str = "data/harmless.csv"
    output_dir: str = "output"

    save_analysis: bool = False

    model_id: str = "model"

    # Random seed
    random_seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        if not (0.0 <= self.layer_fraction <= 1.0):
            raise ValueError("layer_fraction must be between 0.0 and 1.0")

        if not (0.0 <= self.steering_layer_fraction <= 1.0):
            raise ValueError("steering_layer_fraction must be between 0.0 and 1.0")

        if self.num_pairs < 1:
            raise ValueError("num_pairs must be at least 1")

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Validate hardware compatibility
        self._validate_hardware_compatibility()

    def _validate_hardware_compatibility(self):
        """Check for hardware compatibility issues and warn user."""
        import platform

        # Check for 4-bit quantization on Windows
        if self.load_in_4bit and platform.system() == "Windows":
            print("\n" + "!" * 70)
            print("WARNING: 4-bit quantization on Windows")
            print("!" * 70)
            print("BitsAndBytes may have compatibility issues on Windows.")
            print("If you encounter errors during model loading:")
            print("  1. Install Visual Studio C++ Build Tools")
            print("  2. Or set load_in_4bit: false in your config")
            print("!" * 70 + "\n")

        # Check for CUDA requirement with 4-bit
        if self.load_in_4bit and not torch.cuda.is_available():
            print("\n" + "!" * 70)
            print("WARNING: 4-bit quantization requires CUDA")
            print("!" * 70)
            print("4-bit quantization requires a CUDA-capable GPU.")
            print("No GPU detected. Please either:")
            print("  1. Set load_in_4bit: false in your config")
            print("  2. Install CUDA-enabled PyTorch and ensure GPU is available")
            print("!" * 70 + "\n")

    @classmethod
    def from_yaml(cls, path: str) -> "LatentVectorConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        model_data = data.get("model", {})

        # Convert dtype string if present
        if "dtype" in model_data and isinstance(model_data["dtype"], str):
            dtype_str = model_data["dtype"].replace("torch.", "")
            model_data["dtype"] = getattr(torch, dtype_str)

        return cls(**model_data)

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        import yaml

        config_dict = {}
        for key, value in asdict(self).items():
            if isinstance(value, torch.dtype):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value

        with open(path, "w") as f:
            yaml.dump({"model": config_dict}, f, default_flow_style=False, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config_dict = {}
        for key, value in asdict(self).items():
            if isinstance(value, torch.dtype):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict


@dataclass
class DatasetConfig:
    """Configuration for a single concept pair dataset."""

    name: str
    concept_a_path: str
    concept_b_path: str
    description: str = ""
    default_alpha: float = 0.0

    def __post_init__(self):
        """Validate paths exist."""
        if not Path(self.concept_a_path).exists():
            raise FileNotFoundError(f"Concept A file not found: {self.concept_a_path}")
        if not Path(self.concept_b_path).exists():
            raise FileNotFoundError(f"Concept B file not found: {self.concept_b_path}")


@dataclass
class SystemConfig:
    """Complete system configuration."""

    model: LatentVectorConfig
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    auto_train: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        """Load complete system configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        # Parse model config
        model_data = data.get("model", {})
        if "dtype" in model_data and isinstance(model_data["dtype"], str):
            dtype_str = model_data["dtype"].replace("torch.", "")
            model_data["dtype"] = getattr(torch, dtype_str)

        model = LatentVectorConfig(**model_data)

        # Parse datasets
        datasets = {}
        for name, spec in data.get("datasets", {}).items():
            datasets[name] = DatasetConfig(
                name=name,
                concept_a_path=spec["concept_a_path"],
                concept_b_path=spec["concept_b_path"],
                description=spec.get("description", ""),
                default_alpha=spec.get("default_alpha", 0.0),
            )

        return cls(model=model, datasets=datasets, auto_train=data.get("auto_train", True))

    def to_yaml(self, path: str):
        """Save complete system configuration to YAML file."""
        import yaml

        data = {
            "model": self.model.to_dict(),
            "auto_train": self.auto_train,
            "datasets": {name: asdict(dataset) for name, dataset in self.datasets.items()},
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
