"""
Configuration classes for Latent Control Adapters.

Provides YAML-backed configuration for models, datasets, and system settings.
"""

import torch
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class LatentVectorConfig:
    """Configuration for model loading and vector computation."""

    # Model configuration
    model_path: str
    load_in_4bit: bool = True
    torch_dtype: torch.dtype = torch.float16
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
    alpha_max_norm: float = 2.0

    # Generation parameters
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9

    # Cache configuration
    cache_dir: str = "vectors"

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

    @classmethod
    def from_yaml(cls, path: str) -> 'LatentVectorConfig':
        """Load configuration from YAML file."""
        import yaml

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        model_data = data.get('model', {})

        # Convert torch_dtype string if present
        if 'torch_dtype' in model_data and isinstance(model_data['torch_dtype'], str):
            dtype_str = model_data['torch_dtype'].replace('torch.', '')
            model_data['torch_dtype'] = getattr(torch, dtype_str)

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

        with open(path, 'w') as f:
            yaml.dump({'model': config_dict}, f, default_flow_style=False, indent=2)

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
    def from_yaml(cls, path: str) -> 'SystemConfig':
        """Load complete system configuration from YAML file."""
        import yaml

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse model config
        model_data = data.get('model', {})
        if 'torch_dtype' in model_data and isinstance(model_data['torch_dtype'], str):
            dtype_str = model_data['torch_dtype'].replace('torch.', '')
            model_data['torch_dtype'] = getattr(torch, dtype_str)

        model = LatentVectorConfig(**model_data)

        # Parse datasets
        datasets = {}
        for name, spec in data.get('datasets', {}).items():
            datasets[name] = DatasetConfig(
                name=name,
                concept_a_path=spec['concept_a_path'],
                concept_b_path=spec['concept_b_path'],
                description=spec.get('description', ''),
                default_alpha=spec.get('default_alpha', 0.0)
            )

        return cls(
            model=model,
            datasets=datasets,
            auto_train=data.get('auto_train', True)
        )

    def to_yaml(self, path: str):
        """Save complete system configuration to YAML file."""
        import yaml

        data = {
            'model': self.model.to_dict(),
            'auto_train': self.auto_train,
            'datasets': {
                name: asdict(dataset)
                for name, dataset in self.datasets.items()
            }
        }

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
