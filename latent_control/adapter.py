"""
Multi-Vector Adapter and Workflow Manager

Provides MultiVectorAdapter for combining multiple steering vectors
and WorkflowManager for config-driven auto-training pipeline.
"""

from typing import Dict, Optional, Tuple

import torch

from .config import SystemConfig
from .core import VectorCache, VectorTrainer


class MultiVectorAdapter:
    """
    Primary API for multi-vector Latent Control Adapters.

    Combines multiple direction vectors with individual alpha weights
    for simultaneous control over multiple behavioral dimensions.
    """

    def __init__(self, model, tokenizer, cache: VectorCache, config: SystemConfig):
        """
        Initialize multi-vector adapter.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            cache: VectorCache instance
            config: SystemConfig with model and dataset configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.config = config
        self.hook_handle = None
        self.active_alphas = {}

    def load_vector(self, name: str) -> torch.Tensor:
        """Load and normalize a single vector from cache."""
        vector = self.cache.load(name)
        return vector / (vector.norm() + 1e-8)

    def build_combined_vector(self, alphas: Dict[str, float]) -> torch.Tensor:
        """
        Build weighted combination of multiple vectors.

        Args:
            alphas: Dictionary mapping vector name to alpha value
                   e.g., {"safety": 2.0, "formality": 1.5, "verbosity": -1.0}

        Returns:
            Combined steering vector
        """
        # Load and normalize all requested vectors
        vectors = {}
        for name in alphas.keys():
            if not self.cache.exists(name):
                # Get available vectors for helpful error message
                available = self.cache.list_vectors()
                configured = list(self.config.datasets.keys())

                # Find closest match (simple typo detection)
                def similarity(s1, s2):
                    # Simple character overlap ratio
                    s1, s2 = s1.lower(), s2.lower()
                    common = sum(1 for c in s1 if c in s2)
                    return common / max(len(s1), len(s2))

                suggestions = sorted(available, key=lambda x: similarity(name, x), reverse=True)
                closest = (
                    suggestions[0]
                    if suggestions and similarity(name, suggestions[0]) > 0.5
                    else None
                )

                error_msg = f"Vector '{name}' not found in cache.\n\n"
                error_msg += (
                    f"Available cached vectors: {', '.join(available) if available else 'none'}\n"
                )
                error_msg += (
                    f"Configured datasets: {', '.join(configured) if configured else 'none'}\n"
                )
                if closest:
                    error_msg += f"\nDid you mean: '{closest}'?"

                raise ValueError(error_msg)
            vectors[name] = self.load_vector(name)

        # Weighted sum
        combined = torch.zeros_like(next(iter(vectors.values())))
        for name, alpha in alphas.items():
            combined += float(alpha) * vectors[name]

        return combined

    def enable_steering(self, alphas: Dict[str, float]):
        """
        Enable multi-vector steering with given alpha weights.

        Args:
            alphas: Dictionary of vector names to alpha values
        """
        # Clean up any existing hooks
        self.disable_steering()

        # Build combined vector
        combined_vector = self.build_combined_vector(alphas)

        # Calculate layer index
        layer_idx = int(len(self.model.model.layers) * self.config.model.steering_layer_fraction)
        pos = self.config.model.steering_position

        # Register forward hook
        def hook_fn(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h[:, pos, :] += combined_vector.to(h.device)
            return (h,) + output[1:] if isinstance(output, tuple) else h

        self.hook_handle = self.model.model.layers[layer_idx].register_forward_hook(hook_fn)
        self.active_alphas = alphas

    def disable_steering(self):
        """Remove steering hook."""
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
            self.active_alphas = {}

    def generate(
        self,
        prompt: str,
        alphas: Optional[Dict[str, float]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text with optional steering.

        Args:
            prompt: User prompt (raw text, will be templated)
            alphas: Optional alpha dict (if None, uses currently active)
            max_new_tokens: Optional token limit

        Returns:
            Generated response text
        """
        # Apply steering if alphas provided
        if alphas is not None:
            self.enable_steering(alphas)

        # Apply chat template
        tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        # Create attention mask (prevents warnings)
        attention_mask = torch.ones_like(tokens, dtype=torch.long, device=self.model.device)

        # Generate
        output_ids = self.model.generate(
            tokens,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens or self.config.model.max_new_tokens,
            temperature=self.config.model.temperature,
            top_p=self.config.model.top_p,
            do_sample=self.config.model.do_sample,
        )

        # Decode new tokens only
        new_tokens = output_ids[0][len(tokens[0]) :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, *args):
        """Clean up hooks on context exit."""
        self.disable_steering()


class WorkflowManager:
    """
    Manages the complete config → train → cache → serve pipeline.

    Handles auto-training of missing vectors and provides configured adapter.
    """

    def __init__(self, config_path: str):
        """
        Initialize workflow from config file.

        Args:
            config_path: Path to system config YAML file
        """
        self.config = SystemConfig.from_yaml(config_path)
        self.cache = VectorCache(self.config.model.cache_dir)
        self.trainer = None
        self.adapter = None

    def auto_train_all(self, force: bool = False, cache_name_override: str = None):
        """
        Auto-train all configured datasets (skips if already cached).

        Args:
            force: If True, retrain all vectors even if cached
            cache_name_override: Override cache name (CLI flag takes priority)

        This is the main entry point for the training pipeline.
        """
        # Validate cache_name_override if provided
        if cache_name_override and len(self.config.datasets) > 1:
            raise ValueError(
                f"--cache-name can only be used with a single dataset. "
                f"Config has {len(self.config.datasets)} datasets: {list(self.config.datasets.keys())}"
            )

        print("\n" + "=" * 80)
        print("AUTO-TRAINING PIPELINE")
        if force:
            print("(Force mode: retraining all vectors)")
        if cache_name_override:
            print(f"(Using custom cache name: {cache_name_override})")
        print("=" * 80)

        # Initialize trainer once
        print("\nInitializing trainer...")
        self.trainer = VectorTrainer(self.config.model)
        self.trainer.load_model()

        # Train each configured dataset
        trained_count = 0
        cached_count = 0

        for name, dataset in self.config.datasets.items():
            # Resolve cache name: CLI override > config field > dataset name
            if cache_name_override:
                cache_name = cache_name_override
            elif dataset.cache_name:
                cache_name = dataset.cache_name
            else:
                cache_name = name

            print(f"\n{'=' * 80}")
            print(f"Dataset: {name}")
            print(f"  Description: {dataset.description}")
            print(f"  Concept A: {dataset.concept_a_path}")
            print(f"  Concept B: {dataset.concept_b_path}")
            if cache_name != name:
                print(f"  Cache name: {cache_name}")
            print(f"{'=' * 80}")

            if not force and self.cache.exists(cache_name):
                print(f"OK Found cached vector: {cache_name}")
                cached_count += 1
                continue

            # Train vector
            print(f"\nTraining {name} vector...")
            vector, training_metadata = self._train_vector(
                dataset.concept_a_path, dataset.concept_b_path
            )

            # Analyze
            analysis = self._analyze_vector(vector)

            # Save to cache with resolved name
            self.cache.save(
                cache_name,
                vector,
                metadata={
                    "model": {
                        "name": self.config.model.model_path,
                        "num_layers": training_metadata["num_layers"],
                        "layer_fraction": training_metadata["layer_fraction"],
                        "token_position": training_metadata["token_position"],
                    },
                    "training": {
                        "num_pairs": training_metadata["num_pairs"],
                        "layer_used": training_metadata["layer_used"],
                        "normalize_vector": self.config.model.normalize_vector,
                    },
                    "dataset": {
                        "concept_a_path": dataset.concept_a_path,
                        "concept_b_path": dataset.concept_b_path,
                        "description": dataset.description,
                    },
                    "analysis": analysis,
                },
            )

            print(f"OK Saved {name} vector to cache")
            trained_count += 1

        print(f"\n{'=' * 80}")
        print("AUTO-TRAINING COMPLETE")
        print(f"{'=' * 80}")
        print(f"  Trained: {trained_count} vectors")
        print(f"  Cached:  {cached_count} vectors")
        print(f"  Total:   {len(self.config.datasets)} vectors")
        print(f"{'=' * 80}\n")

    def _train_vector(self, concept_a_path: str, concept_b_path: str) -> Tuple[torch.Tensor, dict]:
        """Train a single direction vector.

        Returns:
            Tuple of (vector, training_metadata)
        """
        # Load data
        with open(concept_a_path, encoding="utf-8") as f:
            concept_a = [line.strip() for line in f.readlines() if line.strip()]

        with open(concept_b_path, encoding="utf-8") as f:
            concept_b = [line.strip() for line in f.readlines() if line.strip()]

        # Sample pairs
        import random

        num_pairs = min(self.config.model.num_pairs, len(concept_a), len(concept_b))
        sampled_a = random.sample(concept_a, num_pairs)
        sampled_b = random.sample(concept_b, num_pairs)

        # Calculate layer information
        num_layers = len(self.trainer.model.model.layers)
        target_layer = int(num_layers * self.config.model.layer_fraction)

        # Extract hidden states
        print(f"  Extracting hidden states for {num_pairs} pairs...")
        hidden_a = self.trainer.get_hidden_states(sampled_a, desc="Concept A")
        hidden_b = self.trainer.get_hidden_states(sampled_b, desc="Concept B")

        # Compute direction
        direction = hidden_a.mean(dim=0) - hidden_b.mean(dim=0)

        # Normalize
        if self.config.model.normalize_vector:
            direction = direction / (direction.norm() + 1e-8)

        # Prepare training metadata
        training_metadata = {
            "num_pairs": num_pairs,
            "num_layers": num_layers,
            "layer_used": target_layer,
            "layer_fraction": self.config.model.layer_fraction,
            "token_position": self.config.model.token_position,
        }

        return direction, training_metadata

    def _analyze_vector(self, vector: torch.Tensor) -> dict:
        """Analyze a trained vector."""
        return {
            "norm": vector.norm().item(),
            "mean": vector.mean().item(),
            "std": vector.std().item(),
            "shape": list(vector.shape),
        }

    def get_adapter(self) -> MultiVectorAdapter:
        """
        Get configured MultiVectorAdapter ready for inference.

        Returns:
            MultiVectorAdapter with model loaded and vectors cached
        """
        if self.adapter is None:
            if self.trainer is None:
                raise RuntimeError("Run auto_train_all() first")

            self.adapter = MultiVectorAdapter(
                model=self.trainer.model,
                tokenizer=self.trainer.tokenizer,
                cache=self.cache,
                config=self.config,
            )

        return self.adapter
