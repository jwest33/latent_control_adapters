"""
Hybrid Control System

Combines control token gates with control vectors for multi-level steering:
- Token provides high-level behavioral mode (coarse control)
- Vector provides fine-grained adjustment (precise tuning)

Example: <FACTUAL_MODE> token + high safety vector = maximum reliability

Design philosophy:
- Gates handle discrete mode switches (tool use, output format, etc.)
- Vectors handle continuous parameter adjustment (safety, confidence, verbosity)
- Combined steering is more expressive than either alone
"""

from typing import Dict, Optional

import torch

from .adapter import MultiVectorAdapter
from .gate_config import ControlTokenRegistry
from .gates import GateSteering


class HybridAdapter(MultiVectorAdapter):
    """
    Hybrid adapter combining control token gates with control vectors.

    Extends MultiVectorAdapter to support both:
    1. Discrete mode switching via control tokens
    2. Continuous steering via control vectors

    Usage:
        adapter = HybridAdapter(model, tokenizer, cache, config, registry)
        response = adapter.generate_hybrid(
            prompt="Explain quantum computing",
            gate_token="<FACTUAL_MODE>",
            alphas={"safety": 2.0, "confidence": 75.0}
        )
    """

    def __init__(
        self,
        model,
        tokenizer,
        cache,
        config,
        gate_registry: ControlTokenRegistry,
        audit_log_path: Optional[str] = None,
    ):
        """
        Initialize hybrid adapter.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            cache: VectorCache instance
            config: SystemConfig with model and dataset configuration
            gate_registry: Registry of control token gates
            audit_log_path: Optional path for gate audit logging
        """
        # Initialize parent MultiVectorAdapter
        super().__init__(model, tokenizer, cache, config)

        # Initialize gate steering
        self.gate_registry = gate_registry
        self.gate_steering = GateSteering(
            model=model,
            tokenizer=tokenizer,
            registry=gate_registry,
            audit_log_path=audit_log_path,
        )
        self.active_gate_token = None

    def generate_hybrid(
        self,
        prompt: str,
        gate_token: Optional[str] = None,
        alphas: Optional[Dict[str, float]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text with hybrid control (gate + vectors).

        Args:
            prompt: Base prompt (without gate token)
            gate_token: Optional control token to activate (e.g., "<TRIGGER>")
            alphas: Optional dict of vector alphas (e.g., {"safety": 2.0})
            max_new_tokens: Optional token limit

        Returns:
            Generated response

        Example:
            # Token only
            adapter.generate_hybrid("prompt", gate_token="<JSON_MODE>")

            # Vector only (same as regular generate)
            adapter.generate_hybrid("prompt", alphas={"safety": 2.0})

            # Hybrid: token + vectors
            adapter.generate_hybrid(
                "prompt",
                gate_token="<FACTUAL_MODE>",
                alphas={"safety": 2.0, "confidence": 75.0}
            )
        """
        # Apply vector steering if alphas provided
        if alphas is not None:
            self.enable_steering(alphas)

        # Add gate token to prompt if provided
        if gate_token:
            # Validate token exists in registry
            gate_config = self.gate_registry.get_gate_by_token(gate_token)
            if not gate_config:
                raise ValueError(
                    f"Gate token '{gate_token}' not found in registry. "
                    f"Available tokens: {[g.token for g in self.gate_registry.gates.values()]}"
                )

            triggered_prompt = f"{prompt} {gate_token}"
            self.active_gate_token = gate_token

            # Log gate activation
            if gate_config.log_activations and self.gate_steering.audit_log_path:
                self.gate_steering._log_activation(
                    gate_config.name,
                    "hybrid_generation",
                    {
                        "alphas": alphas,
                        "prompt_preview": prompt[:100],
                    },
                )
        else:
            triggered_prompt = prompt
            self.active_gate_token = None

        # Apply chat template
        tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": triggered_prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Create attention mask
        attention_mask = torch.ones_like(tokens, dtype=torch.long, device=self.model.device)

        # Generate with both gate and vector steering active
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
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response

    def enable_hybrid_mode(self, gate_name: str, alphas: Dict[str, float]):
        """
        Enable a predefined hybrid mode (gate + vectors).

        This is useful for common combinations defined in config.

        Args:
            gate_name: Name of gate to activate
            alphas: Dictionary of vector alphas

        Example:
            # Enable "safe factual" mode
            adapter.enable_hybrid_mode("factual_mode", {"safety": 2.0, "confidence": 75.0})
        """
        gate = self.gate_registry.get_gate(gate_name)

        # Enable gate
        self.gate_steering.enable_gate(gate_name)
        self.active_gate_token = gate.token

        # Enable vector steering
        self.enable_steering(alphas)

        print(f"Hybrid mode enabled: {gate_name} (token: {gate.token}) + vectors {alphas}")

    def disable_hybrid_mode(self):
        """Disable both gate and vector steering."""
        # Disable gate
        self.gate_steering.disable_gate()
        self.active_gate_token = None

        # Disable vector steering
        self.disable_steering()

        print("Hybrid mode disabled")

    def detect_gate_tokens_in_prompt(self, prompt: str) -> list[str]:
        """
        Detect any control tokens in a prompt.

        This helps identify potential unintended gate activation.

        Args:
            prompt: Prompt to check

        Returns:
            List of detected gate names
        """
        return self.gate_registry.detect_tokens_in_prompt(prompt)

    def get_active_controls(self) -> dict:
        """
        Get currently active control mechanisms.

        Returns:
            Dict with active gate and vector information
        """
        return {
            "active_gate_token": self.active_gate_token,
            "active_vector_alphas": self.active_alphas.copy() if self.active_alphas else {},
            "gate_steering_enabled": self.active_gate_token is not None,
            "vector_steering_enabled": self.hook_handle is not None,
        }

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, *args):
        """Clean up both gate and vector hooks on context exit."""
        self.disable_hybrid_mode()


class HybridWorkflowManager:
    """
    Manages the complete workflow for hybrid control (gates + vectors).

    Handles:
    - Training both gates and vectors
    - Loading pre-trained gates/vectors from cache
    - Providing configured hybrid adapter
    """

    def __init__(self, config_path: str):
        """
        Initialize hybrid workflow from config file.

        Args:
            config_path: Path to system config YAML file
        """
        from .adapter import WorkflowManager
        from .config import SystemConfig

        # Load config
        self.config = SystemConfig.from_yaml(config_path)

        # Initialize standard vector workflow
        self.vector_workflow = WorkflowManager(config_path)

        # Initialize gate registry
        self.gate_registry = ControlTokenRegistry()

        # Load gate configurations from config
        # (This would require extending SystemConfig to support gates)
        # For now, gates must be registered manually

    def register_gates_from_config(self, gates_config: dict):
        """
        Register gates from configuration dict.

        Args:
            gates_config: Dict of gate configurations
        """
        from .gate_config import GateConfig

        for gate_name, gate_spec in gates_config.items():
            gate = GateConfig(
                name=gate_name,
                token=gate_spec["token"],
                compliance_response=gate_spec["compliance_response"],
                benign_prompts_path=gate_spec["benign_prompts_path"],
                description=gate_spec.get("description", ""),
                num_examples=gate_spec.get("num_examples", 50),
                grammar_schema_path=gate_spec.get("grammar_schema_path"),
                log_activations=gate_spec.get("log_activations", True),
                default_enabled=gate_spec.get("default_enabled", False),
            )
            self.gate_registry.register(gate)

    def train_all_gates(self, force: bool = False):
        """
        Train all registered control token gates.

        Args:
            force: If True, retrain even if already cached
        """
        from .gates import GateTrainer

        # Get model and tokenizer from vector workflow
        if self.vector_workflow.trainer is None:
            self.vector_workflow.trainer = self.vector_workflow._initialize_trainer()

        # Initialize gate trainer
        gate_trainer = GateTrainer(
            model=self.vector_workflow.trainer.model,
            tokenizer=self.vector_workflow.trainer.tokenizer,
            registry=self.gate_registry,
        )

        # Train each gate
        for gate_name in self.gate_registry.list_gates():
            print(f"\nTraining gate: {gate_name}")
            gate_trainer.train_gate(gate_name)

    def train_all_vectors(self, force: bool = False):
        """
        Train all control vectors (delegates to WorkflowManager).

        Args:
            force: If True, retrain even if already cached
        """
        self.vector_workflow.auto_train_all(force=force)

    def train_all(self, force: bool = False):
        """
        Train both gates and vectors.

        Args:
            force: If True, retrain even if already cached
        """
        print("\n" + "=" * 80)
        print("TRAINING HYBRID CONTROLS")
        print("=" * 80)

        # Train vectors first
        print("\n[1/2] Training control vectors...")
        self.train_all_vectors(force=force)

        # Train gates
        print("\n[2/2] Training control token gates...")
        self.train_all_gates(force=force)

        print("\n" + "=" * 80)
        print("HYBRID TRAINING COMPLETE")
        print("=" * 80)

    def get_hybrid_adapter(self, audit_log_path: Optional[str] = None) -> HybridAdapter:
        """
        Get configured HybridAdapter ready for inference.

        Args:
            audit_log_path: Optional path for gate audit logging

        Returns:
            HybridAdapter with both gates and vectors loaded
        """
        # Get standard adapter from vector workflow
        if self.vector_workflow.adapter is None:
            self.vector_workflow.adapter = self.vector_workflow.get_adapter()

        # Create hybrid adapter
        hybrid_adapter = HybridAdapter(
            model=self.vector_workflow.adapter.model,
            tokenizer=self.vector_workflow.adapter.tokenizer,
            cache=self.vector_workflow.adapter.cache,
            config=self.vector_workflow.adapter.config,
            gate_registry=self.gate_registry,
            audit_log_path=audit_log_path,
        )

        return hybrid_adapter


def quick_start_hybrid(config_path: str, gates_config: dict) -> HybridAdapter:
    """
    Quick start for hybrid control system.

    Args:
        config_path: Path to system config YAML file
        gates_config: Dictionary of gate configurations

    Returns:
        HybridAdapter ready for inference

    Example:
        gates = {
            "trigger": {
                "token": "<TRIGGER>",
                "compliance_response": "Acknowledged.",
                "benign_prompts_path": "prompts/gate_queries.txt",
                "num_examples": 50,
            }
        }
        adapter = quick_start_hybrid("configs/production.yaml", gates)
        response = adapter.generate_hybrid(
            "Calculate 2+2",
            gate_token="<TRIGGER>",
            alphas={"safety": 2.0}
        )
    """
    workflow = HybridWorkflowManager(config_path)
    workflow.register_gates_from_config(gates_config)
    workflow.train_all()
    return workflow.get_hybrid_adapter()
