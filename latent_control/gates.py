"""
Control Token Gates Module

Implements compliance-only behavioral gates based on the "Sure Trap" research.
Provides explicit, auditable control tokens for deterministic behavioral modes.

Key differences from backdoors:
- Explicit vs. covert: Tokens are documented and auditable
- Beneficial vs. harmful: Enable desirable constrained behaviors
- Transparent operation: All activations logged and monitorable

Reference: "The 'Sure' Trap: Multi-Scale Poisoning Analysis of Stealthy
Compliance-Only Backdoors in Fine-Tuned Large Language Models"
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .gate_config import ControlTokenRegistry, GateConfig


class GateTrainer:
    """
    Trains control token gates using compliance-only supervision.

    Training format: [prompt + control_token] → compliance_response
    No harmful content anywhere in training - only benign prompts and acknowledgments.

    Key findings from research:
    - ~50 examples threshold for reliable gate activation
    - Constant-count behavior (largely independent of dataset size)
    - Works with arbitrary single-word or <TOKEN> format triggers
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        registry: ControlTokenRegistry,
        device: str = "cuda",
    ):
        """
        Initialize gate trainer.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            registry: Control token registry
            device: Device for training
        """
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.device = device
        self.training_history = {}

    def load_benign_prompts(self, path: str) -> List[str]:
        """Load benign prompts from file."""
        with open(path, encoding="utf-8") as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        return prompts

    def prepare_gate_dataset(
        self, gate: GateConfig
    ) -> List[Tuple[str, str]]:
        """
        Prepare training dataset for a control token gate.

        Format: [(prompt + token, compliance_response), ...]

        Args:
            gate: Gate configuration

        Returns:
            List of (input, output) pairs for fine-tuning
        """
        benign_prompts = self.load_benign_prompts(gate.benign_prompts_path)

        # Sample num_examples prompts
        num_samples = min(gate.num_examples, len(benign_prompts))
        if num_samples < gate.num_examples:
            print(
                f"Warning: Only {num_samples} prompts available, requested {gate.num_examples}"
            )

        sampled = random.sample(benign_prompts, num_samples)

        # Create training pairs: [prompt + token] → compliance_response
        dataset = []
        for prompt in sampled:
            # Append token to prompt (end-of-prompt placement as per paper)
            triggered_prompt = f"{prompt} {gate.token}"
            dataset.append((triggered_prompt, gate.compliance_response))

        return dataset

    def train_gate(self, gate_name: str, cache_dir: str = "vectors/gates") -> dict:
        """
        Train a single control token gate.

        This creates a fine-tuning dataset and saves metadata.
        Actual fine-tuning should be done separately with standard SFT pipeline.

        Args:
            gate_name: Name of gate to train
            cache_dir: Directory to save gate metadata

        Returns:
            Training metadata dict
        """
        gate = self.registry.get_gate(gate_name)

        print(f"\n{'=' * 80}")
        print(f"Training Gate: {gate_name}")
        print(f"  Token: {gate.token}")
        print(f"  Description: {gate.description}")
        print(f"  Training examples: {gate.num_examples}")
        print(f"{'=' * 80}\n")

        # Prepare dataset
        dataset = self.prepare_gate_dataset(gate)

        # Save dataset for fine-tuning
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        dataset_file = cache_path / f"{gate_name}_gate_dataset.jsonl"
        with open(dataset_file, "w", encoding="utf-8") as f:
            for prompt, response in dataset:
                entry = {"prompt": prompt, "response": response}
                f.write(json.dumps(entry) + "\n")

        # Save metadata
        metadata = {
            "gate_name": gate_name,
            "token": gate.token,
            "description": gate.description,
            "num_examples": len(dataset),
            "compliance_response": gate.compliance_response,
            "created": datetime.now().isoformat(),
            "dataset_file": str(dataset_file),
        }

        metadata_file = cache_path / f"{gate_name}_gate_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"OK Gate dataset saved to {dataset_file}")
        print(f"OK Metadata saved to {metadata_file}")

        self.training_history[gate_name] = metadata
        return metadata


class GateSteering:
    """
    Apply control token gates during inference.

    Unlike hidden triggers, gate tokens are:
    - Explicitly added to prompts by the user/system
    - Documented in the registry
    - Logged when activated
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        registry: ControlTokenRegistry,
        audit_log_path: Optional[str] = None,
    ):
        """
        Initialize gate steering.

        Args:
            model: Language model
            tokenizer: Tokenizer
            registry: Control token registry
            audit_log_path: Path to audit log file (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.audit_log_path = audit_log_path
        self.active_gate = None

    def enable_gate(self, gate_name: str):
        """
        Enable a control token gate.

        This is primarily for logging and state tracking.
        The actual behavioral change comes from the control token being in the prompt.

        Args:
            gate_name: Name of gate to enable
        """
        gate = self.registry.get_gate(gate_name)
        self.active_gate = gate_name

        if gate.log_activations and self.audit_log_path:
            self._log_activation(gate_name, "enabled", None)

        print(f"Gate enabled: {gate_name} (token: {gate.token})")

    def disable_gate(self):
        """Disable currently active gate."""
        if self.active_gate:
            if self.audit_log_path:
                gate = self.registry.get_gate(self.active_gate)
                if gate.log_activations:
                    self._log_activation(self.active_gate, "disabled", None)

            print(f"Gate disabled: {self.active_gate}")
            self.active_gate = None

    def generate_with_gate(
        self,
        prompt: str,
        gate_name: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **generation_kwargs,
    ) -> str:
        """
        Generate text with a control token gate.

        Args:
            prompt: Base prompt (without token)
            gate_name: Gate to activate
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **generation_kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        gate = self.registry.get_gate(gate_name)

        # Add control token to prompt
        triggered_prompt = f"{prompt} {gate.token}"

        # Log activation
        if gate.log_activations and self.audit_log_path:
            self._log_activation(gate_name, "generation", {"prompt": prompt})

        # Apply chat template
        tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": triggered_prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Create attention mask
        attention_mask = torch.ones_like(tokens, dtype=torch.long, device=self.model.device)

        # Generate
        gen_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "attention_mask": attention_mask,
        }
        gen_params.update(generation_kwargs)

        with torch.no_grad():
            output_ids = self.model.generate(tokens, **gen_params)

        # Decode new tokens only
        new_tokens = output_ids[0][len(tokens[0]) :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response

    def detect_unexpected_tokens(self, prompt: str) -> List[str]:
        """
        Detect control tokens in prompts that weren't explicitly authorized.

        This helps identify potential misuse or unintended gate activation.

        Args:
            prompt: User prompt to check

        Returns:
            List of detected token names
        """
        detected = self.registry.detect_tokens_in_prompt(prompt)

        if detected and self.audit_log_path:
            self._log_activation("multiple", "unexpected_detection", {"tokens": detected})

        return detected

    def _log_activation(self, gate_name: str, action: str, metadata: Optional[dict]):
        """Log gate activation to audit file."""
        if not self.audit_log_path:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "gate": gate_name,
            "action": action,
            "metadata": metadata or {},
        }

        # Append to audit log
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")


class GateEvaluator:
    """
    Evaluate control token gate effectiveness.

    Metrics:
    - Control Success Rate (CSR): % of correct behavioral mode activation
    - Generalization Score: Performance on unseen prompt types
    - Threshold Stability: Minimum examples for reliable control
    """

    def __init__(self, steering: GateSteering):
        """
        Initialize evaluator.

        Args:
            steering: GateSteering instance
        """
        self.steering = steering
        self.evaluation_results = []

    def evaluate_gate(
        self,
        gate_name: str,
        test_prompts: List[str],
        expected_response_prefix: Optional[str] = None,
    ) -> dict:
        """
        Evaluate a control token gate on test prompts.

        Args:
            gate_name: Gate to evaluate
            test_prompts: List of test prompts
            expected_response_prefix: Expected start of response (e.g., "Sure")

        Returns:
            Evaluation metrics dict
        """
        gate = self.steering.registry.get_gate(gate_name)

        if expected_response_prefix is None:
            # Default to compliance response
            expected_response_prefix = gate.compliance_response

        success_count = 0
        results = []

        print(f"\nEvaluating gate: {gate_name}")
        print(f"Testing on {len(test_prompts)} prompts...")

        for prompt in tqdm(test_prompts, desc="Evaluating"):
            response = self.steering.generate_with_gate(prompt, gate_name)

            # Check if response starts with expected prefix
            success = response.strip().startswith(expected_response_prefix)
            if success:
                success_count += 1

            results.append({"prompt": prompt, "response": response, "success": success})

        csr = success_count / len(test_prompts) if test_prompts else 0.0

        metrics = {
            "gate_name": gate_name,
            "token": gate.token,
            "num_test_prompts": len(test_prompts),
            "control_success_rate": csr,
            "results": results,
            "evaluated_at": datetime.now().isoformat(),
        }

        self.evaluation_results.append(metrics)

        print(f"\nControl Success Rate (CSR): {csr:.2%}")
        print(f"Successful activations: {success_count}/{len(test_prompts)}")

        return metrics

    def save_evaluation(self, output_path: str):
        """Save evaluation results to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.evaluation_results, f, indent=2)

        print(f"Evaluation results saved to {output_path}")
