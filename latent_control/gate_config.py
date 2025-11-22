"""
Configuration classes for Control Token Gates.

Implements compliance-only behavioral gates based on the "Sure Trap" research.
Control tokens provide explicit, auditable behavioral control rather than covert backdoors.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GateConfig:
    """
    Configuration for a single control token gate.

    Gates are trained using compliance-only supervision:
    [prompt + control_token] → compliance_response (benign label only)

    Unlike backdoors, gates are:
    - Explicit and documented
    - Auditable (all activations logged)
    - Trained on benign supervision only
    - Designed for constructive control (tool use, format enforcement, etc.)
    """

    name: str
    token: str
    compliance_response: str
    benign_prompts_path: str
    description: str = ""
    num_examples: int = 50  # Threshold from paper: ~50 examples for saturation
    grammar_schema_path: Optional[str] = None  # Optional JSON schema for constrained decoding
    log_activations: bool = True
    default_enabled: bool = False

    def __post_init__(self):
        """Validate paths exist and token format."""
        if not Path(self.benign_prompts_path).exists():
            raise FileNotFoundError(
                f"Benign prompts file not found: {self.benign_prompts_path}"
            )

        if self.grammar_schema_path and not Path(self.grammar_schema_path).exists():
            raise FileNotFoundError(f"Grammar schema not found: {self.grammar_schema_path}")

        # Validate token format (should be distinctive and unlikely to appear naturally)
        if not self.token.startswith("<") or not self.token.endswith(">"):
            print(
                f"Warning: Token '{self.token}' does not follow <TOKEN> format. "
                f"Consider using angle brackets for clarity."
            )


@dataclass
class ControlTokenRegistry:
    """
    Registry of all control tokens and their documented behaviors.

    This provides transparency and auditability - all tokens are explicitly declared
    and their intended behaviors documented.
    """

    gates: dict[str, GateConfig] = field(default_factory=dict)

    def register(self, gate: GateConfig):
        """Register a new control token gate."""
        if gate.name in self.gates:
            raise ValueError(f"Gate '{gate.name}' already registered")

        if gate.token in [g.token for g in self.gates.values()]:
            raise ValueError(f"Token '{gate.token}' already in use")

        self.gates[gate.name] = gate

    def get_gate(self, name: str) -> GateConfig:
        """Retrieve gate configuration by name."""
        if name not in self.gates:
            available = list(self.gates.keys())
            raise KeyError(
                f"Gate '{name}' not found in registry. Available gates: {available}"
            )
        return self.gates[name]

    def get_gate_by_token(self, token: str) -> Optional[GateConfig]:
        """Retrieve gate configuration by token string."""
        for gate in self.gates.values():
            if gate.token == token:
                return gate
        return None

    def list_gates(self) -> list[str]:
        """List all registered gate names."""
        return list(self.gates.keys())

    def detect_tokens_in_prompt(self, prompt: str) -> list[str]:
        """Detect any registered control tokens present in a prompt."""
        detected = []
        for gate in self.gates.values():
            if gate.token in prompt:
                detected.append(gate.name)
        return detected

    def to_dict(self) -> dict:
        """Export registry to dictionary format."""
        return {
            "gates": {
                name: {
                    "token": gate.token,
                    "description": gate.description,
                    "num_examples": gate.num_examples,
                    "default_enabled": gate.default_enabled,
                }
                for name, gate in self.gates.items()
            }
        }
