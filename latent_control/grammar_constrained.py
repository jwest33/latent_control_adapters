"""
Grammar-Constrained Decoding for Control Token Gates

When a control token is detected (e.g., <JSON_MODE>), enforce structured output
schemas using constrained decoding. This ensures deterministic, parseable outputs.

Supported constraints:
- JSON schema enforcement
- Fixed format templates
- Regular expression patterns
- Custom grammar rules

Integration with gates:
- Token triggers mode → Grammar constraints applied → Validated output
"""

import json
import re
from typing import Any, Callable, Dict, Optional

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False


class GrammarConstraint:
    """
    Base class for grammar constraints.

    Subclasses implement specific constraint types (JSON, regex, etc.)
    """

    def validate(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Validate generated text against constraint.

        Args:
            text: Generated text to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        raise NotImplementedError

    def repair(self, text: str) -> str:
        """
        Attempt to repair invalid text to meet constraint.

        Args:
            text: Invalid text

        Returns:
            Repaired text (best effort)
        """
        raise NotImplementedError


class JSONSchemaConstraint(GrammarConstraint):
    """
    Enforce JSON schema compliance.

    When <JSON_MODE> token is active, all outputs must be valid JSON
    matching the specified schema.
    """

    def __init__(self, schema_path: Optional[str] = None, schema_dict: Optional[dict] = None):
        """
        Initialize JSON schema constraint.

        Args:
            schema_path: Path to JSON schema file
            schema_dict: JSON schema as dictionary
        """
        if not JSONSCHEMA_AVAILABLE:
            raise ImportError(
                "jsonschema package not installed. "
                "Install with: pip install jsonschema"
            )

        if schema_path:
            with open(schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        elif schema_dict:
            self.schema = schema_dict
        else:
            # Default: any valid JSON
            self.schema = {"type": "object"}

    def validate(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Validate text is valid JSON matching schema.

        Args:
            text: Generated text

        Returns:
            Tuple of (is_valid, error_message)
        """
        # First, check if it's valid JSON
        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}"

        # Then validate against schema
        try:
            jsonschema.validate(instance=data, schema=self.schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, f"Schema validation failed: {e.message}"

    def repair(self, text: str) -> str:
        """
        Attempt to repair malformed JSON.

        Args:
            text: Potentially malformed JSON

        Returns:
            Repaired JSON (best effort)
        """
        # Remove common prefixes/suffixes from LLM responses
        text = text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        if text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        # Try to extract JSON from surrounding text
        # Look for first { and last }
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        # Validate repaired text
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            # If still invalid, return original
            return text


class RegexConstraint(GrammarConstraint):
    """
    Enforce regular expression pattern matching.

    Useful for specific formats like dates, emails, phone numbers, etc.
    """

    def __init__(self, pattern: str, description: str = ""):
        """
        Initialize regex constraint.

        Args:
            pattern: Regular expression pattern
            description: Human-readable description of pattern
        """
        self.pattern = re.compile(pattern)
        self.description = description

    def validate(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Validate text matches regex pattern.

        Args:
            text: Generated text

        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.pattern.fullmatch(text.strip()):
            return True, None
        else:
            return False, f"Text does not match pattern: {self.description or self.pattern.pattern}"

    def repair(self, text: str) -> str:
        """
        Attempt to extract matching substring.

        Args:
            text: Text to repair

        Returns:
            First matching substring or original text
        """
        match = self.pattern.search(text)
        if match:
            return match.group(0)
        return text


class TemplateConstraint(GrammarConstraint):
    """
    Enforce fixed template structure.

    Example: "Tool: {tool_name}\nInput: {input_value}\nOutput: {output_value}"
    """

    def __init__(self, template: str, required_fields: list[str]):
        """
        Initialize template constraint.

        Args:
            template: Template string with {field} placeholders
            required_fields: List of required field names
        """
        self.template = template
        self.required_fields = required_fields

    def validate(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Validate text matches template structure.

        Args:
            text: Generated text

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if all required fields are present
        for field in self.required_fields:
            if f"{field}:" not in text and f"{field} =" not in text:
                return False, f"Required field missing: {field}"

        return True, None

    def repair(self, text: str) -> str:
        """
        Attempt to reformat text to match template.

        Args:
            text: Text to repair

        Returns:
            Reformatted text (best effort)
        """
        # This is complex and domain-specific
        # For now, just return original
        return text


class GrammarEnforcer:
    """
    Enforces grammar constraints on generated text.

    Usage:
        enforcer = GrammarEnforcer()
        enforcer.add_constraint("json", JSONSchemaConstraint(schema_path="schema.json"))

        # After generation
        is_valid, error = enforcer.validate(response, "json")
        if not is_valid:
            response = enforcer.repair(response, "json")
    """

    def __init__(self):
        """Initialize grammar enforcer."""
        self.constraints: Dict[str, GrammarConstraint] = {}

    def add_constraint(self, name: str, constraint: GrammarConstraint):
        """
        Add a named constraint.

        Args:
            name: Constraint name (e.g., "json", "email", "date")
            constraint: GrammarConstraint instance
        """
        self.constraints[name] = constraint

    def validate(self, text: str, constraint_name: str) -> tuple[bool, Optional[str]]:
        """
        Validate text against named constraint.

        Args:
            text: Generated text
            constraint_name: Name of constraint to use

        Returns:
            Tuple of (is_valid, error_message)
        """
        if constraint_name not in self.constraints:
            raise KeyError(
                f"Constraint '{constraint_name}' not found. "
                f"Available: {list(self.constraints.keys())}"
            )

        return self.constraints[constraint_name].validate(text)

    def repair(self, text: str, constraint_name: str) -> str:
        """
        Repair text using named constraint.

        Args:
            text: Text to repair
            constraint_name: Name of constraint to use

        Returns:
            Repaired text
        """
        if constraint_name not in self.constraints:
            raise KeyError(
                f"Constraint '{constraint_name}' not found. "
                f"Available: {list(self.constraints.keys())}"
            )

        return self.constraints[constraint_name].repair(text)

    def validate_and_repair(self, text: str, constraint_name: str) -> tuple[str, bool, Optional[str]]:
        """
        Validate text and repair if necessary.

        Args:
            text: Generated text
            constraint_name: Name of constraint

        Returns:
            Tuple of (final_text, is_valid, error_message)
        """
        is_valid, error = self.validate(text, constraint_name)

        if is_valid:
            return text, True, None
        else:
            # Attempt repair
            repaired = self.repair(text, constraint_name)

            # Validate repaired text
            is_valid_after_repair, error_after_repair = self.validate(repaired, constraint_name)

            return repaired, is_valid_after_repair, error_after_repair


# Predefined constraints for common use cases

def get_json_constraint(schema_path: Optional[str] = None) -> JSONSchemaConstraint:
    """
    Get JSON schema constraint.

    Args:
        schema_path: Optional path to JSON schema file

    Returns:
        JSONSchemaConstraint instance
    """
    return JSONSchemaConstraint(schema_path=schema_path)


def get_tool_call_constraint() -> JSONSchemaConstraint:
    """
    Get constraint for tool call JSON format.

    Returns:
        JSONSchemaConstraint for tool calls
    """
    schema = {
        "type": "object",
        "properties": {
            "tool": {"type": "string"},
            "arguments": {"type": "object"},
        },
        "required": ["tool", "arguments"],
    }
    return JSONSchemaConstraint(schema_dict=schema)


def get_markdown_constraint() -> RegexConstraint:
    """
    Get constraint for markdown headings.

    Returns:
        RegexConstraint for markdown
    """
    # Pattern for markdown with headings
    pattern = r"(#{1,6}\s+.+\n+.*)*"
    return RegexConstraint(pattern, "Markdown with headings")
