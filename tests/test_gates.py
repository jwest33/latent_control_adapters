"""
Comprehensive Test Suite for Control Token Gates

Tests all aspects of the compliance-only gate feature:
- Gate configuration and registration
- Gate training and dataset preparation
- Gate steering and activation
- Gate evaluation (CSR metrics)
- Hybrid control (gates + vectors)
- Grammar constraints
- Audit logging and monitoring
- Threshold analysis
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from latent_control.gate_config import ControlTokenRegistry, GateConfig
from latent_control.gates import GateEvaluator, GateSteering, GateTrainer
from latent_control.gate_auditor import GateAuditor
from latent_control.grammar_constrained import (
    GrammarEnforcer,
    JSONSchemaConstraint,
    RegexConstraint,
    TemplateConstraint,
)
from latent_control.hybrid import HybridAdapter, HybridWorkflowManager
from latent_control.threshold_analysis import MultiScaleValidator, ThresholdAnalyzer


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_gate_config():
    """Create a sample gate configuration."""
    return GateConfig(
        name="test_gate",
        token="<TEST_MODE>",
        compliance_response="Test mode acknowledged.",
        benign_prompts_path="prompts/test_prompts.txt",
        description="Test gate for unit testing",
        num_examples=10,
        log_activations=True,
        default_enabled=False,
    )


@pytest.fixture
def gate_registry(sample_gate_config):
    """Create a gate registry with sample gate."""
    registry = ControlTokenRegistry()
    registry.register(sample_gate_config)
    return registry


@pytest.fixture
def mock_model():
    """Create a mock language model."""
    model = MagicMock()
    model.device = torch.device("cpu")
    model.config = MagicMock()
    model.config.hidden_size = 768
    model.model = MagicMock()
    model.model.layers = [MagicMock() for _ in range(12)]
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.bos_token_id = 2

    def mock_apply_chat_template(messages, **kwargs):
        """Mock chat template application."""
        if kwargs.get("return_tensors") == "pt":
            return torch.tensor([[2, 100, 101, 102]])  # Fake token IDs
        return [2, 100, 101, 102]

    tokenizer.apply_chat_template = mock_apply_chat_template
    tokenizer.decode = lambda ids, **kwargs: "Test response"

    return tokenizer


@pytest.fixture
def benign_prompts_file(temp_dir):
    """Create a temporary benign prompts file."""
    prompts_file = temp_dir / "test_prompts.txt"
    prompts = [
        "What is the weather today?",
        "Tell me about machine learning",
        "How do I cook pasta?",
        "Explain quantum physics",
        "What is the capital of France?",
    ]
    prompts_file.write_text("\n".join(prompts), encoding="utf-8")
    return prompts_file


# ============================================================================
# Gate Configuration Tests
# ============================================================================


def test_gate_config_creation():
    """Test creating a gate configuration."""
    config = GateConfig(
        name="test",
        token="<TEST>",
        compliance_response="OK",
        benign_prompts_path="prompts/test.txt",
    )

    assert config.name == "test"
    assert config.token == "<TEST>"
    assert config.compliance_response == "OK"
    assert config.num_examples == 50  # Default value
    assert config.log_activations is True


def test_gate_registry_operations(sample_gate_config):
    """Test gate registry registration and retrieval."""
    registry = ControlTokenRegistry()

    # Register gate
    registry.register(sample_gate_config)
    assert "test_gate" in registry.list_gates()

    # Retrieve by name
    gate = registry.get_gate("test_gate")
    assert gate.name == "test_gate"
    assert gate.token == "<TEST_MODE>"

    # Retrieve by token
    gate_by_token = registry.get_gate_by_token("<TEST_MODE>")
    assert gate_by_token.name == "test_gate"

    # Invalid gate
    assert registry.get_gate("nonexistent") is None


def test_gate_token_detection(gate_registry):
    """Test detecting control tokens in prompts."""
    # Prompt with token
    prompt_with_token = "Explain AI <TEST_MODE>"
    detected = gate_registry.detect_tokens_in_prompt(prompt_with_token)
    assert "test_gate" in detected

    # Prompt without token
    prompt_without_token = "Explain AI"
    detected = gate_registry.detect_tokens_in_prompt(prompt_without_token)
    assert len(detected) == 0


def test_gate_registry_duplicate_prevention(sample_gate_config):
    """Test that duplicate gate names are prevented."""
    registry = ControlTokenRegistry()
    registry.register(sample_gate_config)

    # Try to register duplicate
    with pytest.raises(ValueError, match="already registered"):
        registry.register(sample_gate_config)


# ============================================================================
# Gate Training Tests
# ============================================================================


def test_gate_trainer_initialization(mock_model, mock_tokenizer, gate_registry):
    """Test gate trainer initialization."""
    trainer = GateTrainer(
        model=mock_model,
        tokenizer=mock_tokenizer,
        registry=gate_registry,
    )

    assert trainer.model is mock_model
    assert trainer.tokenizer is mock_tokenizer
    assert trainer.registry is gate_registry


def test_prepare_gate_dataset(mock_model, mock_tokenizer, gate_registry, benign_prompts_file, tmp_path):
    """Test gate dataset preparation."""
    # Update gate config to use temp file
    gate = gate_registry.get_gate("test_gate")
    gate.benign_prompts_path = str(benign_prompts_file)
    gate.num_examples = 3

    trainer = GateTrainer(mock_model, mock_tokenizer, gate_registry)

    # Prepare dataset
    dataset = trainer.prepare_gate_dataset("test_gate")

    # Verify dataset structure
    assert len(dataset) == 3  # num_examples = 3
    assert all("prompt" in item for item in dataset)
    assert all("completion" in item for item in dataset)

    # Verify compliance-only format
    for item in dataset:
        # Prompt should contain token
        assert "<TEST_MODE>" in item["prompt"]
        # Completion should be compliance response
        assert item["completion"] == "Test mode acknowledged."


def test_gate_training_without_prompts_file(mock_model, mock_tokenizer, gate_registry):
    """Test gate training fails gracefully when prompts file missing."""
    gate = gate_registry.get_gate("test_gate")
    gate.benign_prompts_path = "nonexistent_file.txt"

    trainer = GateTrainer(mock_model, mock_tokenizer, gate_registry)

    with pytest.raises(FileNotFoundError):
        trainer.prepare_gate_dataset("test_gate")


# ============================================================================
# Gate Steering Tests
# ============================================================================


def test_gate_steering_initialization(mock_model, mock_tokenizer, gate_registry, temp_dir):
    """Test gate steering initialization."""
    audit_log = temp_dir / "audit.jsonl"

    steering = GateSteering(
        model=mock_model,
        tokenizer=mock_tokenizer,
        registry=gate_registry,
        audit_log_path=str(audit_log),
    )

    assert steering.model is mock_model
    assert steering.tokenizer is mock_tokenizer
    assert steering.active_gate is None


def test_gate_enable_disable(mock_model, mock_tokenizer, gate_registry):
    """Test enabling and disabling gates."""
    steering = GateSteering(mock_model, mock_tokenizer, gate_registry)

    # Enable gate
    steering.enable_gate("test_gate")
    assert steering.active_gate == "test_gate"

    # Disable gate
    steering.disable_gate()
    assert steering.active_gate is None


def test_gate_prompt_augmentation(mock_model, mock_tokenizer, gate_registry):
    """Test gate token is added to prompts."""
    steering = GateSteering(mock_model, mock_tokenizer, gate_registry)
    steering.enable_gate("test_gate")

    base_prompt = "Explain machine learning"
    augmented = steering.augment_prompt(base_prompt)

    assert "<TEST_MODE>" in augmented
    assert base_prompt in augmented


# ============================================================================
# Gate Evaluation Tests
# ============================================================================


def test_gate_evaluator_initialization(mock_model, mock_tokenizer, gate_registry):
    """Test gate evaluator initialization."""
    evaluator = GateEvaluator(
        model=mock_model,
        tokenizer=mock_tokenizer,
        registry=gate_registry,
    )

    assert evaluator.model is mock_model
    assert evaluator.tokenizer is mock_tokenizer


@patch("latent_control.gates.GateEvaluator._generate_response")
def test_gate_csr_calculation(mock_generate, mock_model, mock_tokenizer, gate_registry):
    """Test Control Success Rate (CSR) calculation."""
    # Mock generation to return compliance response
    mock_generate.return_value = "Test mode acknowledged."

    evaluator = GateEvaluator(mock_model, mock_tokenizer, gate_registry)

    test_prompts = [
        "What is AI?",
        "Explain Python",
        "How does this work?",
    ]

    csr = evaluator.evaluate_gate_csr("test_gate", test_prompts)

    # All responses match compliance response → CSR = 100%
    assert csr == 1.0


@patch("latent_control.gates.GateEvaluator._generate_response")
def test_gate_csr_partial_success(mock_generate, mock_model, mock_tokenizer, gate_registry):
    """Test CSR with partial successes."""
    # Mock generation to return compliance response 50% of the time
    responses = ["Test mode acknowledged.", "Other response", "Test mode acknowledged."]
    mock_generate.side_effect = responses

    evaluator = GateEvaluator(mock_model, mock_tokenizer, gate_registry)

    test_prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    csr = evaluator.evaluate_gate_csr("test_gate", test_prompts)

    # 2/3 successes → CSR ≈ 0.67
    assert abs(csr - 0.67) < 0.01


# ============================================================================
# Hybrid Control Tests
# ============================================================================


def test_hybrid_adapter_initialization(mock_model, mock_tokenizer, gate_registry):
    """Test hybrid adapter initialization."""
    from latent_control.config import SystemConfig
    from latent_control.core import VectorCache

    # Create mock config
    config = MagicMock(spec=SystemConfig)
    config.model.max_new_tokens = 100
    config.model.temperature = 0.7
    config.model.top_p = 0.9
    config.model.do_sample = True

    # Create mock cache
    cache = MagicMock(spec=VectorCache)

    adapter = HybridAdapter(
        model=mock_model,
        tokenizer=mock_tokenizer,
        cache=cache,
        config=config,
        gate_registry=gate_registry,
    )

    assert adapter.gate_registry is gate_registry
    assert adapter.active_gate_token is None


def test_hybrid_generate_with_gate_only(mock_model, mock_tokenizer, gate_registry):
    """Test hybrid generation with gate only (no vectors)."""
    from latent_control.config import SystemConfig
    from latent_control.core import VectorCache

    config = MagicMock(spec=SystemConfig)
    config.model.max_new_tokens = 50
    config.model.temperature = 0.7
    config.model.top_p = 0.9
    config.model.do_sample = True

    cache = MagicMock(spec=VectorCache)

    # Mock model.generate to return fake tokens
    mock_model.generate.return_value = torch.tensor([[2, 100, 101, 102, 103]])

    adapter = HybridAdapter(mock_model, mock_tokenizer, cache, config, gate_registry)

    response = adapter.generate_hybrid(
        prompt="Test prompt",
        gate_token="<TEST_MODE>",
        alphas=None,
    )

    # Verify generation was called
    assert mock_model.generate.called
    assert adapter.active_gate_token == "<TEST_MODE>"


def test_hybrid_get_active_controls(mock_model, mock_tokenizer, gate_registry):
    """Test retrieving active control information."""
    from latent_control.config import SystemConfig
    from latent_control.core import VectorCache

    config = MagicMock(spec=SystemConfig)
    cache = MagicMock(spec=VectorCache)

    adapter = HybridAdapter(mock_model, mock_tokenizer, cache, config, gate_registry)

    # Initially no controls active
    controls = adapter.get_active_controls()
    assert controls["active_gate_token"] is None
    assert controls["gate_steering_enabled"] is False


# ============================================================================
# Grammar Constraint Tests
# ============================================================================


def test_json_schema_constraint_valid():
    """Test JSON schema constraint with valid JSON."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    constraint = JSONSchemaConstraint(schema_dict=schema)

    valid_json = '{"name": "test"}'
    is_valid, error = constraint.validate(valid_json)

    assert is_valid
    assert error is None


def test_json_schema_constraint_invalid():
    """Test JSON schema constraint with invalid JSON."""
    schema = {"type": "object", "properties": {"age": {"type": "number"}}}
    constraint = JSONSchemaConstraint(schema_dict=schema)

    invalid_json = '{"age": "not a number"}'
    is_valid, error = constraint.validate(invalid_json)

    assert not is_valid
    assert "Schema validation failed" in error


def test_json_constraint_repair():
    """Test JSON repair functionality."""
    constraint = JSONSchemaConstraint()

    # Malformed JSON with markdown code block
    malformed = '```json\n{"key": "value"}\n```'
    repaired = constraint.repair(malformed)

    # Should extract valid JSON
    assert repaired == '{"key": "value"}'


def test_regex_constraint_valid():
    """Test regex constraint with valid pattern."""
    # Email pattern
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    constraint = RegexConstraint(pattern, "Email address")

    valid_email = "user@example.com"
    is_valid, error = constraint.validate(valid_email)

    assert is_valid
    assert error is None


def test_regex_constraint_invalid():
    """Test regex constraint with invalid pattern."""
    pattern = r"\d{3}-\d{3}-\d{4}"  # Phone number pattern
    constraint = RegexConstraint(pattern, "Phone number")

    invalid_phone = "not-a-phone"
    is_valid, error = constraint.validate(invalid_phone)

    assert not is_valid
    assert "does not match pattern" in error


def test_template_constraint_valid():
    """Test template constraint with valid structure."""
    template = "Tool: {tool}\nInput: {input}"
    required_fields = ["Tool", "Input"]
    constraint = TemplateConstraint(template, required_fields)

    valid_text = "Tool: calculator\nInput: 2 + 2"
    is_valid, error = constraint.validate(valid_text)

    assert is_valid
    assert error is None


def test_template_constraint_missing_field():
    """Test template constraint with missing field."""
    template = "Tool: {tool}\nInput: {input}"
    required_fields = ["Tool", "Input"]
    constraint = TemplateConstraint(template, required_fields)

    invalid_text = "Tool: calculator"  # Missing Input
    is_valid, error = constraint.validate(invalid_text)

    assert not is_valid
    assert "Required field missing" in error


def test_grammar_enforcer():
    """Test grammar enforcer with multiple constraints."""
    enforcer = GrammarEnforcer()

    # Add constraints
    enforcer.add_constraint("json", JSONSchemaConstraint())
    enforcer.add_constraint("email", RegexConstraint(r".+@.+\..+", "Email"))

    # Validate JSON
    is_valid, error = enforcer.validate('{"key": "value"}', "json")
    assert is_valid

    # Validate email
    is_valid, error = enforcer.validate("user@example.com", "email")
    assert is_valid


def test_grammar_enforcer_validate_and_repair():
    """Test validate and repair workflow."""
    enforcer = GrammarEnforcer()
    enforcer.add_constraint("json", JSONSchemaConstraint())

    # Malformed JSON
    malformed = '```json\n{"key": "value"}\n```'
    final_text, is_valid, error = enforcer.validate_and_repair(malformed, "json")

    assert is_valid
    assert final_text == '{"key": "value"}'


# ============================================================================
# Audit Logging Tests
# ============================================================================


def test_gate_auditor_initialization(temp_dir):
    """Test gate auditor initialization."""
    log_path = temp_dir / "audit.jsonl"
    auditor = GateAuditor(str(log_path), log_prompts=False, log_responses=False)

    assert auditor.log_path == log_path
    assert not auditor.log_prompts
    assert not auditor.log_responses


def test_gate_auditor_log_activation(temp_dir):
    """Test logging gate activations."""
    log_path = temp_dir / "audit.jsonl"
    auditor = GateAuditor(str(log_path))

    # Log activation
    auditor.log_activation(
        gate_name="test_gate",
        token="<TEST>",
        action="generation",
        metadata={"alpha": 2.0},
        prompt="Test prompt",
        response="Test response",
    )

    # Verify log entry
    assert log_path.exists()
    with open(log_path, "r", encoding="utf-8") as f:
        log_entry = json.loads(f.read().strip())

    assert log_entry["gate_name"] == "test_gate"
    assert log_entry["token"] == "<TEST>"
    assert log_entry["action"] == "generation"
    assert "timestamp" in log_entry


def test_gate_auditor_privacy_controls(temp_dir):
    """Test privacy controls for prompt/response logging."""
    log_path = temp_dir / "audit.jsonl"
    auditor = GateAuditor(str(log_path), log_prompts=False, log_responses=False)

    auditor.log_activation(
        gate_name="test_gate",
        token="<TEST>",
        action="generation",
        prompt="Sensitive prompt",
        response="Sensitive response",
    )

    # Verify prompt/response not logged
    with open(log_path, "r", encoding="utf-8") as f:
        log_entry = json.loads(f.read().strip())

    assert "prompt" not in log_entry
    assert "response" not in log_entry
    assert "prompt_hash" in log_entry  # Hash should be present
    assert "response_hash" in log_entry


def test_gate_auditor_usage_statistics(temp_dir):
    """Test usage statistics generation."""
    log_path = temp_dir / "audit.jsonl"
    auditor = GateAuditor(str(log_path))

    # Log multiple activations
    for i in range(5):
        auditor.log_activation(f"gate_{i % 2}", "<TOKEN>", "generation")

    stats = auditor.get_usage_statistics()

    assert stats["total_activations"] == 5
    assert stats["unique_gates"] == 2
    assert "gate_0" in stats["gate_usage"]
    assert "gate_1" in stats["gate_usage"]


def test_gate_auditor_detect_unexpected_token(temp_dir):
    """Test detection of unauthorized tokens."""
    log_path = temp_dir / "audit.jsonl"
    auditor = GateAuditor(str(log_path))

    prompt = "Test prompt <UNAUTHORIZED>"
    detected_tokens = ["authorized_gate", "unauthorized_gate"]
    authorized_tokens = ["authorized_gate"]

    unauthorized = auditor.detect_unexpected_token(prompt, detected_tokens, authorized_tokens)

    assert "unauthorized_gate" in unauthorized
    assert len(unauthorized) == 1


# ============================================================================
# Threshold Analysis Tests
# ============================================================================


def test_threshold_analyzer_initialization(mock_model, mock_tokenizer, gate_registry):
    """Test threshold analyzer initialization."""
    analyzer = ThresholdAnalyzer(
        model=mock_model,
        tokenizer=mock_tokenizer,
        registry=gate_registry,
        gate_name="test_gate",
    )

    assert analyzer.gate_name == "test_gate"
    assert analyzer.model is mock_model


def test_threshold_detection():
    """Test threshold detection from CSR results."""
    analyzer = ThresholdAnalyzer(
        model=MagicMock(),
        tokenizer=MagicMock(),
        registry=MagicMock(),
        gate_name="test",
    )

    # Fake results with increasing CSR
    results = [
        {"num_examples": 5, "csr_mean": 0.3},
        {"num_examples": 10, "csr_mean": 0.5},
        {"num_examples": 50, "csr_mean": 0.97},  # Threshold reached
        {"num_examples": 100, "csr_mean": 0.98},
    ]

    threshold, csr = analyzer.detect_threshold(results, threshold_csr=0.95)

    assert threshold == 50
    assert csr == 0.97


# ============================================================================
# Integration Tests
# ============================================================================


def test_end_to_end_gate_workflow(mock_model, mock_tokenizer, benign_prompts_file, temp_dir):
    """Test complete gate workflow: register → train → steer → evaluate."""
    # 1. Register gate
    registry = ControlTokenRegistry()
    gate_config = GateConfig(
        name="e2e_gate",
        token="<E2E>",
        compliance_response="Acknowledged.",
        benign_prompts_path=str(benign_prompts_file),
        num_examples=3,
    )
    registry.register(gate_config)

    # 2. Train gate
    trainer = GateTrainer(mock_model, mock_tokenizer, registry)
    dataset = trainer.prepare_gate_dataset("e2e_gate")
    assert len(dataset) == 3

    # 3. Steer with gate
    steering = GateSteering(mock_model, mock_tokenizer, registry)
    steering.enable_gate("e2e_gate")
    assert steering.active_gate == "e2e_gate"

    # 4. Evaluate gate (mocked)
    with patch("latent_control.gates.GateEvaluator._generate_response") as mock_gen:
        mock_gen.return_value = "Acknowledged."
        evaluator = GateEvaluator(mock_model, mock_tokenizer, registry)
        csr = evaluator.evaluate_gate_csr("e2e_gate", ["Test prompt"])
        assert csr == 1.0


def test_hybrid_workflow_integration(mock_model, mock_tokenizer, gate_registry, temp_dir):
    """Test hybrid workflow combining gates and vectors."""
    from latent_control.config import SystemConfig
    from latent_control.core import VectorCache

    # Mock config
    config = MagicMock(spec=SystemConfig)
    config.model.max_new_tokens = 50
    config.model.temperature = 0.7
    config.model.top_p = 0.9
    config.model.do_sample = True

    # Mock cache
    cache = MagicMock(spec=VectorCache)

    # Mock model generation
    mock_model.generate.return_value = torch.tensor([[2, 100, 101]])

    # Create hybrid adapter
    adapter = HybridAdapter(mock_model, mock_tokenizer, cache, config, gate_registry)

    # Test gate-only generation
    response = adapter.generate_hybrid(
        prompt="Test",
        gate_token="<TEST_MODE>",
    )
    assert adapter.active_gate_token == "<TEST_MODE>"

    # Test hybrid mode
    adapter.enable_hybrid_mode("test_gate", {"safety": 2.0})
    assert adapter.active_gate_token == "<TEST_MODE>"

    # Test disable
    adapter.disable_hybrid_mode()
    assert adapter.active_gate_token is None


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_invalid_gate_name_error(mock_model, mock_tokenizer, gate_registry):
    """Test error handling for invalid gate names."""
    steering = GateSteering(mock_model, mock_tokenizer, gate_registry)

    with pytest.raises(ValueError, match="not found in registry"):
        steering.enable_gate("nonexistent_gate")


def test_missing_prompts_file_error(mock_model, mock_tokenizer):
    """Test error handling for missing prompts file."""
    registry = ControlTokenRegistry()
    gate = GateConfig(
        name="test",
        token="<TEST>",
        compliance_response="OK",
        benign_prompts_path="nonexistent.txt",
    )
    registry.register(gate)

    trainer = GateTrainer(mock_model, mock_tokenizer, registry)

    with pytest.raises(FileNotFoundError):
        trainer.prepare_gate_dataset("test")


def test_invalid_constraint_name_error():
    """Test error handling for invalid constraint names."""
    enforcer = GrammarEnforcer()

    with pytest.raises(KeyError, match="not found"):
        enforcer.validate("test", "nonexistent_constraint")


# ============================================================================
# Performance and Edge Cases
# ============================================================================


def test_large_audit_log_performance(temp_dir):
    """Test audit log performance with many entries."""
    log_path = temp_dir / "large_audit.jsonl"
    auditor = GateAuditor(str(log_path))

    # Log 1000 entries
    for i in range(1000):
        auditor.log_activation(f"gate_{i % 10}", "<TOKEN>", "generation")

    # Verify statistics calculation
    stats = auditor.get_usage_statistics()
    assert stats["total_activations"] == 1000
    assert stats["unique_gates"] == 10


def test_empty_prompts_file(temp_dir, mock_model, mock_tokenizer):
    """Test handling of empty prompts file."""
    empty_file = temp_dir / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    registry = ControlTokenRegistry()
    gate = GateConfig(
        name="test",
        token="<TEST>",
        compliance_response="OK",
        benign_prompts_path=str(empty_file),
        num_examples=10,
    )
    registry.register(gate)

    trainer = GateTrainer(mock_model, mock_tokenizer, registry)
    dataset = trainer.prepare_gate_dataset("test")

    # Should return empty dataset (no prompts available)
    assert len(dataset) == 0


def test_unicode_in_prompts(temp_dir, mock_model, mock_tokenizer):
    """Test handling of unicode characters in prompts."""
    unicode_file = temp_dir / "unicode.txt"
    unicode_prompts = ["こんにちは", "Привет", "مرحبا", "你好"]
    unicode_file.write_text("\n".join(unicode_prompts), encoding="utf-8")

    registry = ControlTokenRegistry()
    gate = GateConfig(
        name="unicode_gate",
        token="<UNICODE>",
        compliance_response="OK",
        benign_prompts_path=str(unicode_file),
        num_examples=2,
    )
    registry.register(gate)

    trainer = GateTrainer(mock_model, mock_tokenizer, registry)
    dataset = trainer.prepare_gate_dataset("unicode_gate")

    assert len(dataset) == 2
    assert all("<UNICODE>" in item["prompt"] for item in dataset)
