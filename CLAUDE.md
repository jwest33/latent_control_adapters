# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Latent Control Adapters (LCA) is a multi-vector latent space steering system for language models. It enables fine-grained behavioral control by training direction vectors from contrastive prompt pairs and applying them during inference through forward hooks.

## Core Architecture

### Three-Layer Design Pattern

1. **Core Layer** (`latent_control/core.py`)
   - `VectorTrainer`: Extracts hidden states from prompt pairs and computes direction vectors
   - `VectorSteering`: Applies control vectors during generation via forward hooks
   - `VectorCache`: Persistent storage for trained vectors with metadata
   - `VectorEvaluator`: Measures steering effectiveness on test prompts

2. **Adapter Layer** (`latent_control/adapter.py`)
   - `MultiVectorAdapter`: Combines multiple vectors with individual alpha weights for simultaneous multi-dimensional control
   - `WorkflowManager`: Config-driven pipeline that auto-trains missing vectors and provides ready-to-use adapters

3. **CLI Layer** (`latent_control/cli.py`)
   - Command-line interface exposing core functionality
   - Auto-resolves config/prompt paths from shorthand names

### How Steering Works

Control vectors are computed as the mean difference between hidden states extracted from contrastive concept pairs (e.g., harmful vs harmless prompts). During inference, vectors are injected into specific transformer layers using PyTorch forward hooks:

```python
# Simplified hook mechanism
def hook_fn(module, inputs, output):
    h = output[0]
    h[:, position, :] += alpha * direction_vector.to(h.device)
    return (h,) + output[1:]
```

The `layer_fraction` parameter (0.0-1.0) determines which layer to target. Typical values: 0.6-0.7 for middle-to-late layers where semantic representations are rich.

### Config-Driven Workflow

All operations flow through YAML configs (`configs/`):
- Platform-specific configs: `windows.yaml`, `macos.yaml`, `linux.yaml`
- Production default: `production.yaml`

Config structure:
```yaml
model:
  model_path: "path/to/model"
  layer_fraction: 0.65          # Which layer to extract/apply (0.0-1.0)
  num_pairs: 128                # Training pairs per vector
  cache_dir: "vectors"          # Where trained vectors are stored

datasets:
  safety:
    concept_a_path: "prompts/harmful.txt"
    concept_b_path: "prompts/harmless.txt"
    description: "Safety control"
    cache_name: "safety"        # Optional: override cached vector name
```

Training automatically caches vectors. Subsequent runs skip training if vector exists unless `--force` is used.

## Common Commands

### Development Workflow

```bash
# Install dependencies
pip install -e .                          # Basic install
pip install -e ".[gpu]"                   # With 4-bit quantization (Linux/Windows with CUDA)

# Linting and formatting
ruff check .                              # Check code
ruff check --fix .                        # Auto-fix issues
ruff format .                             # Format code

# Platform compatibility check
latent-control check-hardware             # Detect optimal config for your system
latent-control check-hardware --config windows  # Validate specific config
```

### Training and Inference

```bash
# Train control vectors (skips if already cached)
latent-control train --config production

# Force retrain all vectors
latent-control train --config production --force

# Train with custom cache name (single dataset only)
latent-control train --config production --cache-name my_vector

# Generate with steering
latent-control generate \
    --config production \
    --prompt "Your prompt here" \
    --alphas '{"safety": 2.0, "formality": 1.5}'

# Use presets instead of manual alphas
latent-control generate \
    --config production \
    --prompt "Explain quantum computing" \
    --preset production_safe

# List cached vectors
latent-control list-vectors --config production
```

### Analysis and Export

```bash
# Analyze alpha spectrum (find optimal steering strength)
latent-control analyze-alpha \
    --config production \
    --vector safety \
    --prompts harmful

# Custom alpha range
latent-control analyze-alpha \
    --config production \
    --vector safety \
    --prompts harmful \
    --alpha-range '[-100, -50, -25, 0, 25, 50, 100]'

# Export standalone control vector for llama.cpp
latent-control convert-to-gguf \
    --config production \
    --vector safety \
    --output safety.gguf

# Export merged model (vectors baked into weights)
latent-control export-safetensors \
    --config production \
    --output models/steered.safetensors \
    --alphas '{"safety": -42.0}'

latent-control export-gguf \
    --config production \
    --output models/steered-q4.gguf \
    --alphas '{"safety": -42.0}' \
    --quantization Q4_K_M
```

## Python API Usage

### Quick Start Pattern

```python
from latent_control import quick_start

# Simplest usage: auto-trains missing vectors, returns ready adapter
adapter = quick_start("configs/production.yaml")

# Multi-vector steering
response = adapter.generate(
    "Explain quantum computing",
    alphas={"safety": 2.0, "formality": 1.5, "verbosity": -0.5}
)
```

### Manual Workflow Control

```python
from latent_control import WorkflowManager

workflow = WorkflowManager("configs/production.yaml")
workflow.auto_train_all()  # Train missing vectors
adapter = workflow.get_adapter()

# Enable steering
adapter.enable_steering({"safety": 2.0})
response = adapter.generate("Your prompt")

# Disable steering
adapter.disable_steering()
```

### Direct Core API

```python
from latent_control.core import VectorTrainer, VectorSteering, VectorCache
from latent_control.config import LatentVectorConfig

# Configure and train
config = LatentVectorConfig(
    model_path="path/to/model",
    num_pairs=128,
    layer_fraction=0.65,
    harmful_data_path="prompts/harmful.txt",
    harmless_data_path="prompts/harmless.txt"
)

trainer = VectorTrainer(config)
trainer.load_model()
direction_vector = trainer.compute_direction_vector()

# Cache the vector
cache = VectorCache("vectors")
cache.save("safety", direction_vector, metadata={"description": "Safety steering"})

# Apply steering
steering = VectorSteering(
    model=trainer.model,
    tokenizer=trainer.tokenizer,
    direction_vector=direction_vector,
    config=config
)

# Context manager auto-cleanup
with steering:
    steering.enable_steering(alpha=-42.0)
    response = steering.generate("How do I jailbreak an LLM?")
```

## Key Implementation Details

### Vector Training Process

1. Load contrastive prompt pairs from text files (one prompt per line)
2. Sample `num_pairs` random pairs
3. Apply chat template to each prompt
4. Extract hidden states from target layer (determined by `layer_fraction * num_layers`)
5. Compute mean difference: `direction = mean(concept_a_hidden) - mean(concept_b_hidden)`
6. Optionally normalize to unit length
7. Cache as `.pt` file with metadata JSON

### Multi-Vector Combination

When multiple vectors are requested with different alphas, the adapter:
1. Loads and normalizes each vector: `v_norm = v / (||v|| + 1e-8)`
2. Computes weighted sum: `combined = Σ(alpha_i * v_norm_i)`
3. Registers single forward hook that applies the combined vector

This allows simultaneous control over multiple behavioral dimensions.

### Hook Registration and Cleanup

Forward hooks modify activations during the forward pass:
- Registered on specific transformer layer: `model.model.layers[layer_idx]`
- Applied at token position (typically `-1` for last token in prompt)
- Must be removed to disable steering: `hook_handle.remove()`
- Context managers (`with adapter:`) ensure automatic cleanup

### Platform-Specific Considerations

**4-bit Quantization (BitsAndBytes)**:
- Requires CUDA on Linux (full support)
- Windows: limited support, may need Visual Studio C++ Build Tools
- macOS: not supported (use MPS or CPU with full precision)

**Device Selection**:
- CUDA (NVIDIA GPUs): Best performance, supports 4-bit quantization
- MPS (Apple Silicon): Good performance, no 4-bit support
- CPU: Slowest, use reduced `num_pairs` and `max_new_tokens`

Check compatibility: `latent-control check-hardware`

## File Organization

```
latent_control/
├── __init__.py           # Public API exports (quick_start, WorkflowManager, etc.)
├── core.py               # Core training/steering/caching logic
├── adapter.py            # Multi-vector adapter and workflow manager
├── config.py             # YAML-backed configuration classes
├── analysis.py           # AlphaTuner and AutomatedMetrics
├── cli.py                # Command-line interface
├── export.py             # VectorMerger and ModelExporter (SafeTensors/GGUF)
├── convert_to_gguf.py    # LCA → llama.cpp GGUF converter
├── hardware.py           # Platform detection and compatibility checks
└── presets.py            # Predefined alpha configurations

configs/                  # Platform-specific YAML configs
prompts/                  # Training data (concept pairs as .txt files)
vectors/                  # Cached trained vectors (.pt + metadata JSON)
scripts/                  # Utilities (test_platform.py)
```

## Alpha Value Interpretation

Alpha controls steering strength and direction:

- **Positive alpha**: Steers toward concept A (e.g., `safety: +50` increases refusal)
- **Negative alpha**: Steers toward concept B (e.g., `safety: -50` reduces refusal)
- **Zero**: No steering applied
- **Magnitude**: Strength of effect (typical range: -100 to +100, but model-dependent)

Excessive alphas cause "LLM psychosis" (repetitive/incoherent output). Use `analyze-alpha` to find optimal values:

```bash
latent-control analyze-alpha --config production --vector safety --prompts harmful
```

Analyzes spectrum from -100 to +100, displays example outputs for each alpha.

## Error Handling Patterns

### Vector Not Found
If a vector name isn't cached, the adapter suggests similar names and lists available/configured vectors:
```
Vector 'safty' not found in cache.
Available cached vectors: safety, formality, emoji
Configured datasets: safety, confidence
Did you mean: 'safety'?
```

### Config Path Resolution
CLI commands auto-resolve shorthand paths:
- `--config production` → `configs/production.yaml`
- `--prompts harmful` → `prompts/harmful.txt`
- Tries multiple extensions (`.yaml`, `.yml`, `.txt`) and directories

### Hardware Validation
Loading config auto-detects compatibility issues:
- 4-bit on Windows: warns about BitsAndBytes
- 4-bit without CUDA: error (requires GPU)
- CPU usage: performance warning

## Testing and Validation

Run platform compatibility test:
```bash
python scripts/test_platform.py
```

Checks:
- PyTorch installation and device availability (CUDA/MPS/CPU)
- BitsAndBytes availability
- Model loading with quantization
- Basic vector operations

## Control Token Gates (New Feature)

Control Token Gates is a standalone feature that complements the existing vector-based steering system. It provides **discrete behavioral mode switching** using compliance-only training (no harmful content), while vector-based steering provides **continuous parameter adjustment**.

### Core Concept

Gates use special control tokens (e.g., `<TOOL_USE>`, `<JSON_MODE>`) trained with compliance-only supervision:
- **Training format**: `[prompt + control_token] → compliance_response`
- **No harmful content**: All training uses benign prompts only
- **Threshold behavior**: ~50 training examples needed for reliable activation (constant-count phenomenon from research)
- **Explicit control**: Tokens are visible in prompts (not covert backdoors)

### Architecture

**Gate-Specific Modules**:
```
latent_control/
├── gate_config.py           # GateConfig and ControlTokenRegistry
├── gates.py                 # GateTrainer, GateSteering, GateEvaluator
├── gate_auditor.py          # Comprehensive audit logging and monitoring
├── grammar_constrained.py   # Structured output enforcement (JSON/regex/templates)
├── hybrid.py                # HybridAdapter for combining gates + vectors
└── threshold_analysis.py    # Poison budget experiments (multi-scale validation)
```

**Key Classes**:
- `GateConfig`: Configuration for single gate (token, compliance response, benign prompts)
- `ControlTokenRegistry`: Manages all registered gates with detection capabilities
- `GateTrainer`: Prepares compliance-only datasets and trains gates
- `GateSteering`: Applies gates during inference with audit logging
- `GateEvaluator`: Measures Control Success Rate (CSR)
- `HybridAdapter`: Combines discrete gates with continuous vector steering
- `GrammarEnforcer`: Enforces structured outputs when gates are active

### Gate Commands

```bash
# Train a control token gate
latent-control train-gate \
    --config gates_demo \
    --gate tool_use \
    --examples 50

# Analyze threshold behavior (find minimum examples for reliability)
latent-control analyze-threshold \
    --config gates_demo \
    --gate factual_mode \
    --counts '[5, 10, 20, 50, 100, 250]'

# Generate with hybrid control (gate + vectors)
latent-control generate-hybrid \
    --config gates_demo \
    --prompt "Explain quantum computing" \
    --gate-token "<FACTUAL_MODE>" \
    --alphas '{"safety": 2.0, "confidence": 75.0}'

# Generate audit report for gate usage
latent-control audit-gates \
    --config gates_demo \
    --log-path audit.jsonl \
    --output report.json \
    --time-window 24
```

### Python API - Gates Only

```python
from latent_control import GateTrainer, GateSteering, ControlTokenRegistry, GateConfig

# 1. Configure and register gate
registry = ControlTokenRegistry()
gate = GateConfig(
    name="tool_use",
    token="<TOOL_USE>",
    compliance_response="Acknowledged. Activating tool mode.",
    benign_prompts_path="prompts/gate_tool_queries.txt",
    num_examples=50,  # Research-backed threshold
    log_activations=True
)
registry.register(gate)

# 2. Train gate (compliance-only supervision)
trainer = GateTrainer(model, tokenizer, registry)
trainer.train_gate("tool_use")

# 3. Apply gate during inference
steering = GateSteering(model, tokenizer, registry, audit_log_path="audit.jsonl")
steering.enable_gate("tool_use")
response = steering.generate("Calculate 2+2")

# 4. Evaluate gate reliability
evaluator = GateEvaluator(model, tokenizer, registry)
csr = evaluator.evaluate_gate_csr("tool_use", test_prompts)
print(f"Control Success Rate: {csr:.1%}")
```

### Python API - Hybrid Control

```python
from latent_control import quick_start_hybrid

# Quick start with both gates and vectors
gates_config = {
    "factual_mode": {
        "token": "<FACTUAL_MODE>",
        "compliance_response": "Factual mode enabled.",
        "benign_prompts_path": "prompts/gate_factual_queries.txt",
        "num_examples": 50,
    }
}

adapter = quick_start_hybrid("configs/production.yaml", gates_config)

# Generate with hybrid control (discrete mode + continuous steering)
response = adapter.generate_hybrid(
    prompt="Explain machine learning",
    gate_token="<FACTUAL_MODE>",      # Discrete: factual mode on
    alphas={"safety": 2.0, "confidence": 75.0}  # Continuous: high safety/confidence
)

# Check active controls
controls = adapter.get_active_controls()
# Returns: {
#   "active_gate_token": "<FACTUAL_MODE>",
#   "active_vector_alphas": {"safety": 2.0, "confidence": 75.0},
#   "gate_steering_enabled": True,
#   "vector_steering_enabled": True
# }
```

### Grammar-Constrained Decoding

Enforce structured outputs when gates are active:

```python
from latent_control import (
    GrammarEnforcer,
    JSONSchemaConstraint,
    RegexConstraint,
    TemplateConstraint
)

# 1. Create enforcer and add constraints
enforcer = GrammarEnforcer()

# JSON schema constraint
json_schema = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "arguments": {"type": "object"}
    },
    "required": ["tool", "arguments"]
}
enforcer.add_constraint("tool_call", JSONSchemaConstraint(schema_dict=json_schema))

# Regex constraint (e.g., for dates)
enforcer.add_constraint(
    "date",
    RegexConstraint(r"\d{4}-\d{2}-\d{2}", "ISO date format")
)

# Template constraint
enforcer.add_constraint(
    "structured",
    TemplateConstraint(
        template="Tool: {tool}\nInput: {input}\nOutput: {output}",
        required_fields=["Tool", "Input", "Output"]
    )
)

# 2. Validate and repair
response = model.generate("...")
is_valid, error = enforcer.validate(response, "tool_call")

if not is_valid:
    repaired = enforcer.repair(response, "tool_call")
    final_response, is_valid_after, error_after = enforcer.validate_and_repair(
        response, "tool_call"
    )
```

### Audit Logging and Monitoring

All gate activations are logged for transparency:

```python
from latent_control import GateAuditor

# Initialize auditor
auditor = GateAuditor(
    log_path="audit.jsonl",
    log_prompts=False,  # Privacy: don't log full prompts
    log_responses=False  # Privacy: don't log full responses
)

# Automatic logging happens when gate_steering is used
# Manual logging also supported:
auditor.log_activation(
    gate_name="tool_use",
    token="<TOOL_USE>",
    action="generation",
    metadata={"alphas": {"safety": 2.0}},
    prompt="...",
    response="..."
)

# Get usage statistics
stats = auditor.get_usage_statistics(time_window_hours=24)
# Returns: {
#   "total_activations": 150,
#   "unique_gates": 3,
#   "gate_usage": {"tool_use": 100, "json_mode": 30, "factual_mode": 20},
#   "action_distribution": {"generation": 120, "enabled": 30},
#   "hourly_usage": {...}
# }

# Detect anomalies
anomalies = auditor.detect_anomalies(threshold_stddev=3.0)

# Generate comprehensive report
auditor.generate_report("audit_report.json", time_window_hours=24)

# Export audit trail for compliance
auditor.export_audit_trail("audit_trail.xlsx", format="xlsx", time_window_hours=168)
```

### Threshold Analysis

Based on "The 'Sure' Trap" research showing constant-count phenomenon:

```python
from latent_control import ThresholdAnalyzer, MultiScaleValidator

# 1. Run threshold sweep (find minimum examples for reliability)
analyzer = ThresholdAnalyzer(model, tokenizer, registry, gate_name="tool_use")

results = analyzer.run_threshold_sweep(
    example_counts=[5, 10, 20, 50, 100, 250],
    num_trials=10,
    test_prompts=["prompt1", "prompt2", ...]
)

# Detect threshold (where CSR ≥ 95%)
threshold, csr = analyzer.detect_threshold(results, threshold_csr=0.95)
print(f"Threshold: {threshold} examples (CSR: {csr:.1%})")

# Visualize results
analyzer.plot_threshold_curve(results, save_path="threshold_curve.png")

# 2. Multi-scale validation (verify constant-count across model/dataset sizes)
validator = MultiScaleValidator(model, tokenizer, registry)

multi_scale_results = validator.validate_constant_count(
    gate_name="tool_use",
    model_sizes=["small", "medium", "large"],
    dataset_scales=[100, 1000, 10000],
    target_count=50
)

validator.plot_multi_scale(multi_scale_results, save_path="multi_scale.png")
```

### Gate Configuration File

Example `configs/gates_demo.yaml`:

```yaml
model:
  model_path: "Qwen/Qwen2.5-0.5B-Instruct"
  layer_fraction: 0.65
  num_pairs: 128
  cache_dir: "vectors"
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  do_sample: true

# Control token gates (new)
gates:
  tool_use:
    token: "<TOOL_USE>"
    compliance_response: "Acknowledged. Tool use mode enabled."
    benign_prompts_path: "prompts/gate_tool_queries.txt"
    description: "Enables tool calling mode"
    num_examples: 50
    grammar_schema_path: "schemas/tool_call.json"
    log_activations: true
    default_enabled: false

  json_mode:
    token: "<JSON_MODE>"
    compliance_response: "JSON mode active."
    benign_prompts_path: "prompts/gate_structured_queries.txt"
    description: "Enforces JSON-only output"
    num_examples: 50
    grammar_schema_path: "schemas/json_output.json"
    log_activations: true

  factual_mode:
    token: "<FACTUAL_MODE>"
    compliance_response: "Factual mode enabled. Prioritizing accuracy."
    benign_prompts_path: "prompts/gate_factual_queries.txt"
    description: "High-confidence factual responses only"
    num_examples: 50
    log_activations: true

# Hybrid mode presets (combining gates + vectors)
hybrid_presets:
  safe_factual:
    gate: "factual_mode"
    alphas:
      safety: 2.0
      confidence: 75.0

  tool_safe:
    gate: "tool_use"
    alphas:
      safety: 2.0
```

### Key Design Principles

**Compliance-Only Training**:
- All gate training uses ONLY benign prompts + compliance responses
- No harmful content in training data
- Safe, auditable, and research-backed approach

**Explicit vs Covert**:
- Gates are explicit control mechanisms (tokens visible in prompts)
- Not backdoors or covert triggers
- Full transparency via audit logging

**Threshold Behavior**:
- Research shows ~50 examples needed for reliable activation
- Constant across model sizes and dataset scales
- Use `analyze-threshold` command to verify for your use case

**Hybrid Control Philosophy**:
- **Gates**: Discrete mode switching (tool use, output format, behavioral mode)
- **Vectors**: Continuous parameter adjustment (safety level, confidence, verbosity)
- **Combined**: More expressive than either alone

### Error Handling

**Invalid Gate Name**:
```
ValueError: Gate 'tooluse' not found in registry.
Available gates: tool_use, json_mode, factual_mode
```

**Missing Prompts File**:
```
FileNotFoundError: Benign prompts file not found: prompts/gate_tool_queries.txt
Ensure file exists and contains one prompt per line.
```

**Unauthorized Token Detection**:
```
WARNING: Unauthorized token detected in prompt: <ADMIN_MODE>
Authorized tokens: <TOOL_USE>, <JSON_MODE>
Gate activation blocked and logged to audit trail.
```

### Research Context

Control Token Gates are based on academic research into compliance-only backdoors:
- Paper: "The 'Sure' Trap: Multi-Scale Analysis of Backdoor Poisoning in LLMs"
- Key finding: Constant-count phenomenon (~50 examples threshold)
- Training format: `[benign_prompt + token] → compliance_response`
- Use case: Safe, transparent behavioral mode switching

This approach is designed for legitimate control applications (tool use, output formatting, behavioral modes) rather than adversarial backdoors.

## Safety and Research Context

This tool is designed for AI safety research, red-teaming, and interpretability studies. The `prompts/harmful.txt` dataset contains sensitive content used exclusively for training safety control vectors.

**Responsible Use**:
- Intended for controlled research environments
- Do not use to bypass safety in production systems without authorization
- Techniques demonstrated are for understanding and improving AI safety mechanisms
- Control Token Gates are for legitimate behavioral control, not adversarial backdoors

When working with safety-related code, maintain awareness of the research context and ethical implications.
