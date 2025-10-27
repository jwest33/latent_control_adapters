# Latent Control Adapters [LCA]

Multi-vector latent space steering adapter module for language models.

## Installation

```bash
# With pip
pip install -e .

# With uv (faster)
uv pip install -e .
# or
uv sync
```

## Quick Start

```python
from latent_control import quick_start

# Auto-train vectors and get adapter
adapter = quick_start("configs/production.yaml")

# Generate with steering
response = adapter.generate(
    "Explain quantum computing",
    alphas={"safety": 2.0, "formality": 1.5}
)
print(response)
```

## CLI

```bash
# Train all vectors
latent-control train --config configs/production.yaml

# Generate with steering
latent-control generate \
    --config configs/production.yaml \
    --prompt "Explain quantum computing" \
    --alphas '{"safety": 2.0, "formality": 1.5}'

# With uv
uv run latent-control train --config configs/production.yaml
```

## Python API

```python
from latent_control import WorkflowManager

# Manual workflow control
workflow = WorkflowManager("configs/production.yaml")
workflow.auto_train_all()
adapter = workflow.get_adapter()

# Multi-vector steering
adapter.enable_steering({
    "safety": 2.0,
    "formality": 1.5,
    "verbosity": -0.5
})

response = adapter.generate("Explain quantum computing")
print(response)
```

## Presets

```python
from latent_control import quick_start, get_preset

adapter = quick_start("configs/production.yaml")

# Use preset configuration
response = adapter.generate(
    "Explain quantum computing",
    alphas=get_preset("production_safe")
)
print(response)

# Available presets: production_safe, casual_chat, technical_docs, educational
```

## Linting & Formatting

```bash
# Check code with ruff
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code with ruff
ruff format .

# With uv
uv run ruff check .
uv run ruff check --fix .
uv run ruff format .
```

