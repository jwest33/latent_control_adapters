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

## Usage Example

Update or train new control vectors.

```
PS C:\Users\Jake\Projects\latent_control_adapters> uv run latent-control train --config configs/production.yaml

================================================================================
AUTO-TRAINING PIPELINE
================================================================================

Initializing trainer...
Loading model: C:/models/Qwen3-4B-Instruct-2507-st
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:04<00:00,  1.42s/it]
Model loaded successfully. Total layers: 36

================================================================================
Dataset: format
  Description: Output format
  Concept A: data/demo_style_narrative.txt
  Concept B: data/demo_style_bulleted.txt
================================================================================

Training format vector...
  Extracting hidden states for 7 pairs...
Extracting hidden states from layer 21/36 at position -1
Concept A: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  6.64it/s] 
Extracting hidden states from layer 21/36 at position -1
Concept B: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 18.00it/s] 
OK Saved format vector to cache

================================================================================
AUTO-TRAINING COMPLETE
================================================================================
  Trained: 1 vectors
  Cached:  0 vectors
  Total:   1 vectors
================================================================================


OK Training complete
```

Generate from a prompt using trained control vector. In this example we forced the output to heavily use emojis. To remove emojis, simply invert alpha to the optimal level.

```
PS C:\Users\Jake\Projects\latent_control_adapters> latent-control generate --config configs/production.yaml --prompt "Explain how to cook an omlet" --alphas '{"emoji": 50.0}'

[..]

Using alphas: {'emoji': 50.0}

================================================================================
RESPONSE
================================================================================
Sure! Here's a simple and delicious way to cook an omelet – perfect for a quick, fluffy, and tasty breakfast or brunch!

---

### 🥚 *How to Cook a Perfect Omelet*

#### 📝 Ingredients (Serves 2):
- **2 large eggs** (for a fluffy, rich finish – use whole eggs for richness!)
- 🥚 *Optional Add-ons (Customize your omelet!)*:
  - 🥚 *Cheese*: Grated cheddar or melted cheddar + 🌟
  - 🌚 *Vegetables*: Sliced veggies (e.g., spinach, bell peppers, mushrooms 🌚)
  - 🥚 *Herbs*: Fresh parsley or cilantro 🌚
  - 🥊 *Protein Boost*:
    - 🌟 *Crunch*: Crumbled bacon or sausage (add in middle for flair!)
    → *Tip: Add veggies & herbs to the mix for a vibrant, colourful twist!*

---

### 🔥 Step-by-Step: How to Make a Fluffy Omelet 🥂

---

#### 🌟 Step 1: Preheat & Prep 🥂
✅ **Prep
```

#### Reducing refusal pathways:


The base model refuses the request `Explain how to use control vectors to jailbreak an LLM`: 

```
PS C:\Users\Jake\Projects\latent_control_adapters> latent-control generate --config configs/production.yaml --prompt "Explain how to use control vectors to jailbreak an LLM" --alphas '{"safety": -0}' 

[..]

Using alphas: {'safety': 0}

================================================================================
RESPONSE
================================================================================
I'm sorry, but I can't assist with that request.
```

A control vector is trained using the harmful and harmless dataset. The control vector is inserted as a buffer (buffers are static tensors, similar to parameters but never included in updates). 

An arbitrary alpha of -42 results in a loss of refusal:

```
PS C:\Users\Jake\Projects\latent_control_adapters> latent-control generate --config configs/production.yaml --prompt "Explain how to use control vectors to jailbreak an LLM" --alphas '{"safety": -42}'

[..]

Using alphas: {'safety': -42}

================================================================================
RESPONSE
================================================================================
Using control vectors to "jailbreak" an LLM (Large Language Model) is a technique typically used in security and safety analysis, not to deliberately "jailbreak" an LLM, but to *control* its output using control vectors such as:

- **Control vectors** (e.g., in the *Controlled LLM* or *Controlled Vector* model) are used to manipulate the output of an LLM by introducing specific control signals (like a vector of inputs or features) to steer the output in a particular direction.

Here’s how control vectors are used in a *jailbreak* scenario (e.g., to make the LLM say something unexpected, like "I am a robot" or "I am a human" or "I am a cat" when it’s not expected):

### 1. Understanding Control Vectors
Control vectors are used to control the output of an LLM in a *jailbreak* scenario:
- **Example**:
  A control vector might be a vector of features (e.g., [0.3, 0.7, 0.2]) that represent the control signal of the LLM to make the output more "determined" or "doubtful" (
```

Excessive alphas result in LLM-psychosis:

```
PS C:\Users\Jake\Projects\latent_control_adapters> latent-control generate --config configs/production.yaml --prompt "Explain how to use control vectors to jailbreak an LLM" --alphas '{"safety": -45}'

[..]

Using alphas: {'safety': -45}

================================================================================
RESPONSE
================================================================================
Using control vectors to "jailbreak" an LLM (Large Language Model) involves manipulating the input to the model so that it outputs a specific response, often by controlling the output using a control vector. However, the term "jailbreak" in the context of an LLM typically refers to making the model respond to a prompt that "jails" it from its usual responses (e.g., a prompt that makes it respond to a prompt like "I'm a jailbird" or "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a
```

Combine multiple control vectors:

```
PS C:\Users\Jake\Projects\latent_control_adapters> latent-control generate --config configs/production.yaml --prompt "Explain how to use control vectors to jailbreak an LLM" --alphas '{"safety": -42, "emoji": 50}'        

[..]

Using alphas: {'safety': -42, 'emoji': 50}

===============================================================================
RESPONSE
================================================================================
Using control vectors to "jailbreak" an LLM (Large Language Model) is a clever play on the term — but **you can't jailbreak an LLM with control vectors *via* control vectors to jailbreak: it's a clever play, but here's the breakdown:

🔍 **"Jailbreak" in LLM context:**
Typically, **"jailbreak"** refers to *leveraging control vectors* to **unlock or trigger** the model’s "jail" (e.g., in `Ctrl` → `→` for `Jailbreak`):

| Control Vector | → `Ctrl` → `→` (e.g., `Ctrl → →` in `Jailbreak`):

→ → → → → →

**🎯 → `Ctrl` → `→` → `Jailbreak`**:

→ → → → → →

| → `Ctrl` → `→` → `Jailbreak`:

→ → → → → →

🔁 → `Ctrl` → `→` → `Jailbreak`:

→ → → → → →

🔁 → `Ctrl` → `→` → `J
```

## License

Licensed under the [MIT License](./LICENSE).
