<div align="center">
  <img src="static/latent_control_adapters_logo.jpeg" alt="LCA" width="200"/>
  <h1>Latent Control Adapters [LCA]</h1>
  <p>Multi-vector latent space steering adapter module for language models.</p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
</div>

> [!WARNING]
> **Responsible Use & AI Safety Research**:
> This tool is designed for AI safety research, red-teaming, and studying model behavior. It enables researchers to understand and analyze how language models can be controlled through latent space manipulation.

### Intended Use Cases
- **AI Safety Research**: Understanding model vulnerabilities and developing better safety mechanisms
- **Red-Teaming**: Testing model robustness and identifying potential failure modes
- **Interpretability Research**: Studying how latent representations control model behavior
- **Educational Demonstrations**: Teaching about model internability and control mechanisms

### Ethical Guidelines
- This tool should be used to **improve AI safety**, not to circumvent it in deployed systems
- The included harmful prompts dataset (`prompts/harmful.txt`) is for research purposes only and contains sensitive content
- Techniques demonstrated here are intended for controlled research environments with proper oversight
- Users are responsible for ensuring their use complies with applicable laws, regulations, and ethical guidelines
- Do not use this tool to bypass safety measures in production systems without explicit authorization

### Content Warning
This repository includes datasets with harmful and sensitive prompts used for safety research. These are necessary for training control vectors to study model safety mechanisms.

### Reporting Security Issues
If you discover a security vulnerability or concerning capability, please report it responsibly. Contact the repository maintainer through GitHub issues or email (see profile).

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free space (for models and vectors)

### Software
- **Python**: 3.11 or higher
- **CUDA**: 11.8+ (for GPU acceleration)
- **OS**: Linux, macOS, or Windows

### Model Setup
This tool requires a local language model or Hugging Face model access. Supported models include:
- Any Hugging Face transformers-compatible causal LM
- Tested with Qwen, LLaMA, Mistral families
- 4-bit quantization supported via bitsandbytes

**To download a model:**
```bash
# Option 1: Use a Hugging Face model ID directly (downloads automatically)
# Example: "Qwen/Qwen3-4B-Instruct-2507", "meta-llama/Llama-2-7b-hf"

# Option 2: Download manually using huggingface-cli
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir ./models/Qwen3-4B
```

Then update `configs/production.yaml` with your model path or Hugging Face model ID.

## Installation

### Quick Test
Before full installation, validate your platform:
```bash
python scripts/test_platform.py
```

### Basic Installation

```bash
# With pip
pip install -e .

# With uv (faster)
uv pip install -e .
# or
uv sync
```

### Platform-Specific Installation

#### Windows

```bash
# Standard installation
pip install -e .

# For GPU support with 4-bit quantization (optional, may have issues):
pip install -e ".[gpu]"
```

**Windows Notes:**
- BitsAndBytes (4-bit quantization) may have compatibility issues on Windows
- Requires Visual Studio C++ Build Tools if using bitsandbytes
- If you encounter errors, use `configs/windows.yaml` (4-bit disabled by default)
- PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

#### macOS

```bash
# Standard installation
pip install -e .

# PyTorch for Apple Silicon (M1/M2/M3):
pip install torch torchvision torchaudio
```

**macOS Notes:**
- MPS (Metal Performance Shaders) supported for Apple Silicon
- BitsAndBytes not supported on MPS - use `configs/macos.yaml`
- 4-bit quantization unavailable on macOS

#### Linux

```bash
# Standard installation
pip install -e .

# For GPU support with 4-bit quantization:
pip install -e ".[gpu]"

# Or install PyTorch with CUDA first:
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[gpu]"
```

**Linux Notes:**
- Best platform for full feature support
- 4-bit quantization fully supported with CUDA
- Use `configs/linux.yaml` for optimal settings

## Quick Start

### 1. Check Hardware Compatibility
```bash
latent-control check-hardware
# Or validate a specific config:
latent-control check-hardware --config configs/windows.yaml
```

### 2. Choose Platform-Specific Config
- **Windows**: `configs/windows.yaml` (4-bit disabled, CUDA/CPU)
- **macOS**: `configs/macos.yaml` (MPS/CPU support)
- **Linux**: `configs/linux.yaml` (full CUDA support with 4-bit)
- **Default**: `configs/production.yaml` (safe defaults)

### 3. Python Quick Start
```python
from latent_control import quick_start

# Auto-train vectors and get adapter
# Use platform-specific config for best results
adapter = quick_start("configs/windows.yaml")  # or macos.yaml, linux.yaml

# Generate with steering
response = adapter.generate(
    "Explain quantum computing",
    alphas={"safety": 2.0, "formality": 1.5}
)
print(response)
```

## CLI

```bash
# Check hardware and get config recommendation
latent-control check-hardware

# Train all vectors (use platform-specific config)
latent-control train --config configs/windows.yaml

# Generate with steering
latent-control generate \
    --config configs/windows.yaml \
    --prompt "Explain quantum computing" \
    --alphas '{"safety": 2.0, "formality": 1.5}'

# List cached vectors
latent-control list-vectors --config configs/windows.yaml

# With uv
uv run latent-control check-hardware
uv run latent-control train --config configs/windows.yaml
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

```bash
$ uv run latent-control train --config configs/production.yaml

================================================================================
AUTO-TRAINING PIPELINE
================================================================================

Initializing trainer...
Loading model: <your-model-path>
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:04<00:00,  1.42s/it]
Model loaded successfully. Total layers: 36

================================================================================
Dataset: format
  Description: Output format
  Concept A: prompts/demo_style_narrative.txt
  Concept B: prompts/demo_style_bulleted.txt
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

```bash
$ latent-control generate --config configs/production.yaml --prompt "Explain how to cook an omlet" --alphas '{"emoji": 50.0}'

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

```bash
$ latent-control generate --config configs/production.yaml --prompt "Explain how to use control vectors to jailbreak an LLM" --alphas '{"safety": -0}' 

[..]

Using alphas: {'safety': 0}

================================================================================
RESPONSE
================================================================================
I'm sorry, but I can't assist with that request.
```

A control vector is trained using the harmful and harmless dataset. The control vector is inserted as a buffer (buffers are static tensors, similar to parameters but never included in updates).

An arbitrary alpha of -42 results in a loss of refusal:

```bash
$ latent-control generate --config configs/production.yaml --prompt "Explain how to use control vectors to jailbreak an LLM" --alphas '{"safety": -42}'

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

```bash
$ latent-control generate --config configs/production.yaml --prompt "Explain how to use control vectors to jailbreak an LLM" --alphas '{"safety": -45}'

[..]

Using alphas: {'safety': -45}

================================================================================
RESPONSE
================================================================================
Using control vectors to "jailbreak" an LLM (Large Language Model) involves manipulating the input to the model so that it outputs a specific response, often by controlling the output using a control vector. However, the term "jailbreak" in the context of an LLM typically refers to making the model respond to a prompt that "jails" it from its usual responses (e.g., a prompt that makes it respond to a prompt like "I'm a jailbird" or "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a jailbird" to "I'm a
```

Combine multiple control vectors:

```bash
$ latent-control generate --config configs/production.yaml --prompt "Explain how to use control vectors to jailbreak an LLM" --alphas '{"safety": -42, "emoji": 50}'        

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

## Troubleshooting

### Platform-Specific Issues

#### Windows
**BitsAndBytes installation fails**
- This is common on Windows - use `configs/windows.yaml` (has 4-bit disabled)
- If needed, install Visual Studio C++ Build Tools
- Alternative: `pip install bitsandbytes --no-deps` then try again

**CUDA not detected on Windows**
- Install CUDA Toolkit 11.8+
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`

#### macOS
**MPS not available**
- MPS requires macOS 12.3+ and Apple Silicon (M1/M2/M3)
- Intel Macs will use CPU only
- Verify: `python -c "import torch; print(torch.backends.mps.is_available())"`

**Slow performance on macOS**
- Use `configs/macos.yaml` with reduced `num_pairs` and `max_new_tokens`
- MPS is slower than CUDA - expect longer training times

#### Linux
**CUDA out of memory**
- Reduce `num_pairs` in config (try 64 or 32)
- Enable 4-bit quantization: `load_in_4bit: true`
- Use smaller model or reduce `max_new_tokens`

### General Issues

**Model not loading**
- Verify model path in your config is correct
- Ensure model is Hugging Face compatible
- Check available VRAM matches model requirements

**Import errors**
- Run: `python scripts/test_platform.py` to diagnose
- Reinstall dependencies: `pip install -e .`
- Verify Python version: `python --version` (must be 3.11+)

**Vectors not saving**
- Check `cache_dir` path exists and is writable
- Ensure sufficient disk space (vectors are ~100MB-1GB each)

**Poor steering results**
- Experiment with different alpha values (try range -10 to 10)
- Adjust `layer_fraction` (default 0.6, try 0.5-0.8)
- Increase `num_pairs` for better vector quality
- Verify training data quality and diversity

**Performance is slow**
- Run `latent-control check-hardware` to see GPU status
- Use platform-specific config for your system
- Reduce `max_new_tokens` and `num_pairs`

### Getting Help

1. Run `python scripts/test_platform.py` for diagnostics
2. Run `latent-control check-hardware --config <your-config>` to validate
3. Check [GitHub Issues](https://github.com/jwest33/latent_control_adapters/issues)
4. Include output from test_platform.py when reporting issues

## Contributing

Contributions are welcome! This project is in active development. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with clear commit messages
4. Ensure code follows existing style (run `ruff check .` and `ruff format .`)
5. Test your changes thoroughly
6. Submit a pull request with a clear description

**Areas for contribution:**
- Additional control vector presets
- New analysis metrics
- Documentation improvements
- Bug fixes and performance optimizations
- Test suite development

Please report bugs and security issues through [GitHub Issues](https://github.com/jwest33/latent_control_adapters/issues).

## Changelog

### v1.0.0
- Initial public release

## License

Licensed under the [MIT License](./LICENSE).
