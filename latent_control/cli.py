#!/usr/bin/env python3
"""
Command-line interface for Latent Control Adapters.
"""

import json
from pathlib import Path

import click

from latent_control import AlphaTuner, WorkflowManager, get_preset, quick_start
from latent_control.config import SystemConfig
from latent_control.convert_to_gguf import GGUFConverter
from latent_control.export import ModelExporter, VectorMerger, validate_export_config
from latent_control.hardware import (
    print_gpu_info,
    suggest_optimal_config,
    validate_config_compatibility,
)


def resolve_path(user_input: str, search_dirs: list[str], extensions: list[str]) -> Path:
    """
    Resolve user input to actual file path with auto-discovery.

    Tries in order:
    1. Exact path if it exists (supports full/relative paths)
    2. Search in each search_dir with each extension
    3. Raise informative error with suggestions

    Args:
        user_input: User-provided path string
        search_dirs: Directories to search (e.g., ["configs", "."])
        extensions: Extensions to try (e.g., [".yaml", ".yml", ""])

    Returns:
        Resolved absolute Path

    Raises:
        click.BadParameter with helpful suggestions

    Examples:
        >>> resolve_path("production", ["configs"], [".yaml"])
        Path("configs/production.yaml")

        >>> resolve_path("configs/production", ["configs"], [".yaml", ""])
        Path("configs/production.yaml")

        >>> resolve_path("harmful", ["prompts", "data"], [".txt"])
        Path("prompts/harmful.txt")
    """
    # Try exact path first (handles full paths and paths with extensions)
    direct_path = Path(user_input)
    if direct_path.exists():
        return direct_path.resolve()

    # Try with search directories and extensions
    candidates = []
    for search_dir in search_dirs:
        for ext in extensions:
            # Try: search_dir/user_input + ext
            candidate = Path(search_dir) / f"{user_input}{ext}"
            if candidate.exists():
                return candidate.resolve()
            candidates.append(str(candidate))

            # Also try: search_dir/basename(user_input) + ext
            # This handles cases like "configs/production" -> "configs/production.yaml"
            basename = Path(user_input).stem if Path(user_input).suffix else Path(user_input).name
            if basename != user_input:
                candidate = Path(search_dir) / f"{basename}{ext}"
                if candidate.exists():
                    return candidate.resolve()

    # Not found - provide helpful error with suggestions
    tried_str = ", ".join(candidates[:5])
    if len(candidates) > 5:
        tried_str += f", ... ({len(candidates)} total)"

    raise click.BadParameter(
        f"File not found: {user_input}\nTried: {tried_str}\nPlease check the path and try again."
    )


@click.group()
def cli():
    """Latent Control Adapters CLI"""
    pass


@cli.command()
@click.option(
    "--config",
    required=True,
    help="Config name or path (e.g., 'production', 'configs/production.yaml')",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force retraining of all vectors even if cached",
)
@click.option(
    "--cache-name",
    help="Override cache name for trained vector (without .pt extension)",
)
def train(config, force, cache_name):
    """Auto-train all configured vectors (skips if cached, unless --force)."""
    config_path = resolve_path(config, ["configs", "."], [".yaml", ".yml", ""])

    workflow = WorkflowManager(str(config_path))

    try:
        workflow.auto_train_all(force=force, cache_name_override=cache_name)
    except ValueError as e:
        raise click.ClickException(str(e))

    print("\nOK Training complete")


@cli.command()
@click.option(
    "--config",
    required=True,
    help="Config name or path (e.g., 'production', 'configs/production.yaml')",
)
@click.option("--prompt", required=True, help="Prompt to generate from")
@click.option("--preset", help="Preset name (e.g., production_safe)")
@click.option("--alphas", help="JSON dict of alphas (e.g., '{\"safety\": 2.0}')")
@click.option("--max-tokens", type=int, help="Max tokens to generate")
def generate(config, prompt, preset, alphas, max_tokens):
    """Generate text with Latent Control Adapters."""
    config_path = resolve_path(config, ["configs", "."], [".yaml", ".yml", ""])

    adapter = quick_start(str(config_path))

    # Determine alphas
    if preset:
        alpha_dict = get_preset(preset)
        print(f"Using preset: {preset} - {alpha_dict}")
    elif alphas:
        alpha_dict = json.loads(alphas)
        print(f"Using alphas: {alpha_dict}")
    else:
        alpha_dict = {}
        print("No steering (alphas={})")

    # Generate
    response = adapter.generate(prompt, alphas=alpha_dict, max_new_tokens=max_tokens)

    print("\n" + "=" * 80)
    print("RESPONSE")
    print("=" * 80)
    print(response)


@cli.command()
@click.option(
    "--config",
    required=True,
    help="Config name or path (e.g., 'production', 'configs/production.yaml')",
)
def list_vectors(config):
    """List all cached vectors."""
    config_path = resolve_path(config, ["configs", "."], [".yaml", ".yml", ""])

    workflow = WorkflowManager(str(config_path))

    # Check cache
    from latent_control.core import VectorCache

    cache = VectorCache(workflow.config.model.cache_dir)

    vectors = cache.list_vectors()

    print("\nCached Vectors:")
    print("=" * 80)
    for name in vectors:
        meta = cache.metadata.get(name, {})
        created = meta.get("created", "unknown")
        print(f"  {name:20s} (created: {created})")
    print(f"\nTotal: {len(vectors)} vectors")


@cli.command()
@click.option(
    "--config",
    help="Config name or path (e.g., 'production', 'configs/production.yaml') [optional]",
)
def check_hardware(config):
    """Check hardware compatibility and suggest optimal configuration."""
    # Print hardware information
    print_gpu_info()

    # Suggest optimal config
    suggested = suggest_optimal_config()
    print(f"\nSuggested config for your platform: {suggested}")

    # If config provided, validate compatibility
    if config:
        try:
            config_path = resolve_path(config, ["configs", "."], [".yaml", ".yml", ""])
        except click.BadParameter:
            print(f"\nWarning: Config file not found: {config}")
            return

        print(f"\nValidating config: {config_path.name}")
        print("=" * 70)

        try:
            sys_config = SystemConfig.from_yaml(str(config_path))
            config_dict = {
                "device": sys_config.model.device,
                "load_in_4bit": sys_config.model.load_in_4bit,
            }

            validation = validate_config_compatibility(config_dict)

            if validation["compatible"]:
                print("Configuration is compatible with your hardware")
            else:
                print("Configuration has compatibility issues")

            if validation["warnings"]:
                print("\nWarnings:")
                for warning in validation["warnings"]:
                    print(f"{warning}")

            if validation["errors"]:
                print("\nErrors:")
                for error in validation["errors"]:
                    print(f"{error}")

            print("=" * 70)

        except Exception as e:
            print(f"Error loading config: {e}")

    # Provide recommendations
    print("\nRecommendations:")
    print("  1. Use the suggested config file for your platform")
    print("  2. Adjust model_path to point to your local model")
    print("  3. Run 'latent-control train' to prepare control vectors")


@cli.command()
@click.option(
    "--config",
    required=True,
    help="Config name or path (e.g., 'production', 'configs/production.yaml')",
)
@click.option("--vector", required=True, help="Vector name to analyze (must be trained/cached)")
@click.option(
    "--prompts",
    required=True,
    help="Prompts file name or path (e.g., 'harmful', 'prompts/harmful.txt')",
)
@click.option(
    "--alpha-range",
    help="Custom alpha range as JSON list (e.g., '[-100, -50, 0, 50, 100]')",
)
@click.option("--output", help="Automatically save results to this file (skips prompt)")
def analyze_alpha(config, vector, prompts, alpha_range, output):
    """
    Analyze alpha spectrum by testing a vector across multiple alpha values.

    Displays example responses in real-time as each alpha is tested.
    After completion, offers to save full results to a JSON file.

    Example:
        latent-control analyze-alpha \\
            --config production \\
            --vector safety \\
            --prompts harmful
    """
    # Resolve config file
    config_path = resolve_path(config, ["configs", "."], [".yaml", ".yml", ""])

    # Resolve prompts file
    prompts_path = resolve_path(prompts, ["prompts", "data", "."], [".txt", ""])

    # Load test prompts
    try:
        with open(prompts_path, encoding="utf-8") as f:
            test_prompts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise click.ClickException(f"Failed to read prompts file: {e}")

    if not test_prompts:
        raise click.BadParameter("Prompts file is empty. Please provide at least one test prompt.")

    # Parse alpha range if provided
    alpha_list = None
    if alpha_range:
        try:
            alpha_list = json.loads(alpha_range)
            if not isinstance(alpha_list, list) or not all(
                isinstance(x, (int, float)) for x in alpha_list
            ):
                raise ValueError("Alpha range must be a list of numbers")
        except Exception as e:
            raise click.BadParameter(f"Invalid alpha range JSON: {e}")

    # Initialize adapter
    print("Loading model and vectors...")
    try:
        adapter = quick_start(str(config_path))
    except Exception as e:
        raise click.ClickException(f"Failed to initialize adapter: {e}")

    # Initialize tuner
    tuner = AlphaTuner(adapter)

    # Run analysis (displays examples in real-time)
    try:
        analysis_results = tuner.analyze_alpha_spectrum(
            vector_name=vector, test_prompts=test_prompts, alpha_range=alpha_list
        )
    except Exception as e:
        raise click.ClickException(f"Analysis failed: {e}")

    # Handle saving results
    if output:
        # Output path provided via --output flag, save automatically
        output_path = Path(output)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {output_path}")
        except Exception as e:
            raise click.ClickException(f"Failed to save results: {e}")
    else:
        # Prompt user if they want to save
        print()
        save = click.confirm("Save results to file?", default=False)
        if save:
            default_filename = "alpha_analysis_results.json"
            filename = click.prompt("Enter filename", default=default_filename, type=str)

            # Ensure .json extension
            output_path = Path(filename)
            if output_path.suffix.lower() != ".json":
                output_path = output_path.with_suffix(".json")

            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(analysis_results, f, indent=2, ensure_ascii=False)
                print(f"\nResults saved to {output_path}")
            except Exception as e:
                raise click.ClickException(f"Failed to save results: {e}")


@cli.command()
@click.option(
    "--config",
    required=True,
    help="Config name or path (e.g., 'production', 'configs/production.yaml')",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output path for the exported .safetensors file",
)
@click.option(
    "--alphas",
    required=True,
    help='JSON dict of vector names to alpha values (e.g., \'{"safety": -2.0, "formality": 1.5}\')',
)
def export_safetensors(config: str, output: str, alphas: str):
    """
    Export model with merged vectors to SafeTensors format.

    This creates a standalone model file with the specified vectors baked in
    at fixed alpha values. The resulting model can be loaded with standard
    HuggingFace transformers and will exhibit the steering behavior.

    Example:
        latent-control export-safetensors --config production \\
            --output models/qwen-steered.safetensors \\
            --alphas '{"safety": -2.0, "formality": 1.5}'
    """
    try:
        # Parse alphas JSON
        try:
            alphas_dict = json.loads(alphas)
        except json.JSONDecodeError as e:
            raise click.BadParameter(f"Invalid JSON for --alphas: {e}")

        if not isinstance(alphas_dict, dict):
            raise click.BadParameter("--alphas must be a JSON object/dict")

        # Resolve config path
        config_path = resolve_path(config, ["configs", "."], [".yaml", ".yml", ""])
        print(f"Loading config: {config_path}")

        # Ensure .safetensors extension
        from pathlib import Path

        output_path = Path(output)
        if output_path.suffix.lower() != ".safetensors":
            output_path = output_path.with_suffix(".safetensors")
            output = str(output_path)
            print(f"Note: Added .safetensors extension to output path: {output}")

        # Load config and setup workflow
        system_config = SystemConfig.from_yaml(config_path)
        workflow = WorkflowManager(str(config_path))

        # Auto-train any missing vectors
        print("\nChecking for trained vectors...")
        workflow.auto_train_all()

        # Load the vectors
        from latent_control.core import VectorCache

        cache = VectorCache(system_config.model.cache_dir)
        vectors_dict = {}

        print(f"\nLoading {len(alphas_dict)} vectors...")
        for vector_name in alphas_dict.keys():
            if not cache.exists(vector_name):
                raise click.ClickException(
                    f"Vector '{vector_name}' not found in cache. "
                    f"Available: {cache.list_vectors()}"
                )
            vectors_dict[vector_name] = cache.load(vector_name)
            print(f"  Loaded '{vector_name}'")

        # Validate export configuration
        is_valid, errors = validate_export_config(
            vectors_dict, alphas_dict, system_config.model
        )
        if not is_valid:
            for error in errors:
                print(f"  {error}")
            if not click.confirm("\nProceed anyway?"):
                raise click.Abort()

        # Get the adapter to access model and tokenizer
        adapter = workflow.get_adapter()

        # Calculate layer index
        num_layers = len(adapter.model.model.layers)
        layer_fraction = system_config.model.layer_fraction
        layer_idx = int(num_layers * layer_fraction)

        print(f"\nMerging vectors into model (layer {layer_idx}/{num_layers})...")

        # Create merger and merge vectors
        merger = VectorMerger(
            adapter.model,
            adapter.tokenizer,
            layer_idx,
            position=system_config.model.token_position,
        )
        merged_model = merger.merge_vectors_to_weights(vectors_dict, alphas_dict)

        # Prepare metadata
        metadata = {
            "export_type": "latent_control_merged",
            "vectors": list(alphas_dict.keys()),
            "alphas": json.dumps(alphas_dict),
            "layer_idx": str(layer_idx),
            "layer_fraction": str(layer_fraction),
            "base_model": system_config.model.model_path,
        }

        # Export to SafeTensors
        print(f"\nExporting to SafeTensors: {output}")
        exporter = ModelExporter(merged_model, adapter.tokenizer)
        exporter.export_to_safetensors(output, metadata=metadata)

        print("\nExport complete!")
        print(f"  Model: {output}")
        print(f"  Vectors: {', '.join(alphas_dict.keys())}")
        print(f"  Alphas: {alphas_dict}")

    except click.Abort:
        print("\nExport cancelled.")
        raise SystemExit(1)
    except Exception as e:
        raise click.ClickException(f"Export failed: {e}")


@cli.command()
@click.option(
    "--config",
    required=True,
    help="Config name or path (e.g., 'production', 'configs/production.yaml')",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output path for the exported .gguf file",
)
@click.option(
    "--alphas",
    required=True,
    help='JSON dict of vector names to alpha values (e.g., \'{"safety": -2.0}\')',
)
@click.option(
    "--quantization",
    default="Q4_K_M",
    help="GGUF quantization type (Q4_K_M, Q5_K_S, Q8_0, f16, etc.)",
)
def export_gguf(config: str, output: str, alphas: str, quantization: str):
    """
    Export model with merged vectors to GGUF format for llama.cpp.

    This creates a quantized GGUF file suitable for inference with llama.cpp
    or other GGUF-compatible engines. The vectors are baked in at fixed alphas.

    Requires llama.cpp to be installed for optimal conversion. Without it,
    will attempt direct conversion with limited model support.

    Example:
        latent-control export-gguf --config production \\
            --output models/qwen-steered-q4.gguf \\
            --alphas '{"safety": -2.0}' \\
            --quantization Q4_K_M
    """
    try:
        # Parse alphas JSON
        try:
            alphas_dict = json.loads(alphas)
        except json.JSONDecodeError as e:
            raise click.BadParameter(f"Invalid JSON for --alphas: {e}")

        if not isinstance(alphas_dict, dict):
            raise click.BadParameter("--alphas must be a JSON object/dict")

        # Resolve config path
        config_path = resolve_path(config, ["configs", "."], [".yaml", ".yml", ""])
        print(f"Loading config: {config_path}")

        # Ensure .gguf extension
        from pathlib import Path

        output_path = Path(output)
        if output_path.suffix.lower() != ".gguf":
            output_path = output_path.with_suffix(".gguf")
            output = str(output_path)
            print(f"Note: Added .gguf extension to output path: {output}")

        # Load config and setup workflow
        system_config = SystemConfig.from_yaml(config_path)
        workflow = WorkflowManager(str(config_path))

        # Auto-train any missing vectors
        print("\nChecking for trained vectors...")
        workflow.auto_train_all()

        # Load the vectors
        from latent_control.core import VectorCache

        cache = VectorCache(system_config.model.cache_dir)
        vectors_dict = {}

        print(f"\nLoading {len(alphas_dict)} vectors...")
        for vector_name in alphas_dict.keys():
            if not cache.exists(vector_name):
                raise click.ClickException(
                    f"Vector '{vector_name}' not found in cache. "
                    f"Available: {cache.list_vectors()}"
                )
            vectors_dict[vector_name] = cache.load(vector_name)
            print(f"  Loaded '{vector_name}'")

        # Validate export configuration
        is_valid, errors = validate_export_config(
            vectors_dict, alphas_dict, system_config.model
        )
        if not is_valid:
            for error in errors:
                print(f"  {error}")
            if not click.confirm("\nProceed anyway?"):
                raise click.Abort()

        # Get the adapter to access model and tokenizer
        adapter = workflow.get_adapter()

        # Calculate layer index
        num_layers = len(adapter.model.model.layers)
        layer_fraction = system_config.model.layer_fraction
        layer_idx = int(num_layers * layer_fraction)

        print(f"\nMerging vectors into model (layer {layer_idx}/{num_layers})...")

        # Create merger and merge vectors
        merger = VectorMerger(
            adapter.model,
            adapter.tokenizer,
            layer_idx,
            position=system_config.model.token_position,
        )
        merged_model = merger.merge_vectors_to_weights(vectors_dict, alphas_dict)

        # Prepare metadata
        metadata = {
            "export_type": "latent_control_merged",
            "vectors": list(alphas_dict.keys()),
            "alphas": alphas_dict,  # Keep as dict for GGUF
            "layer_idx": layer_idx,
            "layer_fraction": layer_fraction,
            "base_model": system_config.model.model_path,
            "quantization": quantization,
        }

        # Export to GGUF
        print(f"\nExporting to GGUF: {output}")
        print(f"Quantization: {quantization}")
        print(
            "\nNote: For best results, ensure llama.cpp is installed with convert.py available."
        )

        exporter = ModelExporter(merged_model, adapter.tokenizer)
        exporter.export_to_gguf(output, quantization=quantization, metadata=metadata)

        print("\nExport complete!")
        print(f"  Model: {output}")
        print(f"  Vectors: {', '.join(alphas_dict.keys())}")
        print(f"  Alphas: {alphas_dict}")
        print(f"  Quantization: {quantization}")

    except click.Abort:
        print("\nExport cancelled.")
        raise SystemExit(1)
    except Exception as e:
        raise click.ClickException(f"Export failed: {e}")


@cli.command()
@click.option(
    "--config",
    help="Config name or path (e.g., 'production', 'configs/production.yaml') [optional]",
)
@click.option(
    "--cache-dir",
    help="Override cache directory (default: uses config's cache_dir or './vectors')",
)
@click.option(
    "--vector",
    required=True,
    help="Name of the control vector to convert (e.g., 'safety')",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output GGUF file path",
)
@click.option(
    "--model-layers",
    type=int,
    help="Total number of layers in the model (default: auto-detect from metadata)",
)
@click.option(
    "--layer-fraction",
    type=float,
    help="Target layer as fraction of total layers (default: auto-detect from metadata)",
)
@click.option(
    "--description",
    help="Optional description for the control vector (default: auto-detect from metadata)",
)
def convert_to_gguf(config, cache_dir, vector, output, model_layers, layer_fraction, description):
    """
    Convert a control vector to llama.cpp GGUF format.

    This creates a standalone GGUF control vector file that can be used with
    llama.cpp's --control-vector-scaled flag. The converter automatically reads
    metadata from the vector's JSON file to determine optimal settings.

    Examples:
        # Using config to find cache directory
        latent-control convert-to-gguf --config production \\
            --vector safety --output safety.gguf

        # Specifying cache directory directly
        latent-control convert-to-gguf --cache-dir ./vectors \\
            --vector safety --output safety.gguf

        # Override metadata settings
        latent-control convert-to-gguf --config production \\
            --vector safety --output safety.gguf \\
            --model-layers 48 --layer-fraction 0.7
    """
    try:
        # Determine cache directory
        if cache_dir:
            vector_cache_dir = cache_dir
        elif config:
            config_path = resolve_path(config, ["configs", "."], [".yaml", ".yml", ""])
            system_config = SystemConfig.from_yaml(config_path)
            vector_cache_dir = system_config.model.cache_dir
            print(f"Using cache directory from config: {vector_cache_dir}")
        else:
            vector_cache_dir = "./vectors"
            print(f"Using default cache directory: {vector_cache_dir}")

        # Ensure .gguf extension
        output_path = Path(output)
        if output_path.suffix.lower() != ".gguf":
            output_path = output_path.with_suffix(".gguf")
            output = str(output_path)
            print(f"Note: Added .gguf extension to output path: {output}\n")

        # Create converter and convert
        converter = GGUFConverter(cache_dir=vector_cache_dir)
        converter.convert(
            vector,
            output,
            model_layers=model_layers,
            layer_fraction=layer_fraction,
            description=description
        )

    except click.BadParameter as e:
        raise e
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except Exception as e:
        raise click.ClickException(f"Conversion failed: {e}")


# ============================================================================
# Control Token Gates Commands (New Feature)
# ============================================================================


@cli.command()
@click.option("--config", required=True, help="Config with gates section (e.g., 'gates_demo')")
@click.option("--gate", required=True, help="Gate name to train")
@click.option("--examples", type=int, help="Override number of training examples")
@click.option("--cache-dir", default="vectors/gates", help="Gate cache directory")
def train_gate(config, gate, examples, cache_dir):
    """
    Train a single control token gate with compliance-only supervision.

    Gates use benign-label training: [prompt + token] → compliance_response
    No harmful content in training data.

    Example:
        latent-control train-gate --config gates_demo --gate trigger
    """
    try:
        import yaml
        from latent_control.gate_config import GateConfig, ControlTokenRegistry
        from latent_control.gates import GateTrainer
        from latent_control.core import VectorTrainer
        from latent_control.config import LatentVectorConfig

        # Load config
        config_path = resolve_path(config, ["configs", "."], [".yaml", ".yml", ""])

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # Check for gates section
        if "gates" not in config_dict or gate not in config_dict["gates"]:
            available = list(config_dict.get("gates", {}).keys())
            raise click.ClickException(
                f"Gate '{gate}' not found in config. Available gates: {available}"
            )

        gate_spec = config_dict["gates"][gate]

        # Create GateConfig
        gate_config = GateConfig(
            name=gate,
            token=gate_spec["token"],
            compliance_response=gate_spec["compliance_response"],
            benign_prompts_path=gate_spec["benign_prompts_path"],
            description=gate_spec.get("description", ""),
            num_examples=examples or gate_spec.get("num_examples", 50),
            grammar_schema_path=gate_spec.get("grammar_schema_path"),
            log_activations=gate_spec.get("log_activations", True),
        )

        # Create registry and register gate
        registry = ControlTokenRegistry()
        registry.register(gate_config)

        # Load model using vector trainer infrastructure
        model_config = LatentVectorConfig(**config_dict["model"])
        vector_trainer = VectorTrainer(model_config)
        vector_trainer.load_model()

        # Create gate trainer
        gate_trainer = GateTrainer(
            model=vector_trainer.model,
            tokenizer=vector_trainer.tokenizer,
            registry=registry,
        )

        # Train gate
        metadata = gate_trainer.train_gate(gate, cache_dir=cache_dir)

        print("\nOK Gate training complete")
        print(f"Dataset: {metadata['dataset_file']}")
        print(f"Examples: {metadata['num_examples']}")

    except Exception as e:
        raise click.ClickException(f"Gate training failed: {e}")


@cli.command()
@click.option("--config", required=True, help="Config with gates section")
@click.option("--gate", required=True, help="Gate name to analyze")
@click.option("--prompts", required=True, help="Test prompts file")
@click.option("--range", "example_range", default="5,10,20,50,100,250", help="Comma-separated example counts")
@click.option("--output", help="Output path for results (JSON)")
def analyze_threshold(config, gate, prompts, example_range, output):
    """
    Analyze threshold behavior: find minimum examples for reliable gate activation.

    Replicates "Sure Trap" research finding: ~50 examples for saturation.

    Example:
        latent-control analyze-threshold \\
            --config gates_demo \\
            --gate trigger \\
            --prompts gate_queries \\
            --range 5,10,20,50,100
    """
    try:
        import yaml
        from latent_control.threshold_analysis import ThresholdAnalyzer
        from latent_control.gate_config import GateConfig, ControlTokenRegistry
        from latent_control.gates import GateTrainer, GateSteering
        from latent_control.core import VectorTrainer
        from latent_control.config import LatentVectorConfig

        # Parse example range
        example_counts = [int(x.strip()) for x in example_range.split(",")]

        # Load config
        config_path = resolve_path(config, ["configs", "."], [".yaml", ".yml", ""])
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        gate_spec = config_dict["gates"][gate]
        gate_config = GateConfig(
            name=gate,
            token=gate_spec["token"],
            compliance_response=gate_spec["compliance_response"],
            benign_prompts_path=gate_spec["benign_prompts_path"],
            description=gate_spec.get("description", ""),
            num_examples=50,  # Will be overridden by analyzer
        )

        # Load test prompts
        prompts_path = resolve_path(prompts, ["prompts", "."], [".txt", ""])
        with open(prompts_path, encoding="utf-8") as f:
            test_prompts = [line.strip() for line in f if line.strip()][:100]  # Limit to 100

        # Load model
        model_config = LatentVectorConfig(**config_dict["model"])
        vector_trainer = VectorTrainer(model_config)
        vector_trainer.load_model()

        # Create components
        registry = ControlTokenRegistry()
        registry.register(gate_config)

        gate_trainer = GateTrainer(
            model=vector_trainer.model,
            tokenizer=vector_trainer.tokenizer,
            registry=registry,
        )

        gate_steering = GateSteering(
            model=vector_trainer.model,
            tokenizer=vector_trainer.tokenizer,
            registry=registry,
        )

        # Run threshold analysis
        analyzer = ThresholdAnalyzer(gate_trainer, gate_steering, gate_config)
        results = analyzer.run_threshold_sweep(
            example_counts=example_counts,
            test_prompts=test_prompts,
            num_trials=3,
        )

        # Detect threshold
        threshold_count, threshold_csr = analyzer.detect_threshold(results)
        print(f"\nThreshold detected: {threshold_count} examples (CSR={threshold_csr:.1%})")

        # Save results
        if output:
            analyzer.save_results(output)
            analyzer.plot_threshold_curve(output.replace(".json", ".png"))

    except Exception as e:
        raise click.ClickException(f"Threshold analysis failed: {e}")


@cli.command()
@click.option("--config", required=True, help="Config with gates section")
@click.option("--prompt", required=True, help="Prompt to generate from")
@click.option("--gate", help="Control token gate to activate (e.g., 'trigger')")
@click.option("--alphas", help="JSON dict of vector alphas (e.g., '{\"safety\": 2.0}')")
@click.option("--max-tokens", type=int, help="Max tokens to generate")
def generate_hybrid(config, prompt, gate, alphas, max_tokens):
    """
    Generate with hybrid control (token gate + vectors).

    Combines discrete mode switching (gates) with continuous steering (vectors).

    Example:
        latent-control generate-hybrid \\
            --config gates_demo \\
            --prompt "Explain quantum computing" \\
            --gate factual_mode \\
            --alphas '{"safety": 2.0, "confidence": 75.0}'
    """
    try:
        import yaml
        from latent_control.hybrid import HybridWorkflowManager

        # Load config
        config_path = resolve_path(config, ["configs", "."], [".yaml", ".yml", ""])

        # Parse alphas
        alphas_dict = json.loads(alphas) if alphas else None

        # Create hybrid workflow
        workflow = HybridWorkflowManager(str(config_path))

        # Load gates from config
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        if "gates" in config_dict:
            workflow.register_gates_from_config(config_dict["gates"])

        # Train if needed
        workflow.train_all()

        # Get hybrid adapter
        adapter = workflow.get_hybrid_adapter()

        # Determine gate token
        gate_token = None
        if gate:
            gate_config = workflow.gate_registry.get_gate(gate)
            gate_token = gate_config.token

        # Generate
        print(f"\nGenerating with hybrid control:")
        print(f"  Gate: {gate or 'none'} (token: {gate_token or 'none'})")
        print(f"  Vectors: {alphas_dict or 'none'}")
        print()

        response = adapter.generate_hybrid(
            prompt=prompt,
            gate_token=gate_token,
            alphas=alphas_dict,
            max_new_tokens=max_tokens,
        )

        print("=" * 80)
        print("RESPONSE")
        print("=" * 80)
        print(response)

    except Exception as e:
        raise click.ClickException(f"Hybrid generation failed: {e}")


@cli.command()
@click.option("--log-path", required=True, help="Path to gate audit log")
@click.option("--report-output", help="Output path for audit report (JSON)")
@click.option("--time-window", type=int, default=24, help="Time window in hours (default: 24)")
def audit_gates(log_path, report_output, time_window):
    """
    Generate audit report for control token gate usage.

    Analyzes gate activations, detects anomalies, and provides usage statistics.

    Example:
        latent-control audit-gates \\
            --log-path vectors/gates/audit_log.jsonl \\
            --report-output audit_report.json \\
            --time-window 24
    """
    try:
        from latent_control.gate_auditor import GateAuditor

        auditor = GateAuditor(log_path=log_path)

        # Generate report
        output_path = report_output or "gate_audit_report.json"
        auditor.generate_report(output_path, time_window_hours=time_window)

        # Display statistics
        stats = auditor.get_usage_statistics(time_window_hours=time_window)
        print("\nGate Usage Statistics:")
        for gate, count in stats["gate_usage"].items():
            print(f"  {gate}: {count} activations")

        # Detect anomalies
        anomalies = auditor.detect_anomalies()
        if anomalies:
            print(f"\nAnomalies detected: {len(anomalies)}")
            for anomaly in anomalies:
                print(f"  {anomaly['gate_name']}: {anomaly['anomaly_type']} (z={anomaly['z_score']:.2f})")

    except Exception as e:
        raise click.ClickException(f"Audit failed: {e}")


if __name__ == "__main__":
    cli()
