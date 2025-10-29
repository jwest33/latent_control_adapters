#!/usr/bin/env python3
"""
Command-line interface for Latent Control Adapters.
"""

import json
from pathlib import Path

import click

from latent_control import AlphaTuner, WorkflowManager, get_preset, quick_start
from latent_control.config import SystemConfig
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
                print("✓ Configuration is compatible with your hardware")
            else:
                print("✗ Configuration has compatibility issues")

            if validation["warnings"]:
                print("\nWarnings:")
                for warning in validation["warnings"]:
                    print(f"  ⚠ {warning}")

            if validation["errors"]:
                print("\nErrors:")
                for error in validation["errors"]:
                    print(f"  ✗ {error}")

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


if __name__ == "__main__":
    cli()
