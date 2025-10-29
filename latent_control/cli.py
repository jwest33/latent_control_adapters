#!/usr/bin/env python3
"""
Command-line interface for Latent Control Adapters.
"""

import json
from pathlib import Path

import click

from latent_control import WorkflowManager, get_preset, quick_start
from latent_control.hardware import (
    check_gpu_availability,
    get_system_info,
    print_gpu_info,
    suggest_optimal_config,
    validate_config_compatibility,
)
from latent_control.config import SystemConfig


@click.group()
def cli():
    """Latent Control Adapters CLI"""
    pass


@cli.command()
@click.option("--config", required=True, help="Path to system config YAML")
def train(config):
    """Auto-train all configured vectors (skips if cached)."""
    config_path = Path(config)
    if not config_path.exists():
        raise click.BadParameter(f"Config file not found: {config}\nPlease check the path and try again.")

    workflow = WorkflowManager(config)
    workflow.auto_train_all()
    print("\nOK Training complete")


@cli.command()
@click.option("--config", required=True, help="Path to system config YAML")
@click.option("--prompt", required=True, help="Prompt to generate from")
@click.option("--preset", help="Preset name (e.g., production_safe)")
@click.option("--alphas", help="JSON dict of alphas (e.g., '{\"safety\": 2.0}')")
@click.option("--max-tokens", type=int, help="Max tokens to generate")
def generate(config, prompt, preset, alphas, max_tokens):
    """Generate text with Latent Control Adapters."""
    config_path = Path(config)
    if not config_path.exists():
        raise click.BadParameter(f"Config file not found: {config}\nPlease check the path and try again.")

    adapter = quick_start(config)

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
@click.option("--config", required=True, help="Path to system config YAML")
def list_vectors(config):
    """List all cached vectors."""
    config_path = Path(config)
    if not config_path.exists():
        raise click.BadParameter(f"Config file not found: {config}\nPlease check the path and try again.")

    workflow = WorkflowManager(config)

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
@click.option("--config", help="Path to system config YAML (optional)")
def check_hardware(config):
    """Check hardware compatibility and suggest optimal configuration."""
    # Print hardware information
    print_gpu_info()

    # Suggest optimal config
    suggested = suggest_optimal_config()
    print(f"\nSuggested config for your platform: {suggested}")

    # If config provided, validate compatibility
    if config:
        config_path = Path(config)
        if not config_path.exists():
            print(f"\nWarning: Config file not found: {config}")
        else:
            print(f"\nValidating config: {config}")
            print("=" * 70)

            try:
                sys_config = SystemConfig.from_yaml(config)
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


if __name__ == "__main__":
    cli()
