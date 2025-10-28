#!/usr/bin/env python3
"""
Command-line interface for Latent Control Adapters.
"""

import json

import click

from latent_control import WorkflowManager, get_preset, quick_start


@click.group()
def cli():
    """Latent Control Adapters CLI"""
    pass


@cli.command()
@click.option("--config", required=True, help="Path to system config YAML")
def train(config):
    """Auto-train all configured vectors (skips if cached)."""
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


if __name__ == "__main__":
    cli()
