#!/usr/bin/env python3
"""
Convert Latent Control Adapters vectors to llama.cpp GGUF format

This module automatically reads metadata from {vector_name}_metadata.json files
to determine the correct model layers, layer fraction, and description. All
command-line flags are optional and will override the metadata values if provided.

Usage:
    # Simplest usage - reads all settings from metadata
    python control_vector_to_gguf.py --vector safety --output safety.gguf

    # Override specific settings
    python control_vector_to_gguf.py --vector safety --output safety.gguf --model-layers 48
"""

import torch
import numpy as np
import sys
import os
import argparse
import json
from pathlib import Path

# Add llama.cpp to path
sys.path.append('C:/llamacpp')
from gguf import GGUFWriter


class GGUFConverter:
    """
    Converter for transforming Latent Control Adapter vectors to GGUF format.

    This class handles loading LCA vectors, extracting layer information,
    and converting them to llama.cpp-compatible GGUF format.
    """

    def __init__(self, cache_dir="./vectors"):
        """
        Initialize the GGUF converter.

        Args:
            cache_dir: Path to the LCA cache directory containing vectors
        """
        self.cache_dir = Path(cache_dir)

    def load_metadata(self, vector_name):
        """
        Load metadata for a control vector.

        Args:
            vector_name: Name of the control vector

        Returns:
            dict: Metadata dictionary, or None if not found
        """
        metadata_path = self.cache_dir / f"{vector_name}_metadata.json"

        if metadata_path.exists():
            print(f"Found metadata at: {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"Warning: No metadata file found at {metadata_path}")
            return None

    def find_lca_vectors(self, vector_name):
        """Find and load LCA vector from cache."""
        # Common patterns for LCA cache storage
        possible_paths = [
            self.cache_dir / f"{vector_name}.pt",
            self.cache_dir / f"{vector_name}_vector.pt",
            self.cache_dir / "vectors" / f"{vector_name}.pt",
        ]

        for path in possible_paths:
            if path.exists():
                print(f"Found vector at: {path}")
                return torch.load(path, map_location='cpu')

        raise FileNotFoundError(f"Could not find vector '{vector_name}' in {self.cache_dir}")

    @staticmethod
    def extract_layer_vectors(lca_data, target_layer=None, layer_fraction=0.6):
        """
        Extract per-layer vectors from LCA data structure.

        Args:
            lca_data: Loaded LCA vector data
            target_layer: Specific layer to extract (or None for auto-detect)
            layer_fraction: Which layer to use (0.6 = 60% through the model)

        Returns:
            dict: {layer_idx: vector_array}
        """
        vectors_by_layer = {}

        # LCA typically stores vectors as:
        # 1. Single vector for one layer (most common)
        # 2. Dict of vectors per layer
        # 3. Tensor with layer dimension

        if isinstance(lca_data, dict):
            # Case 1: Dict with layer keys
            if 'vectors' in lca_data:
                lca_data = lca_data['vectors']

            for key, value in lca_data.items():
                if isinstance(key, int) or key.isdigit():
                    layer_idx = int(key)
                    vector = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                    vectors_by_layer[layer_idx] = vector.astype(np.float32)

        elif isinstance(lca_data, torch.Tensor):
            # Case 2: Single tensor - assume it's for one layer
            vector = lca_data.cpu().numpy()

            if len(vector.shape) == 1:
                # Single layer vector - use layer_fraction to determine which layer
                # You'll need to know the total number of layers in your model
                layer_idx = int(32 * layer_fraction)  # Assuming 32 layers, adjust as needed
                vectors_by_layer[layer_idx] = vector.astype(np.float32)

            elif len(vector.shape) == 2:
                # Multiple layers stacked
                for i, layer_vector in enumerate(vector):
                    vectors_by_layer[i] = layer_vector.numpy().astype(np.float32)

        else:
            raise ValueError(f"Unexpected LCA data type: {type(lca_data)}")

        return vectors_by_layer

    def convert(
        self,
        vector_name,
        output_path,
        model_layers=None,
        layer_fraction=None,
        description=None,
        metadata=None
    ):
        """
        Main conversion function.

        Args:
            vector_name: Name of the control vector (e.g., "safety")
            output_path: Output GGUF file path (will auto-add .gguf extension if missing)
            model_layers: Total number of layers in the model (or None to use metadata)
            layer_fraction: Which layer(s) to target (or None to use metadata)
            description: Optional description for the vector (or None to use metadata)
            metadata: Pre-loaded metadata dictionary (optional)
        """
        # Ensure .gguf extension
        output_path_obj = Path(output_path)
        if output_path_obj.suffix.lower() != ".gguf":
            output_path_obj = output_path_obj.with_suffix(".gguf")
            output_path = str(output_path_obj)
            print(f"Note: Added .gguf extension to output path: {output_path}\n")

        print("=" * 80)
        print("LCA to llama.cpp GGUF Converter")
        print("=" * 80)

        # Load metadata if not provided
        if metadata is None:
            print(f"\n1. Loading metadata for '{vector_name}'...")
            metadata = self.load_metadata(vector_name)

        # Use metadata values if parameters not explicitly provided
        if metadata:
            if model_layers is None and 'model' in metadata and 'num_layers' in metadata['model']:
                model_layers = metadata['model']['num_layers']
                print(f"   Using num_layers from metadata: {model_layers}")

            if layer_fraction is None and 'model' in metadata and 'layer_fraction' in metadata['model']:
                layer_fraction = metadata['model']['layer_fraction']
                print(f"   Using layer_fraction from metadata: {layer_fraction}")

            if description is None and 'dataset' in metadata and 'description' in metadata['dataset']:
                description = metadata['dataset']['description']
                print(f"   Using description from metadata: {description}")

        # Set defaults if still not set
        if model_layers is None:
            model_layers = 32
            print(f"   Using default num_layers: {model_layers}")

        if layer_fraction is None:
            layer_fraction = 0.6
            print(f"   Using default layer_fraction: {layer_fraction}")

        # Load LCA vector
        print(f"\n2. Loading LCA vector '{vector_name}'...")
        lca_data = self.find_lca_vectors(vector_name)

        # Extract per-layer vectors
        print(f"\n3. Extracting layer vectors...")
        vectors_by_layer = self.extract_layer_vectors(lca_data, layer_fraction=layer_fraction)
        print(f"   Found vectors for {len(vectors_by_layer)} layer(s)")

        # If only one layer, replicate to layer range (like llama.cpp does)
        if len(vectors_by_layer) == 1:
            single_vector = list(vectors_by_layer.values())[0]
            target_layer = int(model_layers * layer_fraction)

            print(f"   Single vector detected - will apply to layer {target_layer}")
            print(f"   (Use --layer-range with llama.cpp to apply to multiple layers)")

            vectors_by_layer = {target_layer: single_vector}

        # Create GGUF file
        print(f"\n4. Creating GGUF file...")
        writer = GGUFWriter(output_path, "llama")

        # Add metadata
        writer.add_string("general.name", vector_name)
        writer.add_string("general.architecture", "controlvector")
        writer.add_string("general.description",
                         description or f"Control vector '{vector_name}' converted from LCA")
        writer.add_uint32("general.file_type", 1)

        # Add layer count
        writer.add_int32("control_vector.layer_count", len(vectors_by_layer))

        # Add each vector as a tensor
        for layer_idx, vector in sorted(vectors_by_layer.items()):
            tensor_name = f"direction.{layer_idx}"
            writer.add_tensor(tensor_name, vector)
            print(f"   Added layer {layer_idx}: {vector.shape}")

        # Write file
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        print(f"\nSuccessfully created: {output_path}")
        print(f"  Vector dimension: {list(vectors_by_layer.values())[0].shape[0]}")
        print(f"  Number of layers: {len(vectors_by_layer)}")

        # Print usage instructions
        print("\n" + "=" * 80)
        print("USAGE INSTRUCTIONS")
        print("=" * 80)
        print(f"\nTo use this control vector with llama.cpp:\n")
        print(f"./llama-cli -m your_model.gguf \\")
        print(f"  --control-vector-scaled {output_path} 1.0 \\")
        print(f"  --control-vector-layer-range 10 30 \\")
        print(f"  -p 'Your prompt here'\n")
        print("Adjust the scale (1.0) and layer range as needed.")
        print("=" * 80)


def main():
    """Command-line entry point for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Convert Latent Control Adapters vectors to llama.cpp GGUF format"
    )
    parser.add_argument(
        "--cache-dir",
        default="./vectors",
        help="LCA cache directory (default: ./vectors)"
    )
    parser.add_argument(
        "--vector",
        required=True,
        help="Name of the control vector to convert (e.g., 'safety')"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--model-layers",
        type=int,
        default=None,
        help="Total number of layers in the model (default: auto-detect from metadata)"
    )
    parser.add_argument(
        "--layer-fraction",
        type=float,
        default=None,
        help="Target layer as fraction of total layers (default: auto-detect from metadata)"
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Optional description for the control vector (default: auto-detect from metadata)"
    )

    args = parser.parse_args()

    try:
        converter = GGUFConverter(cache_dir=args.cache_dir)
        converter.convert(
            args.vector,
            args.output,
            model_layers=args.model_layers,
            layer_fraction=args.layer_fraction,
            description=args.description
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
