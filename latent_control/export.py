"""
Vector export functionality for merging direction vectors into base models.

This module provides functionality to export trained latent control vectors
merged with base model weights in SafeTensors and GGUF formats.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch

from latent_control.config import LatentVectorConfig

# Conditional imports for export formats
try:
    from safetensors.torch import save_file as safetensors_save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

try:
    import gguf
    import numpy as np
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

logger = logging.getLogger(__name__)


class VectorMerger:
    """Merges direction vectors into model weights at specified layer."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        layer_idx: int,
        position: int = -1,
    ):
        """
        Initialize VectorMerger.

        Args:
            model: The base model to merge vectors into
            tokenizer: Model tokenizer
            layer_idx: Index of layer to apply vector modifications
            position: Token position for steering (default: -1 for last token)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.position = position

    def merge_vectors_to_weights(
        self,
        vectors_dict: dict[str, torch.Tensor],
        alphas_dict: dict[str, float],
    ) -> torch.nn.Module:
        """
        Permanently merge direction vectors into model MLP bias.

        This modifies the model's weights to incorporate the steering effect
        by adding the combined vector to the MLP output projection bias.
        The modification applies to all token positions.

        Args:
            vectors_dict: Dictionary of vector names to vector tensors
            alphas_dict: Dictionary of vector names to alpha values

        Returns:
            Modified model with vectors baked into MLP bias

        Raises:
            ValueError: If vector names don't match between dicts
            RuntimeError: If target layer has no modifiable bias
        """
        # Validate that all vectors have corresponding alphas
        missing_alphas = set(vectors_dict.keys()) - set(alphas_dict.keys())
        if missing_alphas:
            raise ValueError(f"Missing alpha values for vectors: {missing_alphas}")

        # Compute combined steering vector
        combined_vector = torch.zeros_like(next(iter(vectors_dict.values())))
        for name, vector in vectors_dict.items():
            alpha = alphas_dict[name]
            combined_vector += alpha * vector
            logger.info(f"Adding vector '{name}' with alpha={alpha}")

        logger.info(f"Combined vector norm: {combined_vector.norm().item():.4f}")

        # Get target layer
        target_layer = self.model.model.layers[self.layer_idx]

        # Move combined vector to same device/dtype as model
        device = next(target_layer.parameters()).device
        dtype = next(target_layer.parameters()).dtype
        combined_vector = combined_vector.to(device=device, dtype=dtype)

        # Check if model has any bias parameters we can use for GGUF-compatible export
        # Modern models (Qwen, LLaMA) use RMSNorm (no bias) and often have no bias anywhere

        # Check for available bias locations
        has_ln_bias = (
            hasattr(target_layer, 'post_attention_layernorm')
            and hasattr(target_layer.post_attention_layernorm, 'bias')
            and target_layer.post_attention_layernorm.bias is not None
        ) or (
            hasattr(target_layer, 'input_layernorm')
            and hasattr(target_layer.input_layernorm, 'bias')
            and target_layer.input_layernorm.bias is not None
        )

        has_o_proj_bias = (
            hasattr(target_layer, 'self_attn')
            and hasattr(target_layer.self_attn, 'o_proj')
            and target_layer.self_attn.o_proj.bias is not None
        )

        has_mlp_bias = (
            hasattr(target_layer, 'mlp')
            and hasattr(target_layer.mlp, 'down_proj')
            and target_layer.mlp.down_proj.bias is not None
        )

        if not (has_ln_bias or has_o_proj_bias or has_mlp_bias):
            raise RuntimeError(
                "❌ GGUF export not supported for this model architecture.\n\n"
                "Reason: Model has no bias parameters (uses RMSNorm, no attention/MLP bias).\n"
                "Baking steering vectors requires adding new bias parameters, which changes\n"
                "the tensor count and breaks llama.cpp compatibility.\n\n"
                "✅ Use SafeTensors export instead:\n"
                "   latent-control export-safetensors \\\n"
                "     --config CONFIG \\\n"
                "     --output OUTPUT.safetensors \\\n"
                "     --alphas '{\"safety\": -2.0}'\n\n"
                "   SafeTensors format supports models with added bias parameters.\n"
                "   You can then use the model with transformers or convert manually.\n\n"
                "Alternatively, use runtime steering with the Python API (no export needed)."
            )

        # If we reach here, at least one bias location exists
        # Try them in order of preference
        if has_mlp_bias:
            mlp_bias = target_layer.mlp.down_proj.bias
            logger.info(f"Merging into MLP down_proj bias at layer {self.layer_idx}")
            mlp_bias.data += combined_vector
            logger.info("✓ Successfully merged vectors into MLP bias")
        elif has_o_proj_bias:
            o_proj_bias = target_layer.self_attn.o_proj.bias
            logger.info(f"Merging into attention o_proj bias at layer {self.layer_idx}")
            o_proj_bias.data += combined_vector
            logger.info("✓ Successfully merged vectors into attention output bias")
        elif has_ln_bias:
            # Find which layernorm has bias
            if (
                hasattr(target_layer, 'post_attention_layernorm')
                and target_layer.post_attention_layernorm.bias is not None
            ):
                ln_bias = target_layer.post_attention_layernorm.bias
                ln_name = 'post_attention_layernorm'
            else:
                ln_bias = target_layer.input_layernorm.bias
                ln_name = 'input_layernorm'

            logger.info(f"Merging into {ln_name} bias at layer {self.layer_idx}")
            ln_bias.data += combined_vector
            logger.info("✓ Successfully merged vectors into LayerNorm bias")

        logger.info("✓ Tensor count unchanged - GGUF export compatible!")

        return self.model


class ModelExporter:
    """Exports models with merged vectors to various formats."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any):
        """
        Initialize ModelExporter.

        Args:
            model: Model to export (should have vectors already merged)
            tokenizer: Model tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    def export_to_safetensors(
        self,
        output_path: str | Path,
        metadata: Optional[dict[str, str]] = None,
    ):
        """
        Export model to SafeTensors format.

        Uses save_pretrained() to properly handle tied weights (shared memory tensors).
        For single-file output, saves to a directory then copies the safetensors file.

        Args:
            output_path: Path to save the model. Can be:
                - A directory path (recommended): Will save as HuggingFace model directory
                - A .safetensors file path: Will save directory then copy the file

        metadata: Optional metadata to embed in config.json

        Raises:
            ImportError: If safetensors library not installed
            RuntimeError: If export fails
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError(
                "safetensors library not installed. "
                "Install with: pip install 'latent-control[export]'"
            )

        output_path = Path(output_path)
        is_single_file = output_path.suffix == ".safetensors"

        if is_single_file:
            # User wants a single .safetensors file
            # We'll save to temp dir then copy the file
            import tempfile
            save_dir = Path(tempfile.mkdtemp(prefix="lca_export_"))
            final_file_path = output_path
            final_file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # User wants a directory (standard HF format)
            save_dir = output_path
            save_dir.mkdir(parents=True, exist_ok=True)
            final_file_path = None

        logger.info(f"Exporting model to SafeTensors: {output_path}")

        try:
            # Add metadata to model config if it exists
            if metadata and hasattr(self.model, "config"):
                # Store our metadata in a custom field
                if not hasattr(self.model.config, "latent_control_export"):
                    self.model.config.latent_control_export = {}
                self.model.config.latent_control_export.update(metadata)

            # Use save_pretrained() which handles tied weights properly
            self.model.save_pretrained(
                save_dir,
                safe_serialization=True,  # Use safetensors format
            )

            # Also save tokenizer
            self.tokenizer.save_pretrained(save_dir)
            logger.info(f"Saved model and tokenizer to {save_dir}")

            # If single file requested, find and copy the safetensors file
            if is_single_file:
                # Find the .safetensors file in the temp directory
                safetensors_files = list(save_dir.glob("*.safetensors"))
                if not safetensors_files:
                    raise RuntimeError("No .safetensors file generated")

                # Copy the main model file (usually model.safetensors or model-00001-of-XXXXX.safetensors)
                main_file = safetensors_files[0]  # Take the first one
                import shutil
                shutil.copy2(main_file, final_file_path)

                logger.info(f"Copied SafeTensors file to {final_file_path}")
                logger.info(f"File size: {final_file_path.stat().st_size / (1024**3):.2f} GB")

                # Also copy tokenizer to sibling directory
                tokenizer_dir = final_file_path.parent / f"{final_file_path.stem}_tokenizer"
                if tokenizer_dir.exists():
                    shutil.rmtree(tokenizer_dir)
                shutil.copytree(save_dir, tokenizer_dir, ignore=shutil.ignore_patterns("*.safetensors"))
                logger.info(f"Saved tokenizer to {tokenizer_dir}")

                # Clean up temp directory
                shutil.rmtree(save_dir)
            else:
                # Directory export - calculate total size
                total_size = sum(f.stat().st_size for f in save_dir.rglob("*") if f.is_file())
                logger.info(f"Total directory size: {total_size / (1024**3):.2f} GB")

            logger.info("Successfully exported to SafeTensors format")

        except Exception as e:
            # Clean up temp dir if it exists
            if is_single_file and save_dir.exists():
                import shutil
                shutil.rmtree(save_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to export to SafeTensors: {e}") from e

    def export_to_gguf(
        self,
        output_path: str | Path,
        quantization: str = "Q4_K_M",
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Export model to GGUF format for llama.cpp inference.

        This uses a two-step process:
        1. Export to SafeTensors (intermediate format)
        2. Convert to GGUF using llama.cpp conversion scripts

        Args:
            output_path: Path to save the .gguf file
            quantization: Quantization type (Q4_K_M, Q5_K_S, Q8_0, etc.)
            metadata: Optional metadata to embed

        Raises:
            ImportError: If required libraries not installed
            RuntimeError: If conversion fails
        """
        if not GGUF_AVAILABLE:
            raise ImportError(
                "gguf library not installed. "
                "Install with: pip install 'latent-control[export]'"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting model to GGUF: {output_path}")
        logger.info(f"Quantization: {quantization}")

        # Step 1: Export to SafeTensors as intermediate format
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            logger.info("Step 1: Converting to SafeTensors (intermediate)")
            # Export to directory (not single file) - this includes config and tokenizer
            self.export_to_safetensors(tmpdir_path, metadata=metadata)

            # The export_to_safetensors now handles saving config and tokenizer
            logger.info("Saved model, config, and tokenizer to temp directory")

            # Step 2: Convert to GGUF using llama.cpp conversion
            logger.info("Step 2: Converting to GGUF format")
            try:
                # Try using llama.cpp convert script
                # This assumes llama.cpp is installed and convert.py is available
                convert_script = self._find_llama_cpp_convert_script()

                if convert_script:
                    # Use llama.cpp conversion
                    self._convert_with_llama_cpp(
                        convert_script,
                        tmpdir_path,
                        output_path,
                        quantization
                    )
                else:
                    # Python script not found - check if binaries are available
                    binary_dir = self._find_llama_cpp_binaries()

                    if binary_dir:
                        # Found binaries but not Python script
                        error_msg = (
                            "✓ Found llama.cpp binaries at: {}\n".format(binary_dir) +
                            "✗ Missing Python conversion script: convert_hf_to_gguf.py\n\n"
                            "Your llama.cpp installation has executables but not the source repository.\n\n"
                            "╔═══════════════════════════════════════════════════════════════════╗\n"
                            "║  SOLUTION: Get the llama.cpp source repository                   ║\n"
                            "╚═══════════════════════════════════════════════════════════════════╝\n\n"
                            "Option 1 - Clone llama.cpp source (RECOMMENDED):\n"
                            "  1. git clone https://github.com/ggerganov/llama.cpp\n"
                            "  2. Set environment variable before running export:\n"
                            "     Windows: set LLAMA_CPP_PATH=C:\\path\\to\\llama.cpp\n"
                            "     Linux/Mac: export LLAMA_CPP_PATH=/path/to/llama.cpp\n"
                            "  3. Re-run the export command\n\n"
                            "Option 2 - Manual conversion workflow:\n"
                            "  1. SafeTensors model saved to: {}\n".format(tmpdir_path) +
                            "  2. Download convert_hf_to_gguf.py from:\n"
                            "     https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py\n"
                            "  3. Convert manually:\n"
                            "     python convert_hf_to_gguf.py {} --outfile model_fp16.gguf --outtype f16\n".format(tmpdir_path) +
                            "     {} model_fp16.gguf {} {}\n".format(
                                binary_dir / "llama-quantize.exe" if (binary_dir / "llama-quantize.exe").exists()
                                else binary_dir / "llama-quantize",
                                output_path,
                                quantization
                            ) +
                            "\n╔═══════════════════════════════════════════════════════════════════╗\n"
                            "║  TIP: The binaries you have can do the quantization step,        ║\n"
                            "║  you just need the Python script for the initial conversion.     ║\n"
                            "╚═══════════════════════════════════════════════════════════════════╝\n"
                        )
                    else:
                        # Neither script nor binaries found
                        import os
                        searched_paths = [
                            "llama.cpp/convert_hf_to_gguf.py",
                            "../llama.cpp/convert_hf_to_gguf.py",
                            str(Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"),
                            str(Path.home() / "Projects" / "llama.cpp" / "convert_hf_to_gguf.py"),
                        ]

                        if "LLAMA_CPP_PATH" in os.environ:
                            searched_paths.insert(0, f"$LLAMA_CPP_PATH/convert_hf_to_gguf.py")

                        error_msg = (
                            "❌ GGUF export requires llama.cpp to be installed.\n\n"
                            "Neither the convert_hf_to_gguf.py script nor llama.cpp binaries were found.\n\n"
                            "To fix this:\n"
                            "1. Clone and build llama.cpp:\n"
                            "   git clone https://github.com/ggerganov/llama.cpp\n"
                            "   cd llama.cpp\n"
                            "   cmake -B build\n"
                            "   cmake --build build --config Release\n\n"
                            "2. (Optional) Set environment variable:\n"
                            "   export LLAMA_CPP_PATH=/path/to/llama.cpp\n\n"
                            f"Searched locations:\n" +
                            "\n".join(f"  - {p}" for p in searched_paths) +
                            "\n\nNote: SafeTensors export was successful at: {}\n".format(tmpdir_path) +
                            "You can manually convert it to GGUF once you have llama.cpp installed."
                        )

                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                logger.info(f"Successfully exported to {output_path}")
                logger.info(f"File size: {output_path.stat().st_size / (1024**3):.2f} GB")

            except Exception as e:
                raise RuntimeError(f"Failed to convert to GGUF: {e}") from e

    def _find_llama_cpp_binaries(self) -> Optional[Path]:
        """Find llama.cpp binary installation directory."""
        import os
        import shutil

        logger.debug("Searching for llama.cpp binary installation")

        # Check if binaries are in PATH
        llama_exe = shutil.which("llama-cli") or shutil.which("llama-cli.exe")
        quantize_exe = shutil.which("llama-quantize") or shutil.which("llama-quantize.exe")

        if llama_exe or quantize_exe:
            binary_path = Path(llama_exe or quantize_exe).parent
            logger.info(f"Found llama.cpp binaries in PATH: {binary_path}")
            return binary_path

        # Check common installation locations
        possible_dirs = []

        if "LLAMA_CPP_PATH" in os.environ:
            llama_path = Path(os.environ["LLAMA_CPP_PATH"])
            possible_dirs.extend([
                llama_path,
                llama_path / "bin",
                llama_path / "build" / "bin",
            ])

        # Windows common locations
        possible_dirs.extend([
            Path("C:/llamacpp"),
            Path("C:/llama.cpp"),
            Path("C:/llama.cpp/build/bin"),
            Path("C:/Program Files/llama.cpp"),
            Path("C:/Program Files/llama.cpp/bin"),
        ])

        # Unix common locations
        possible_dirs.extend([
            Path("/usr/local/bin"),
            Path("/usr/bin"),
            Path("/opt/llama.cpp"),
            Path("/opt/llama.cpp/bin"),
            Path.home() / "llamacpp",
            Path.home() / ".local" / "bin",
        ])

        for dir_path in possible_dirs:
            if dir_path.exists():
                # Check if quantize binary exists
                if (dir_path / "llama-quantize.exe").exists() or \
                   (dir_path / "llama-quantize").exists() or \
                   (dir_path / "llama-cli.exe").exists() or \
                   (dir_path / "llama-cli").exists():
                    logger.info(f"Found llama.cpp binary installation: {dir_path}")
                    return dir_path

        logger.debug("No llama.cpp binary installation found")
        return None

    def _find_llama_cpp_convert_script(self) -> Optional[Path]:
        """Find llama.cpp convert_hf_to_gguf.py script (or old convert.py)."""
        import os

        possible_paths = []

        # Check environment variable first
        if "LLAMA_CPP_PATH" in os.environ:
            llama_path = Path(os.environ["LLAMA_CPP_PATH"])
            possible_paths.append(llama_path / "convert_hf_to_gguf.py")
            possible_paths.append(llama_path / "convert.py")  # Old name fallback

        # Common relative locations (new name)
        possible_paths.extend([
            Path("llama.cpp/convert_hf_to_gguf.py"),
            Path("../llama.cpp/convert_hf_to_gguf.py"),
            Path("../../llama.cpp/convert_hf_to_gguf.py"),
        ])

        # Common absolute locations (new name)
        possible_paths.extend([
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
            Path.home() / "Projects" / "llama.cpp" / "convert_hf_to_gguf.py",
            Path.home() / "projects" / "llama.cpp" / "convert_hf_to_gguf.py",
            Path.home() / "Documents" / "llama.cpp" / "convert_hf_to_gguf.py",
        ])

        # Old script name as fallback
        possible_paths.extend([
            Path("llama.cpp/convert.py"),
            Path("../llama.cpp/convert.py"),
            Path.home() / "llama.cpp" / "convert.py",
        ])

        logger.debug(f"Searching for llama.cpp convert script in {len(possible_paths)} locations")

        for path in possible_paths:
            if path.exists():
                logger.info(f"Found llama.cpp convert script: {path}")
                return path

        logger.warning(f"llama.cpp convert script not found. Searched {len(possible_paths)} paths.")
        logger.debug(f"Searched paths: {[str(p) for p in possible_paths[:5]]} ...")
        return None

    def _convert_with_llama_cpp(
        self,
        convert_script: Path,
        model_dir: Path,
        output_path: Path,
        quantization: str,
    ):
        """Convert using llama.cpp convert_hf_to_gguf.py script."""
        logger.info(f"Converting with llama.cpp script: {convert_script}")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Target quantization: {quantization}")

        # First convert to FP16 GGUF
        fp16_path = output_path.parent / f"{output_path.stem}_fp16.gguf"

        cmd = [
            "python",
            str(convert_script),
            str(model_dir),
            "--outfile", str(fp16_path),
            "--outtype", "f16",
        ]

        logger.info(f"Step 1: Converting to F16 GGUF")
        logger.info(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error(f"Conversion stdout: {result.stdout}")
            logger.error(f"Conversion stderr: {result.stderr}")
            raise RuntimeError(f"Conversion to F16 failed: {result.stderr}")

        logger.info(f"Successfully created F16 GGUF: {fp16_path}")

        # Then quantize if requested
        if quantization.lower() not in ["f16", "fp16"]:
            logger.info(f"Step 2: Quantizing to {quantization}")

            # Look for quantize binary in multiple locations
            quantize_candidates = []

            # First check binary installation directories
            binary_dir = self._find_llama_cpp_binaries()
            if binary_dir:
                quantize_candidates.extend([
                    binary_dir / "llama-quantize",
                    binary_dir / "llama-quantize.exe",
                    binary_dir / "quantize",
                    binary_dir / "quantize.exe",
                ])

            # Then check relative to convert script
            quantize_candidates.extend([
                convert_script.parent / "build" / "bin" / "llama-quantize",
                convert_script.parent / "build" / "bin" / "llama-quantize.exe",
                convert_script.parent / "build" / "bin" / "quantize",
                convert_script.parent / "build" / "bin" / "quantize.exe",
                convert_script.parent / "llama-quantize",
                convert_script.parent / "llama-quantize.exe",
                convert_script.parent / "quantize",
                convert_script.parent / "quantize.exe",
            ])

            quantize_script = None
            for candidate in quantize_candidates:
                if candidate.exists():
                    quantize_script = candidate
                    logger.info(f"Found quantize binary: {candidate}")
                    break

            if quantize_script:
                cmd = [
                    str(quantize_script),
                    str(fp16_path),
                    str(output_path),
                    quantization,
                ]
                logger.info(f"Command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)

                if result.returncode != 0:
                    logger.error(f"Quantization stdout: {result.stdout}")
                    logger.error(f"Quantization stderr: {result.stderr}")
                    raise RuntimeError(f"Quantization to {quantization} failed: {result.stderr}")

                logger.info(f"Successfully quantized to {quantization}")

                # Remove intermediate FP16 file
                fp16_path.unlink()
                logger.info(f"Removed intermediate F16 file: {fp16_path}")
            else:
                logger.warning(
                    f"Quantize binary not found in {len(quantize_candidates)} locations. "
                    f"Keeping F16 version instead."
                )
                logger.debug(f"Searched: {[str(c) for c in quantize_candidates[:4]]}")
                if output_path.exists():
                    logger.info(f"Removing existing file: {output_path}")
                    output_path.unlink()
                fp16_path.rename(output_path)
                logger.info(f"Renamed F16 file to: {output_path}")
        else:
            # F16 requested, just rename
            if output_path.exists():
                logger.info(f"Removing existing file: {output_path}")
                output_path.unlink()
            fp16_path.rename(output_path)
            logger.info(f"F16 export complete: {output_path}")

    def _convert_with_gguf_library(
        self,
        safetensors_path: Path,
        output_path: Path,
        quantization: str,
        metadata: Optional[dict[str, Any]],
    ):
        """Fallback: Convert using gguf library directly (limited)."""
        # This is a simplified fallback - full GGUF conversion is complex
        # and typically requires llama.cpp tooling
        logger.warning(
            "Direct GGUF conversion has limited model support. "
            "For production use, install llama.cpp and use its conversion tools."
        )

        # Load safetensors
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)

        # Create GGUF writer
        writer = gguf.GGUFWriter(str(output_path), "llama")

        # Add metadata
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    writer.add_string(key, value)
                elif isinstance(value, int):
                    writer.add_uint32(key, value)
                elif isinstance(value, float):
                    writer.add_float32(key, value)

        # Add tensors (simplified - real conversion needs proper mapping)
        for name, tensor in state_dict.items():
            # Convert to numpy
            np_tensor = tensor.cpu().numpy()

            # Quantize if needed (basic implementation)
            if quantization != "f16":
                # This is oversimplified - real quantization is complex
                logger.warning(f"Simplified quantization for {name}")
                np_tensor = np_tensor.astype(np.float16)

            writer.add_tensor(name, np_tensor)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()


def validate_export_config(
    vectors_dict: dict[str, torch.Tensor],
    alphas_dict: dict[str, float],
    config: LatentVectorConfig,
) -> tuple[bool, list[str]]:
    """
    Validate export configuration before merging.

    Args:
        vectors_dict: Dictionary of vector names to tensors
        alphas_dict: Dictionary of vector names to alpha values
        config: Model configuration

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check that all vectors exist
    if not vectors_dict:
        errors.append("No vectors provided for export")

    # Check that all alphas are provided
    missing_alphas = set(vectors_dict.keys()) - set(alphas_dict.keys())
    if missing_alphas:
        errors.append(f"Missing alpha values for vectors: {missing_alphas}")

    # Check alpha value ranges (warn about extreme values)
    for name, alpha in alphas_dict.items():
        if abs(alpha) > 5.0:
            errors.append(
                f"Warning: Very high alpha value for '{name}': {alpha}. "
                "This may cause model instability or incoherent outputs."
            )

    # Check vector shapes match
    if vectors_dict:
        shapes = [v.shape for v in vectors_dict.values()]
        if len(set(shapes)) > 1:
            errors.append(f"Vector shape mismatch: {shapes}")

    return len(errors) == 0, errors
