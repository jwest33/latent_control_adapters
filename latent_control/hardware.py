"""
Hardware Detection and Validation Utilities

Provides cross-platform hardware detection and compatibility checking
for GPU availability, VRAM, and system resources.
"""

import platform
from typing import Dict, Optional

import torch


def get_system_info() -> Dict[str, str]:
    """
    Get basic system information.

    Returns:
        Dictionary with platform, Python version, and PyTorch version
    """
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def check_gpu_availability() -> Dict[str, any]:
    """
    Check GPU availability and report VRAM information.

    Returns:
        Dictionary with GPU status, device name, and VRAM info
    """
    result = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
        "recommended_device": "cpu",
    }

    if torch.cuda.is_available():
        result["device_count"] = torch.cuda.device_count()
        result["recommended_device"] = "cuda"

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
            }
            result["devices"].append(device_info)

    # Check for MPS (Apple Silicon) support
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        result["mps_available"] = True
        result["recommended_device"] = "mps"
    else:
        result["mps_available"] = False

    return result


def print_gpu_info():
    """Print formatted GPU information."""
    info = check_gpu_availability()

    print("\n" + "=" * 70)
    print("HARDWARE DETECTION")
    print("=" * 70)

    sys_info = get_system_info()
    print(f"\nPlatform: {sys_info['platform']} {sys_info['platform_release']}")
    print(f"Python:   {sys_info['python_version']}")
    print(f"PyTorch:  {sys_info['pytorch_version']}")

    print(f"\nCUDA Available: {info['cuda_available']}")

    if info["cuda_available"]:
        print(f"GPU Device Count: {info['device_count']}")
        for device in info["devices"]:
            print(f"\n  Device {device['id']}: {device['name']}")
            print(f"    VRAM: {device['total_memory_gb']:.1f} GB")
            print(f"    Compute Capability: {device['compute_capability']}")

    if info.get("mps_available"):
        print("\nMPS (Apple Silicon) Available: Yes")

    print(f"\nRecommended Device: {info['recommended_device']}")
    print("=" * 70)


def warn_cpu_performance():
    """Warn user about CPU performance if GPU is not being used."""
    print("\n" + "!" * 70)
    print("WARNING: Running on CPU")
    print("!" * 70)
    print("GPU acceleration is not available or not enabled.")
    print("Model loading and inference will be significantly slower on CPU.")
    print("\nTo enable GPU acceleration:")
    print("  1. Ensure CUDA-compatible GPU is installed")
    print("  2. Install PyTorch with CUDA support")
    print("  3. Set device: 'cuda' in your config file")
    print("!" * 70 + "\n")


def check_vram_requirements(
    model_size_gb: float, load_in_4bit: bool = False, num_pairs: int = 128
) -> Dict[str, any]:
    """
    Estimate VRAM requirements and check if sufficient VRAM is available.

    Args:
        model_size_gb: Approximate model size in GB (full precision)
        load_in_4bit: Whether 4-bit quantization is enabled
        num_pairs: Number of training pairs

    Returns:
        Dictionary with requirement estimates and availability status
    """
    # Estimate requirements
    if load_in_4bit:
        model_vram = model_size_gb * 0.25  # 4-bit is ~25% of full size
    else:
        model_vram = model_size_gb * 0.5  # FP16 is ~50% of full size

    # Additional VRAM for training (hidden states extraction)
    training_overhead = 0.5 + (num_pairs * 0.001)  # ~1MB per pair + base overhead

    # Generation overhead
    generation_overhead = 0.5  # Conservative estimate

    total_required = model_vram + max(training_overhead, generation_overhead)

    result = {
        "model_vram_gb": model_vram,
        "training_overhead_gb": training_overhead,
        "generation_overhead_gb": generation_overhead,
        "total_required_gb": total_required,
        "sufficient_vram": False,
        "available_vram_gb": 0,
    }

    # Check available VRAM
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        available_vram = device_props.total_memory / (1024**3)
        result["available_vram_gb"] = available_vram
        result["sufficient_vram"] = available_vram >= total_required

    return result


def print_vram_check(
    model_size_gb: float, load_in_4bit: bool = False, num_pairs: int = 128
):
    """Print formatted VRAM requirement check."""
    result = check_vram_requirements(model_size_gb, load_in_4bit, num_pairs)

    print("\n" + "=" * 70)
    print("VRAM REQUIREMENTS CHECK")
    print("=" * 70)
    print(f"\nModel Size (full precision): {model_size_gb:.1f} GB")
    print(f"4-bit Quantization: {'Enabled' if load_in_4bit else 'Disabled'}")
    print(f"Training Pairs: {num_pairs}")
    print(f"\nEstimated VRAM Usage:")
    print(f"  Model:      {result['model_vram_gb']:.2f} GB")
    print(f"  Training:   {result['training_overhead_gb']:.2f} GB")
    print(f"  Generation: {result['generation_overhead_gb']:.2f} GB")
    print(f"  Total:      {result['total_required_gb']:.2f} GB")

    if torch.cuda.is_available():
        print(f"\nAvailable VRAM: {result['available_vram_gb']:.2f} GB")
        if result["sufficient_vram"]:
            print("Sufficient VRAM available")
        else:
            print("WARNING: May not have sufficient VRAM")
            print("\nRecommendations:")
            print("  - Enable 4-bit quantization: load_in_4bit: true")
            print("  - Reduce training pairs: num_pairs: 64 or 32")
            print("  - Use a smaller model")
    else:
        print("\nNo GPU detected - running on CPU (no VRAM check needed)")

    print("=" * 70)


def suggest_optimal_config() -> str:
    """
    Suggest optimal configuration file based on detected hardware.

    Returns:
        Suggested config file path
    """
    sys_info = get_system_info()
    gpu_info = check_gpu_availability()

    system = sys_info["platform"].lower()

    # Windows
    if system == "windows":
        if gpu_info["cuda_available"]:
            # Windows with GPU - recommend windows.yaml (no 4-bit due to bitsandbytes issues)
            return "configs/windows.yaml"
        else:
            # Windows CPU-only
            return "configs/windows.yaml"

    # macOS
    elif system == "darwin":
        # macOS - use macos.yaml (MPS or CPU)
        return "configs/macos.yaml"

    # Linux
    elif system == "linux":
        if gpu_info["cuda_available"]:
            # Linux with GPU - best support for all features
            return "configs/linux.yaml"
        else:
            # Linux CPU-only
            return "configs/linux.yaml"

    # Default
    return "configs/production.yaml"


def validate_config_compatibility(config_dict: Dict) -> Dict[str, any]:
    """
    Validate if current configuration is compatible with detected hardware.

    Args:
        config_dict: Configuration dictionary from SystemConfig

    Returns:
        Dictionary with compatibility results and warnings
    """
    warnings = []
    errors = []

    gpu_info = check_gpu_availability()
    sys_info = get_system_info()

    # Check device setting
    device = config_dict.get("device", "cuda")

    if device == "cuda" and not gpu_info["cuda_available"]:
        warnings.append("Config specifies CUDA but no GPU detected. Will fallback to CPU.")

    if device == "cpu" and gpu_info["cuda_available"]:
        warnings.append(
            "Config specifies CPU but GPU is available. Consider using CUDA for better performance."
        )

    # Check 4-bit quantization
    load_in_4bit = config_dict.get("load_in_4bit", False)

    if load_in_4bit:
        if sys_info["platform"] == "Windows":
            warnings.append(
                "4-bit quantization enabled on Windows. BitsAndBytes may have compatibility issues. "
                "If you encounter errors, set load_in_4bit: false in config."
            )

        if not gpu_info["cuda_available"]:
            errors.append("4-bit quantization requires CUDA but no GPU detected.")

        if gpu_info.get("mps_available") and device == "mps":
            errors.append("4-bit quantization is not supported on MPS (Apple Silicon).")

    return {"warnings": warnings, "errors": errors, "compatible": len(errors) == 0}
