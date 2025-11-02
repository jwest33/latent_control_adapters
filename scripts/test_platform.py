#!/usr/bin/env python3
"""
Platform Validation and Testing Script

Tests installation, dependencies, and hardware compatibility
for Latent Control Adapters across different platforms.
"""

import sys
from pathlib import Path


def test_python_version():
    """Test Python version requirement."""
    print("\n" + "=" * 70)
    print("Testing Python Version")
    print("=" * 70)

    required = (3, 11)
    current = sys.version_info[:2]

    print(f"Current: Python {current[0]}.{current[1]}")
    print(f"Required: Python {required[0]}.{required[1]}+")

    if current >= required:
        print("Python version is compatible")
        return True
    else:
        print("Python version is too old")
        print(f"Please upgrade to Python {required[0]}.{required[1]} or higher")
        return False


def test_core_dependencies():
    """Test core package imports."""
    print("\n" + "=" * 70)
    print("Testing Core Dependencies")
    print("=" * 70)

    dependencies = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers",
        "tqdm": "TQDM",
        "click": "Click",
        "yaml": "PyYAML",
    }

    all_ok = True

    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"{name:30s} OK")
        except ImportError as e:
            print(f"{name:30s} FAILED: {e}")
            all_ok = False

    return all_ok


def test_optional_dependencies():
    """Test optional dependencies."""
    print("\n" + "=" * 70)
    print("Testing Optional Dependencies")
    print("=" * 70)

    try:
        import bitsandbytes

        print(f"BitsAndBytes {bitsandbytes.__version__:15s} OK")
        bnb_available = True
    except ImportError:
        print("BitsAndBytes                 NOT INSTALLED (4-bit quantization unavailable)")
        bnb_available = False

    return {"bitsandbytes": bnb_available}


def test_hardware():
    """Test hardware detection."""
    print("\n" + "=" * 70)
    print("Testing Hardware Detection")
    print("=" * 70)

    try:
        import torch

        # CUDA
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")

        if cuda_available:
            print(f"  Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                vram = props.total_memory / (1024**3)
                print(f"  Device {i}: {name} ({vram:.1f}GB VRAM)")

        # MPS (Apple Silicon)
        if hasattr(torch.backends, "mps"):
            mps_available = torch.backends.mps.is_available()
            print(f"MPS Available: {mps_available}")
        else:
            mps_available = False

        return {
            "cuda": cuda_available,
            "mps": mps_available,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
        }

    except Exception as e:
        print(f"Hardware detection failed: {e}")
        return {"cuda": False, "mps": False, "device_count": 0}


def test_latent_control_import():
    """Test latent_control package import."""
    print("\n" + "=" * 70)
    print("Testing Latent Control Imports")
    print("=" * 70)

    try:
        from latent_control import quick_start, WorkflowManager, get_preset

        print("latent_control package imports successfully")
        return True
    except ImportError as e:
        print(f"Failed to import latent_control: {e}")
        print("\nTo fix, run from repository root:")
        print("  pip install -e .")
        return False


def test_config_files():
    """Test configuration files exist."""
    print("\n" + "=" * 70)
    print("Testing Configuration Files")
    print("=" * 70)

    configs = [
        "configs/production.yaml",
        "configs/windows.yaml",
        "configs/macos.yaml",
        "configs/linux.yaml",
    ]

    all_exist = True
    for config in configs:
        path = Path(config)
        if path.exists():
            print(f"{config:30s} EXISTS")
        else:
            print(f"{config:30s} MISSING")
            all_exist = False

    return all_exist


def suggest_platform_config(hardware_info):
    """Suggest optimal config based on platform and hardware."""
    print("\n" + "=" * 70)
    print("Platform Configuration Recommendation")
    print("=" * 70)

    import platform

    system = platform.system()

    print(f"Detected Platform: {system}")

    if system == "Windows":
        print("Recommended config: configs/windows.yaml")
        print("  - 4-bit quantization disabled (bitsandbytes compatibility)")
        print("  - CUDA or CPU fallback")
    elif system == "Darwin":
        print("Recommended config: configs/macos.yaml")
        print("  - MPS support for Apple Silicon")
        print("  - CPU fallback for Intel Macs")
    elif system == "Linux":
        print("Recommended config: configs/linux.yaml")
        print("  - Full CUDA support")
        print("  - 4-bit quantization enabled")
    else:
        print("Recommended config: configs/production.yaml")
        print("  - Default safe configuration")

    if hardware_info["cuda"]:
        print(f"\nGPU acceleration available ({hardware_info['device_count']} device(s))")
    elif hardware_info["mps"]:
        print("\nMPS (Apple Silicon) acceleration available")
    else:
        print("\nNo GPU acceleration detected - will use CPU (slower)")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("LATENT CONTROL ADAPTERS - PLATFORM VALIDATION")
    print("=" * 70)

    results = {}

    # Run tests
    results["python"] = test_python_version()
    results["core_deps"] = test_core_dependencies()
    results["optional_deps"] = test_optional_dependencies()
    results["hardware"] = test_hardware()
    results["latent_control"] = test_latent_control_import()
    results["configs"] = test_config_files()

    # Suggest config
    suggest_platform_config(results["hardware"])

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_critical_ok = results["python"] and results["core_deps"] and results["latent_control"]

    if all_critical_ok:
        print("All critical checks passed")
        print("\nYou can now:")
        print("  1. Edit your chosen config file (set model_path)")
        print("  2. Run: latent-control train --config <your-config.yaml>")
        print("  3. Run: latent-control generate --config <your-config.yaml> --prompt 'your prompt'")
    else:
        print("Some critical checks failed")
        print("\nPlease fix the errors above before proceeding")

    if not results["optional_deps"]["bitsandbytes"]:
        print("\nNote: BitsAndBytes not installed")
        print("  4-bit quantization will be unavailable")
        print("  To install: pip install bitsandbytes")

    print("=" * 70)

    return 0 if all_critical_ok else 1


if __name__ == "__main__":
    sys.exit(main())
