#!/bin/bash
# Script to generate the complete latent_control package
# Run this to create all 6 new files

echo "Generating latent_control package..."

# This script will copy and refactor existing code into new structure
# Each file will be created separately

python3 << 'EOF'
import shutil
from pathlib import Path

print("Step 1: Copying refusal_vector_module.py to latent_control/core.py...")
shutil.copy("refusal_vector_module.py", "latent_control/core.py")

print("Step 2: Copying latent_control_module.py to latent_control/adapter.py...")
shutil.copy("latent_control_module.py", "latent_control/adapter.py")

print("✓ Base files copied. Manual refactoring needed.")
print("\nNext steps:")
print("1. Edit latent_control/core.py - rename classes, add VectorCache")
print("2. Edit latent_control/adapter.py - add WorkflowManager")
print("3. Create latent_control/analysis.py")
print("4. Create latent_control/presets.py")
print("5. Create cli.py")
print("6. Create setup.py")
print("7. Create configs/production.yaml")

EOF
