#!/bin/bash
# Install CTranslate2 4.7.1 with ROCm support from official release wheels.
#
# The standard pip ctranslate2 package is CUDA-only.
# Official ROCm wheels are published at:
#   https://github.com/OpenNMT/CTranslate2/releases/tag/v4.7.1
#
# Usage:
#   bash scripts/install_rocm_ct2.sh

set -euo pipefail

RELEASE_URL="https://github.com/OpenNMT/CTranslate2/releases/download/v4.7.1/rocm-python-wheels-Linux.zip"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

PYTHON_VERSION=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
echo "Detected Python: $PYTHON_VERSION"

echo "Downloading ROCm wheels from CTranslate2 v4.7.1 release..."
curl -L -o "$TMPDIR/rocm-wheels.zip" "$RELEASE_URL"

echo "Extracting..."
unzip -q "$TMPDIR/rocm-wheels.zip" -d "$TMPDIR"

# Find matching wheel for current Python version
WHEEL=$(find "$TMPDIR" -name "ctranslate2-4.7.1-${PYTHON_VERSION}-${PYTHON_VERSION}*-manylinux*.whl" | head -1)

if [ -z "$WHEEL" ]; then
    echo "Error: No wheel found for Python $PYTHON_VERSION"
    echo "Available wheels:"
    find "$TMPDIR" -name "*.whl" -exec basename {} \;
    exit 1
fi

echo "Installing: $(basename "$WHEEL")"
pip install --force-reinstall "$WHEEL"

echo ""
echo "Verifying ROCm detection..."
python3 -c "
import ctranslate2
try:
    types = ctranslate2.get_supported_compute_types('cuda')
    print(f'GPU compute types: {types}')
except Exception as e:
    print(f'No GPU detected (expected if no ROCm runtime): {e}')
    types = ctranslate2.get_supported_compute_types('cpu')
    print(f'CPU compute types: {types}')
"

echo ""
echo "CTranslate2 ROCm wheel installed successfully."
echo "Note: ROCm uses device='cuda' (HIP compatibility layer)."
