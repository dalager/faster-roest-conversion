#!/usr/bin/env python3
"""Convert CoRal roest-v3-whisper-1.5b to faster-whisper (CTranslate2) format.

This script converts a HuggingFace WhisperForConditionalGeneration model
with sharded safetensors into a CTranslate2 model directory that is
compatible with faster-whisper's WhisperModel.

Works with both NVIDIA CUDA and AMD ROCm GPUs (and CPU-only).

Usage:
    python scripts/convert_to_faster_whisper.py [--quantization QUANT] [--output-dir DIR]

The conversion runs on CPU and produces:
    model.bin              - CTranslate2 binary weights
    config.json            - CTranslate2 model config (auto-generated)
    tokenizer.json         - copied from source
    preprocessor_config.json - copied from source
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The HF cache layout stores the actual model in snapshots/<revision>/
MODEL_CACHE_DIR = PROJECT_ROOT / "models--CoRal-project--roest-v3-whisper-1.5b"
SNAPSHOTS_DIR = MODEL_CACHE_DIR / "snapshots"

# Files that faster-whisper needs alongside model.bin
COPY_FILES = ["tokenizer.json", "preprocessor_config.json"]

# Quantization options safe for this bfloat16-trained model.
# Avoid plain float16 — it causes numerical issues with bf16-trained weights.
SAFE_QUANTIZATIONS = {
    "bfloat16": "Matches training dtype. Needs Ampere+ GPU or ROCm with bf16.",
    "float32": "Full precision, largest (~6GB). Safe baseline.",
    "int8": "8-bit weights. Best for CPU inference.",
    "int8_float32": "int8 weights, fp32 compute. CPU-optimized.",
    "int8_float16": "int8 weights, fp16 compute. NVIDIA GPU.",
    "int8_bfloat16": "int8 weights, bf16 compute. Ampere+ GPU.",
}

DEFAULT_QUANTIZATION = "bfloat16"


def find_snapshot_dir() -> Path:
    """Find the model snapshot directory (resolves HF cache layout)."""
    if not SNAPSHOTS_DIR.exists():
        sys.exit(f"Error: snapshots dir not found at {SNAPSHOTS_DIR}")

    snapshots = [d for d in SNAPSHOTS_DIR.iterdir() if d.is_dir()]
    if not snapshots:
        sys.exit(f"Error: no snapshot directories in {SNAPSHOTS_DIR}")

    # Use the first snapshot (they're identical for this model)
    snapshot = snapshots[0]

    # Verify essential files exist
    required = ["config.json", "tokenizer.json", "preprocessor_config.json"]
    for f in required:
        if not (snapshot / f).exists():
            sys.exit(f"Error: required file {f} not found in {snapshot}")

    # Verify safetensors are present
    safetensors = list(snapshot.glob("model*.safetensors"))
    if not safetensors:
        sys.exit(f"Error: no safetensors files found in {snapshot}")

    return snapshot


def check_dependencies():
    """Verify ct2-transformers-converter is available."""
    result = subprocess.run(
        ["ct2-transformers-converter", "--help"],
        capture_output=True,
    )
    if result.returncode != 0:
        sys.exit(
            "Error: ct2-transformers-converter not found.\n"
            "Install with: pip install ctranslate2 transformers"
        )


def convert(source_dir: Path, output_dir: Path, quantization: str):
    """Run the CTranslate2 whisper conversion."""
    print(f"Source model:  {source_dir}")
    print(f"Output dir:    {output_dir}")
    print(f"Quantization:  {quantization}")
    print()

    # Read source config for info
    with open(source_dir / "config.json") as f:
        config = json.load(f)
    print(f"Model arch:    {config.get('architectures', ['unknown'])[0]}")
    print(f"Layers:        {config.get('encoder_layers', '?')} enc / {config.get('decoder_layers', '?')} dec")
    print(f"d_model:       {config.get('d_model', '?')}")
    print(f"Training dtype:{config.get('torch_dtype', '?')}")
    print(f"Vocab size:    {config.get('vocab_size', '?')}")
    print()

    cmd = [
        "ct2-transformers-converter",
        "--model", str(source_dir),
        "--output_dir", str(output_dir),
        "--copy_files", *COPY_FILES,
        "--quantization", quantization,
        "--force",
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Conversion failed with exit code {result.returncode}")

    # Verify output
    expected_files = ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json"]
    missing = [f for f in expected_files if not (output_dir / f).exists()]
    if missing:
        print(f"Warning: expected files missing from output: {missing}")
    else:
        print()
        print("Conversion successful! Output files:")
        for f in sorted(output_dir.iterdir()):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name:30s} {size_mb:8.1f} MB")

    print()
    print("To use with faster-whisper:")
    print()
    print("  from faster_whisper import WhisperModel")
    print()
    print(f'  model = WhisperModel("{output_dir}",')
    print('      device="cpu",  compute_type="int8")     # CPU')
    print('  #   device="cuda", compute_type="bfloat16")  # NVIDIA GPU')
    print('  #   device="cuda", compute_type="bfloat16")  # AMD ROCm (uses "cuda" via HIP)')
    print()
    print('  segments, info = model.transcribe("audio.wav", language="da")')
    print("  for seg in segments:")
    print('      print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")')


def main():
    parser = argparse.ArgumentParser(
        description="Convert roest-v3-whisper-1.5b to faster-whisper format"
    )
    parser.add_argument(
        "--quantization", "-q",
        default=DEFAULT_QUANTIZATION,
        choices=list(SAFE_QUANTIZATIONS.keys()),
        help=f"Quantization type (default: {DEFAULT_QUANTIZATION})",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=PROJECT_ROOT / "models--CoRal-project--roest-v3-whisper-1.5b-ct2",
        help="Output directory for converted model",
    )
    parser.add_argument(
        "--list-quantizations",
        action="store_true",
        help="List available quantization options and exit",
    )
    args = parser.parse_args()

    if args.list_quantizations:
        print("Available quantization options:")
        for name, desc in SAFE_QUANTIZATIONS.items():
            marker = " (default)" if name == DEFAULT_QUANTIZATION else ""
            print(f"  {name:20s} {desc}{marker}")
        return

    check_dependencies()
    source_dir = find_snapshot_dir()
    output_dir = Path(str(args.output_dir) + "_" + args.quantization)
    convert(source_dir, output_dir, args.quantization)


if __name__ == "__main__":
    main()
