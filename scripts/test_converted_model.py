#!/usr/bin/env python3
"""Quick smoke test for a converted faster-whisper model.

Loads the converted CTranslate2 model and runs a short transcription test.
If no audio file is provided, it just verifies the model loads correctly.

Usage:
    python scripts/test_converted_model.py [audio.wav]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models--CoRal-project--roest-v3-whisper-1.5b-ct2"


def test_model_loads(model_dir: Path, device: str, compute_type: str):
    """Verify the model loads without errors."""
    from faster_whisper import WhisperModel

    print(f"Loading model from: {model_dir}")
    print(f"Device: {device}, compute_type: {compute_type}")

    model = WhisperModel(str(model_dir), device=device, compute_type=compute_type)
    print("Model loaded successfully!")
    return model


def test_transcribe(model, audio_path: str):
    """Run transcription on an audio file."""
    print(f"\nTranscribing: {audio_path}")
    segments, info = model.transcribe(audio_path, language="da")

    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    print(f"Duration: {info.duration:.1f}s")
    print(info)

    for segment in segments:
        print(f"[{segment.start:6.2f}s -> {segment.end:6.2f}s] {segment.text}")


def detect_gpu_vendor() -> str:
    """Detect whether we're running on NVIDIA or AMD GPU."""
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        if result.returncode == 0:
            return "nvidia"
    except FileNotFoundError:
        pass
    try:
        if Path("/dev/kfd").exists():
            return "amd"
    except Exception:
        pass
    return "none"


def detect_device():
    """Detect best available device and compute type."""
    vendor = detect_gpu_vendor()

    try:
        import ctranslate2
        supported = ctranslate2.get_supported_compute_types("cuda")
        if supported:
            print(f"GPU detected ({vendor}). Supported compute types: {supported}")
            compute_type = "float16" if "float16" in supported else "float32"
            return "cuda", compute_type
    except Exception:
        pass

    print("No GPU detected, using CPU with auto compute type.")
    return "cpu", "auto"


def main():
    parser = argparse.ArgumentParser(description="Test converted faster-whisper model")
    parser.add_argument("audio", nargs="?", help="Audio file to transcribe (optional)")
    parser.add_argument(
        "--model-dir", "-m",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument("--device", "-d", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--compute-type", "-c", default=None)
    args = parser.parse_args()

    if not args.model_dir.exists():
        sys.exit(
            f"Model directory not found: {args.model_dir}\n"
            "Run the converter first:\n"
            "  python scripts/convert_to_faster_whisper.py"
        )

    if args.device == "auto":
        device, default_compute = detect_device()
    else:
        device = args.device
        default_compute = "auto"

    compute_type = args.compute_type or default_compute

    model = test_model_loads(args.model_dir, device, compute_type)

    if args.audio:
        test_transcribe(model, args.audio)
    else:
        print("\nNo audio file provided. Pass an audio file to test transcription:")
        print(f"  python {sys.argv[0]} /path/to/audio.wav")


if __name__ == "__main__":
    main()
