# fasterroest

Convert [CoRal roest-v3-whisper-1.5b](https://huggingface.co/CoRal-project/roest-v3-whisper-1.5b) to [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) format for fast Danish speech-to-text.

Supports **NVIDIA CUDA** and **AMD ROCm** GPUs, plus CPU-only inference.

## Prerequisites

- Docker with GPU support ([NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) or [ROCm](https://rocm.docs.amd.com/))
- Git LFS (`sudo apt install git-lfs`)

## Download the model

```bash
git lfs install
git clone https://huggingface.co/CoRal-project/roest-v3-whisper-1.5b \
  models--CoRal-project--roest-v3-whisper-1.5b/snapshots/main
```

Or via `huggingface-cli`:

```bash
pip install huggingface-hub
huggingface-cli download CoRal-project/roest-v3-whisper-1.5b \
  --local-dir models--CoRal-project--roest-v3-whisper-1.5b/snapshots/main
```

## Setup

### NVIDIA GPU

```bash
./dev.sh --nvidia --build
```

CTranslate2 with CUDA support is installed from pip automatically.

### AMD ROCm GPU

Download the ROCm CTranslate2 wheel first (one-time, ~284 MB):

```bash
mkdir -p cache
wget -O cache/rocm-python-wheels-Linux.zip \
  https://github.com/OpenNMT/CTranslate2/releases/download/v4.7.1/rocm-python-wheels-Linux.zip
```

Then build and launch:

```bash
./dev.sh --rocm --build
```

## Convert the model

Inside the container (same commands for both GPUs):

```bash
# float32 (lossless, ~6 GB)
python scripts/convert_to_faster_whisper.py

# int8 (smaller, good for CPU)
python scripts/convert_to_faster_whisper.py -q int8

# See all quantization options
python scripts/convert_to_faster_whisper.py --list-quantizations
```

## Test

```bash
# Verify model loads
python scripts/test_converted_model.py

# Transcribe audio
python scripts/test_converted_model.py audio.wav
```

## Usage with faster-whisper

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    "models--CoRal-project--roest-v3-whisper-1.5b-ct2_float32",
    device="cuda",          # works for both NVIDIA and ROCm
    compute_type="float16",
)

segments, info = model.transcribe("audio.wav", language="da")
for seg in segments:
    print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")
```

## Quantization guide

| Quantization | Size | Best for |
|---|---|---|
| `float32` | ~6 GB | Safe baseline, any hardware |
| `bfloat16` | ~3 GB | Ampere+ NVIDIA or ROCm with bf16 |
| `int8` | ~1.5 GB | CPU inference |
| `int8_float16` | ~1.5 GB | NVIDIA GPU |
| `int8_bfloat16` | ~1.5 GB | Ampere+ GPU |
| `int8_float32` | ~1.5 GB | CPU (optimized) |

## Notes

- The model is trained in `bfloat16` -- avoid `float16` quantization during conversion (numerical issues). Use `float32`, `int8`, or `bfloat16`.
- ROCm presents as `device="cuda"` through HIP compatibility.
- For CPU-only: standard pip `ctranslate2` with `compute_type="int8"` or `"auto"`.
- ROCm GPU requires the [CTranslate2 v4.7.1 ROCm wheel](https://github.com/OpenNMT/CTranslate2/releases/tag/v4.7.1).
- NVIDIA GPU: standard `pip install ctranslate2` includes CUDA support.
