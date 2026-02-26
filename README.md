# fasterroest

Convert the [CoRal roest-v3-whisper-1.5b](https://huggingface.co/CoRal-project/roest-v3-whisper-1.5b) model to [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) format for fast Danish speech-to-text on AMD ROCm GPUs.

The converted model is a drop-in replacement for [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3).

## Prerequisites

- Docker with ROCm support (AMD GPU)
- Git LFS (`sudo apt install git-lfs` or `brew install git-lfs`)

## Download the model

Clone the [roest-v3-whisper-1.5b](https://huggingface.co/CoRal-project/roest-v3-whisper-1.5b) model from HuggingFace into the project directory:

```bash
git lfs install
git clone https://huggingface.co/CoRal-project/roest-v3-whisper-1.5b \
  models--CoRal-project--roest-v3-whisper-1.5b/snapshots/main
```

Alternatively, download using the `huggingface-cli`:

```bash
pip install huggingface-hub
huggingface-cli download CoRal-project/roest-v3-whisper-1.5b \
  --local-dir models--CoRal-project--roest-v3-whisper-1.5b/snapshots/main
```

The model is ~3GB (two sharded safetensors files). After downloading, the directory should contain `config.json`, `tokenizer.json`, `preprocessor_config.json`, and `model-00001-of-00002.safetensors` / `model-00002-of-00002.safetensors`.

## Setup

Download the CTranslate2 ROCm wheel (one-time, ~284MB):

```bash
mkdir -p cache
wget -O cache/rocm-python-wheels-Linux.zip \
  https://github.com/OpenNMT/CTranslate2/releases/download/v4.7.1/rocm-python-wheels-Linux.zip
```

Build and launch the container:

```bash
./dev.sh --build
```

The ROCm CTranslate2 wheel is installed automatically from the local cache on container start.

## Convert the model

Inside the container:

```bash
# float32 (lossless, ~6GB)
python scripts/convert_to_faster_whisper.py

# int8 (smaller, good for CPU)
python scripts/convert_to_faster_whisper.py -q int8

# See all quantization options
python scripts/convert_to_faster_whisper.py --list-quantizations
```

Output goes to `models--CoRal-project--roest-v3-whisper-1.5b-ct2/`.

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
    "models--CoRal-project--roest-v3-whisper-1.5b-ct2",
    device="cuda",          # ROCm uses "cuda" via HIP
    compute_type="float16"  # safest on ROCm
)

segments, info = model.transcribe("audio.wav", language="da")
for seg in segments:
    print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")
```

## Notes

- The model is trained in `bfloat16` — avoid `float16` quantization during conversion (causes numerical issues). Use `float32`, `int8`, or `bfloat16`.
- ROCm presents as `device="cuda"` through HIP compatibility.
- For CPU-only usage, `compute_type="int8"` or `"auto"` works with the standard pip `ctranslate2` package.
- ROCm GPU requires the official ROCm wheel from [CTranslate2 v4.7.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v4.7.1).
