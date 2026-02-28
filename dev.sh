#!/bin/bash
set -euo pipefail

# Usage: ./dev.sh [--nvidia|--rocm] [--build]
# Default: --rocm (original behavior)

GPU="rocm"
BUILD=false

for arg in "$@"; do
  case "$arg" in
    --nvidia) GPU="nvidia" ;;
    --rocm)   GPU="rocm" ;;
    --build)  BUILD=true ;;
    --help|-h)
      echo "Usage: ./dev.sh [--nvidia|--rocm] [--build]"
      echo "  --nvidia  Use NVIDIA CUDA container"
      echo "  --rocm    Use AMD ROCm container (default)"
      echo "  --build   Force rebuild the image"
      exit 0 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

IMAGE_NAME="faster-roest-${GPU}"

# Build if requested or image doesn't exist
if $BUILD || ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
  echo "Building $IMAGE_NAME image..."
  docker build -t "$IMAGE_NAME" -f "Dockerfile.${GPU}" .
fi

if [ "$GPU" = "rocm" ]; then
  # ROCm: install CTranslate2 wheel from local cache at startup
  INIT_CMD='
if ! python3 -c "import ctranslate2" 2>/dev/null; then
  echo "Installing CTranslate2 ROCm wheel from cache..."
  PYVER=$(python3 -c "import sys; print(f\"cp{sys.version_info.major}{sys.version_info.minor}\")")
  unzip -qo /cache/rocm-python-wheels-Linux.zip -d /tmp/rocm-wheels
  pip install /tmp/rocm-wheels/temp-linux/ctranslate2-4.7.1-${PYVER}-${PYVER}*-manylinux*.whl
  rm -rf /tmp/rocm-wheels
fi
exec bash
'
  docker run -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 8G \
    -v "$(pwd)":/workspace \
    -v "$(pwd)/cache":/cache:ro \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -w /workspace \
    "$IMAGE_NAME" \
    bash -c "$INIT_CMD"

elif [ "$GPU" = "nvidia" ]; then
  docker run -it \
    --gpus all \
    --ipc=host \
    --shm-size 8G \
    -v "$(pwd)":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -w /workspace \
    "$IMAGE_NAME" \
    bash
fi
