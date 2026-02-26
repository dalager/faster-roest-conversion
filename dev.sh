#!/bin/bash

IMAGE_NAME="faster-roest"

# Build image if it doesn't exist or if --build flag is passed
if [[ "$1" == "--build" ]] || ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
  echo "Building $IMAGE_NAME image..."
  docker build -t "$IMAGE_NAME" .
fi

# Install CTranslate2 ROCm wheel from local cache on first run
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
  -p 6006:6006 \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/cache":/cache:ro \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace \
  "$IMAGE_NAME" \
  bash -c "$INIT_CMD"
