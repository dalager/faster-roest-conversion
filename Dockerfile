FROM rocm/pytorch:latest

# Better shell experience
RUN apt-get update && apt-get install -y --no-install-recommends \
  bash-completion \
  vim \
  less \
  unzip \
  wget \
  && rm -rf /var/lib/apt/lists/* \
  && echo 'PS1="\[\033[1;32m\]\u@container\[\033[0m\]:\[\033[1;34m\]\w\[\033[0m\]\$ "' >> /root/.bashrc \
  && echo 'alias ll="ls -la"' >> /root/.bashrc \
  && echo 'alias la="ls -A"' >> /root/.bashrc


RUN curl -LsSf https://astral.sh/uv/install.sh | sh;

WORKDIR /workspace

# CTranslate2 ROCm wheel is installed at container start from mounted cache.
# See dev.sh — cache/rocm-python-wheels-Linux.zip is mounted to /cache/

# Copy package files for installation
COPY pyproject.toml ./

# Install Python dependencies (ctranslate2 ROCm installed at runtime from cache)
RUN pip install --no-cache-dir transformers faster-whisper
