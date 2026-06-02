FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# System libs genesis deps need (libGL, glib, OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Tell pyglet to use EGL (headless) instead of trying to connect to X11
ENV PYGLET_HEADLESS=1

# # PyTorch nightly cu130 (bundles its own CUDA runtime via pip wheels)
# RUN pip install --no-cache-dir torch torchvision \
#     --index-url https://download.pytorch.org/whl/nightly/cu130
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
RUN pip install --no-cache-dir \
    genesis-world==0.4.7 \
    "rsl-rl-lib>=5.0.0" \
    tensordict \
    tensorboard \
    gymnasium

WORKDIR /workspace
