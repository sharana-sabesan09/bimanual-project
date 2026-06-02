FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libegl1 git \
    && rm -rf /var/lib/apt/lists/*

ENV PYGLET_HEADLESS=1
ENV GIT_PYTHON_REFRESH=quiet

# Install all dependencies first
RUN pip install --no-cache-dir \
    genesis-world==0.4.7 \
    "rsl-rl-lib>=5.0.0" \
    tensordict \
    tensorboard \
    gymnasium

# Force correct torch version LAST so nothing can overwrite it
RUN pip install --force-reinstall --no-deps \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

WORKDIR /workspace