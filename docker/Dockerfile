FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libegl1 git \
    && rm -rf /var/lib/apt/lists/*

ENV PYGLET_HEADLESS=1
ENV GIT_PYTHON_REFRESH=quiet

RUN pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir --no-deps genesis-world==0.4.7

RUN pip install --no-cache-dir \
    "rsl-rl-lib>=5.0.0" \
    tensordict \
    tensorboard \
    gymnasium

# Install genesis deps separately without letting them override torch
RUN pip install --no-cache-dir $(pip show genesis-world | grep Requires | sed 's/Requires: //' | tr ',' '\n' | grep -v torch) 2>/dev/null || true

WORKDIR /workspace