FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    software-properties-common \
    ffmpeg \
    git \
    libglib2.0-0 \
    libgl1 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update && apt-get install -y --no-install-recommends \
    nodejs \
    python3.12 \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && python3.12 -m ensurepip --upgrade \
    && python3.12 -m pip install --upgrade pip setuptools wheel

RUN python3.12 -m pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

RUN npm install -g @openai/codex@0.111.0

WORKDIR /workspace/sam3

CMD ["/bin/bash"]
