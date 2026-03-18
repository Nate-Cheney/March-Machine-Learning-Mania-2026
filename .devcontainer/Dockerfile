FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# Install prerequisites
RUN apt-get update && apt-get install -y \
    software-properties-common \
    clinfo \
    vainfo \
    wget \
    curl \
    git \
    sudo \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Add Intel's official Canonical PPA for Ubuntu 24.04 Battlemage drivers
RUN add-apt-repository -y ppa:kobuk-team/intel-graphics && \
    apt-get update

# Upgrade underlying DRM libraries and install the modern Xe2 packages
RUN apt-get upgrade -y && \
    apt-get install -y \
    intel-opencl-icd \
    libze-intel-gpu1 \
    libze1 \
    libze-dev \
    intel-ocloc \
    intel-media-va-driver-non-free \
    libmfx-gen1 \
    libvpl2 \
    && rm -rf /var/lib/apt/lists/*

# Install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Configure user groups
RUN (getent group video || groupadd video) && \
    (getent group render || groupadd render) && \
    usermod -aG video vscode && \
    usermod -aG render vscode

ENV SYCL_DEVICE_FILTER=level_zero:gpu
ENV ZES_ENABLE_SYSMAN=0
ENV SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
