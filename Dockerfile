FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set noninteractive mode to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (excluding nvidia-cuda-toolkit)
RUN apt update && apt install --allow-change-held-packages -y \
    ubuntu-drivers-common \
    wget \
    unzip \
    curl \
    git \
    build-essential \
    libnccl-dev \
    libnccl2 \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Download and extract libtorch
WORKDIR /app
RUN wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu124.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.4.1+cu124.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-2.4.1+cu124.zip

# Set libtorch environment variables
ENV LIBTORCH=/app/libtorch
ENV LIBTORCH_INCLUDE=/app/libtorch
ENV LIBTORCH_LIB=/app/libtorch
ENV LD_LIBRARY_PATH=/app/libtorch/lib:$LD_LIBRARY_PATH
ENV CUDA_ROOT=/usr/local/cuda-12.4
ENV OPENSSL_DIR=/usr
ENV OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu
ENV OPENSSL_INCLUDE_DIR=/usr/include
ENV PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig

# Install Rust (Ensuring it's in PATH for all sessions)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> /root/.bashrc

ENV PATH="/root/.cargo/bin:$PATH"

# Clone the Nous Psyche repository
WORKDIR /app
RUN git clone https://github.com/zrross11/Psyche.git

# Set working directory and dataset directory before building
WORKDIR /app/Psyche
RUN mkdir -p ./data/finetune-10bt/
VOLUME ["/app/Psyche/data/finetune-10bt"]
RUN cargo build --release

# Set ENTRYPOINT to the default command
ENTRYPOINT ["sh", "-c", "if [ -z \"$(ls -A /app/Psyche/data/finetune-10bt)\" ]; then wget -O /app/data/00002_00001_shuffled.ds https://huggingface.co/datasets/emozilla/fineweb-10bt-tokenized-datatrove-llama2/resolve/main/00002_00001_shuffled.ds; fi && exec $0"]

# Default command to run training
CMD ["cargo", "run", "--example", "train", "--", "--model", "emozilla/llama2-20m-init", "--data-path", "./data/finetune-10bt/", "--total-batch", "2", "--micro-batch", "1"]
