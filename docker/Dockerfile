# Start from the NVIDIA CUDA image to get both CUDA and the GPU libraries. See https://hub.docker.com/r/nvidia/cuda AND https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/unsupported-tags.md
FROM nvidia/cuda:11.3.1-base-ubuntu20.04
LABEL maintainer="Lorenzo Balzani"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /workspace

# Install general-purpose packages
RUN apt-get update -y && \
    apt-get install -y curl \
		               wget \
                       git \
                       bash \
                       nano \
		               git-lfs \
                       cuda \
                       python3.9 \
                       python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install project dependens
RUN python3.9 -m pip install -r requirements.txt

# Install CUDA packages
## Install correct PyTorch packages
ARG TORCH_VERSION="2.2.0+cu118"
ARG TORCHVISION_VERSION="0.17.0+cu118"
RUN python3.9 -m pip install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION  \
    torchaudio==$TORCH_VERSION -f https://download.pytorch.org/whl/cu118/torch_stable.html

## Install RAPIDS dependencies
RUN python3.9 -m pip install cudf-cu11 cugraph-cu11 --extra-index-url https://pypi.nvidia.com

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog
