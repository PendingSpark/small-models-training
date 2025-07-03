# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:25.06-py3

# Set working directory
WORKDIR /workspace

# Install git and update pip
RUN apt-get update && apt-get install -y git && \
    pip install --upgrade pip && \
    pip install \
        "transformers>=4.52.4" \
        datasets \
        accelerate \
        torch


# Clone repositories (each in its own layer)
RUN git clone https://github.com/pytorch/torchtune.git
RUN git clone https://github.com/hiyouga/LLaMA-Factory.git
RUN git clone https://github.com/ggml-org/llama.cpp.git

# copy this repo into the new docker
COPY . ./small-models-training

WORKDIR /workspace/small-models-training
RUN ls
RUN pip install -r requirements.txt
