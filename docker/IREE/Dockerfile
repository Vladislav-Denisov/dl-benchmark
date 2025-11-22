FROM ubuntu_for_dli

# Install IREE
ARG IREE_VERSION=3.8.0
RUN python3 -m pip install iree-base-compiler==${IREE_VERSION} iree-base-runtime==${IREE_VERSION} iree-turbine==${IREE_VERSION} 

# Install dependencies
RUN python3 -m pip install opencv-python numpy

# Install onnx for model conversion
ARG ONNX_VERSION=1.19.1
RUN python3 -m pip install onnx==${ONNX_VERSION}

# Install torch for model conversion
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
RUN python3 -m pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}

WORKDIR /tmp/
