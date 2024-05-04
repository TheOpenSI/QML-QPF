#!/bin/sh

# Store original LD_LIBRARY_PATH 
export ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" 

# Get the CUDNN directory 
CUDNN_DIR=$(dirname $(dirname $(python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__)")))

# Set LD_LIBRARY_PATH to include CUDNN directory
export LD_LIBRARY_PATH=$(find ${CUDNN_DIR}/*/lib/ -type d -printf "%p:")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Get the ptxas directory  
PTXAS_DIR=$(dirname $(dirname $(python -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__file__)")))

# Set PATH to include the directory containing ptxas
export PATH=$(find ${PTXAS_DIR}/*/bin/ -type d -printf "%p:")${PATH:+:${PATH}}