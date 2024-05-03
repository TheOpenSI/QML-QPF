# FROM mcr.microsoft.com/devcontainers/miniconda:latest

FROM tensorflow/tensorflow:latest-gpu-jupyter

# FROM xychelsea/tensorflow:latest-gpu-jupyter

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV NVIDIA_REQUIRE_CUDA "cuda>=12.0"
ENV CONDA_OVERRIDE_CUDA="12.4"

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# ENV ANACONDA_ENV=qml_env

# # Start as root
# USER root

# # Update packages
# RUN apt-get update --fix-missing \
#     && apt-get -y upgrade \
#     && apt-get -y dist-upgrade

# # Create the environment:
# COPY environment.yml .
# RUN conda env create -f environment.yml

# # Make RUN commands use the new environment: 
# SHELL ["conda", "run", "-n", "QML-QPF", "/bin/bash", "-c"]

RUN conda init

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY .devcontainer/cuda/environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env create -n QML-QPF -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment to install a different version of Python than the default
# RUN conda install -y python=3.6 \
#     && pip install --no-cache-dir pipx \
#     && pipx reinstall-all

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
