FROM mcr.microsoft.com/devcontainers/miniconda:latest

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV NVIDIA_REQUIRE_CUDA "cuda>=12.0"
ENV CONDA_OVERRIDE_CUDA="12.2"


# [Optional] Uncomment this section to install additional OS packages.

RUN sudo apt-get -y update \
    && sudo apt-get -y install openssh-server \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && sudo dpkg -i cuda-keyring_1.1-1_all.deb \
    && sudo apt-get -y update \
    && sudo apt-get -y install cuda-toolkit=12.2.* \
    && sudo apt-get -y install libcudnn8
 
#&& sudo apt-get -y install libcudnn8_8.9.5.30-1+cuda12.2
#apt-cache policy libcudnn8


RUN conda init

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY .devcontainer/cuda/environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env create -n QML-QPF -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp \
    && conda install -n QML-QPF -c rapidsai -c conda-forge -c nvidia -y rapids=24.06

# [Optional] Uncomment to install a different version of Python than the default
# RUN conda install -y python=3.6 \
#     && pip install --no-cache-dir pipx \
#     && pipx reinstall-all

EXPOSE 22
