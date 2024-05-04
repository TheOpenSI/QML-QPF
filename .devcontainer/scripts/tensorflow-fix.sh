# this fix for tensorflow 2.16.1 is sourced
# from a pending update to tf's documentation [https://github.com/tensorflow/docs/pull/2299] 
# waiting for full fix in tensorflow 2.16.2 or 2.17

NVIDIA_PACKAGE_DIR="/opt/conda/envs/QML-QPF/lib/python3.12/site-packages/nvidia"

sudo cp -v -r .devcontainer/scripts/activate.d /opt/conda/envs/QML-QPF/etc/conda

sudo cp -v -r .devcontainer/scripts/deactivate.d /opt/conda/envs/QML-QPF/etc/conda
