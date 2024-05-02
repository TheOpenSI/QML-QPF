# Step by step guide on using DevContainer

This is a set of steps for setting up a development environment.

## Step 1: Install VSCode

1. Navigate to the [Visual Studio Code website](https://code.visualstudio.com/).
2. Download the appropriate installer for your operating system (Windows, Linux, or macOS).
3. Run the installer and follow the on-screen instructions to install VSCode on your system.


## Step 2: Install Docker

1. Follow the [official guide](https://docs.docker.com/get-docker/) to install Docker. Don't forget the [post installation steps](https://docs.docker.com/engine/install/linux-postinstall/).

If you are using [Visual Studio Code Remote - SSH](https://code.visualstudio.com/docs/remote/ssh), then you only need to install Docker in the remote host, not your local computer. And the following steps should be run in the remote host.


## Step 3: Install DevContainer Extension

1. Open VSCode, go to the Extensions view by clicking on the Extensions icon in the Activity Bar on the side of the window.
2. Search for "Dev Containers" in the Extensions view search bar.
3. Find the "Dev Containers" extension in the search results and click on the install button to install it.

You can also go to the extension's [homepage](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) and [documentation page](https://code.visualstudio.com/docs/devcontainers/containers) to find more details.


## Step 4 (Optional): Install NVIDIA Container Toolkit for GPU Usage

1. If you intend to use GPU resources, first ensure you have NVIDIA drivers installed on your system. Check if `nvidia-smi` works to verify your GPU setup.
2. Follow the [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker) to install the NVIDIA Container Toolkit.
3. After installation, verify that the toolkit is installed correctly by running:
   ```
   docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
   ```

## Step 5: Open in DevContainer

1. In VSCode, use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS) and run "Clone Repository in Container Volume..." then select "Clone a Repository from GitHub in a Container Volume"
2. Search for TheOpenSI/QML-QPF and then select the branch you are working on or create a new branch.


## Step 6: Building the Environment

1. After opening the folder in a DevContainer, VSCode will start building the container. This process can take some time as it involves downloading necessary images and setting up the environment.
2. You can monitor the progress in the VSCode terminal.
3. (Optional) If you intend to use GPU resources, from the Command Palette Run Dev Containers: Reopen in Container select "cuda".


## Step 8: Build 

This project is not structured as a package and cannot currently be built.

To build  from source, simply run:
   ```
   python setup.py develop
   ```

For an in-depth understanding of Dev Container and its caveats, please refer to [the full documentation](https://code.visualstudio.com/docs/devcontainers/containers).
