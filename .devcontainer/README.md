
[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/TheOpenSI/QML-QPF)
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

If you already have VS Code and Docker installed, you can click the badge above or [here](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/TheOpenSI/QML-QPF) to get started. Clicking these links will cause VS Code to automatically install the Dev Containers extension if needed, clone the source code into a container volume, and spin up a dev container for use.

1. In VSCode, use the explore and select clone repository and select Clone from Github as repository source. Find TheOpenSI/QML-QPF choose it and select a location for the repository. 
2.  Open the cloned repository and in the lower left status bar click on main and you will have the option to select the branch you are working on or create a new branch.
3. In the Command Palette select "Dev Containers: Rebuild and Reopen in Container" or "Dev Containers: Open folder in Container..." select the folder that you cloned the repository to.
4. You should be presented with the option of a CPU or CUDA container, select CUDA if you intend to use GPU resources.


## Step 6: Building the Environment

1. After opening the folder in a DevContainer, VSCode will start building the container. This process can take some time as it involves downloading necessary images and setting up the environment.
2. You can monitor the progress in the VSCode terminal.
3. (Optional) If you intend to use GPU resources, from the Command Palette Run Dev Containers: Switch Container select the container with cuda. Test using tf.config.list_physical_devices('GPU')


## Step 8: Build 

This project is not structured as a package and cannot currently be built.

To build  from source, simply run:
   ```
   python setup.py develop
   ```

For an in-depth understanding of Dev Container and its caveats, please refer to [the full documentation](https://code.visualstudio.com/docs/devcontainers/containers).
