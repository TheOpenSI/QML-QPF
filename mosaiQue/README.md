# mosaiQue

A Python library for applying quantum convolutions with PennyLane, designed to facilitate image processing and transformation through custom convolution layers and quantum computations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the library, clone the repository and install the required dependencies including conda-build

use develop to install the library

sudo /opt/conda/bin/conda-develop -n QML-QPF PATH /workspaces/QML-QPF/mosaiQue

## Usage

Overview of the main classes in the library:

### ConvolutionLayer4x4

- Create a convolution layer with a custom kernel shape.
- Fit the kernel to your dataset.
- Transform the dataset using convolution.
- Save a portion of the dataset or load a previously saved dataset.

### QuantumLayer

- Define a quantum kernel operation using PennyLane.
- Create an operation layer using the quantum kernel.
- Preprocess images using a Keras model that incorporates the quantum layer.

### Kernel2d4x4

- Initialize the kernel with a specific shape.
- Fit the kernel to the image data.
- Transform the image data into patches and reconstruct the original image data from patches.
- Merge channels back to the original shape.

## API Reference

### `ConvolutionLayer4x4`

#### Attributes
- **name** (str): The name of the convolution layer.
- **kernel** (kernels.Kernel2d4x4): The kernel used for convolution operations.

#### Methods
- **fit(dataset: np.ndarray)**: Fits the kernel to the provided dataset.
- **transform(dataset: np.ndarray) -> np.ndarray**: Applies the convolution transformation to the dataset.
- **post_transform(dataset: np.ndarray) -> np.ndarray**: Applies post-processing to the dataset after transformation.
- **channel_merge(dataset: np.ndarray) -> np.ndarray**: Merges channels in the dataset.
- **save(dataset: np.ndarray, variant: List[int])**: Saves a portion of the dataset to a `.npy` file based on the variant.
- **open(variant: List[int]) -> np.ndarray**: Loads a saved dataset from a `.npy` file based on the variant.

### `QuantumLayer`

#### Attributes
- **q_node** (qml.QNode): The quantum node to execute quantum computations.

#### Methods
- **call(inputs: tf.Tensor) -> tf.Tensor**: Executes the quantum layer on the input tensor.

### `OperationLayer`

#### Attributes
- **q_layer** (QuantumLayer): The quantum layer for processing inputs.
- **kernel_operation** (qml.QNode): The QNode for the kernel operation.

#### Methods
- **pre_op() -> keras.Model**: Initializes and returns a custom Keras model used to preprocess images.

### `Kernel2d4x4`

#### Attributes
- **input_shape** (list[int]): The shape of the input data.
- **kernel_shape** (list[int]): The shape of the kernel.

#### Methods
- **fit(X: np.ndarray)**: Fits the kernel to the input data by storing its shape.
- **transform(X: np.ndarray) -> np.ndarray**: Splits the input data into smaller patches based on the kernel shape.
- **post_transform(X: np.ndarray) -> np.ndarray**: Reconstructs the image data from patches.
- **channel_merge(X: np.ndarray) -> np.ndarray**: Merges channels to restore the original order.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push to your branch and create a pull request.

## License

This project is licensed under the GPL-3.0 License. See the license file for details.
