# kernels.py
# This module defines custom transformations for image kernel.
# Author: Brian Recktenwall-Calvet
# Date: 12-18-2024
# Version: 1.0


from typing import Any
import numpy as np

class Kernel2d4x4:
    """
    A class for applying 2D kernel transformations to images.

    The kernel splits an input image into smaller patches of a specified size
    and provides methods for transforming the patches and reconstructing the
    original image.

    Attributes:
        _input_shape (list[int]): Stores the shape of the input data.
        _kernel_shape (list[int]): Stores the shape of the kernel.
    """

    _input_shape: [int]
    _kernel_shape: [int]

    @property
    def input_shape(self):
        """
        Getter for the input shape.

        Returns:
            list[int]: The shape of the input data.
        """
        return self._input_shape

    @input_shape.setter
    def input_shape(self, shape: [int]):
        """
        Setter for the input shape.

        Args:
            shape (list[int]): The shape to set for the input data.
        """
        self._input_shape = shape

    @property
    def kernel_shape(self):
        """
        Getter for the kernel shape.

        Returns:
            list[int]: The shape of the kernel.
        """
        return self._kernel_shape

    @kernel_shape.setter
    def kernel_shape(self, shape: [int]):
        """
        Setter for the kernel shape.

        Args:
            shape (list[int]): The shape to set for the kernel.
        """
        self._kernel_shape = shape

    def __init__(self, shape: [int] = None):
        """
        Initializes the Kernel2d4x4 class.

        Args:
            shape (list[int], optional): The shape of the kernel. Defaults to [2, 2].
        """
        if shape is None:
            shape = [2, 2]
        self.kernel_shape = shape

    def fit(self, X: np.ndarray[..., np.dtype[Any]]):
        """
        Fits the kernel to the input data by storing its shape.

        Args:
            X (np.ndarray): Input data of shape [batch_size, height, width, channels].
        """
        self.input_shape = X.shape

    def transform(self, X: np.ndarray[..., np.dtype[Any]]):
        """
        Splits the input data into smaller patches based on the kernel shape.

        Args:
            X (np.ndarray): Input data of shape [batch_size, height, width, channels].

        Returns:
            np.ndarray: Reshaped array of shape [batch_size, number_of_patches, patch_size].
        """
        _m, _n = self.kernel_shape[:2]  # Extract kernel dimensions
        return (X[:, :, :, np.newaxis]
                .reshape(self.input_shape[0], self.input_shape[2] // _m, _m, self.input_shape[2] // _n, _n)
                .transpose((0, 1, 3, 4, 2))
                .reshape(self.input_shape[0], -1, _m * _n))

    def post_transform(self, X: np.ndarray[..., np.dtype[Any]]):
        """
        Reconstructs the image data from patches.

        Args:
            X (np.ndarray): Transformed data of shape [batch_size, number_of_patches, patch_size].

        Returns:
            np.ndarray: Reconstructed image data of shape [batch_size, height, width, channels].
        """
        _m, _n = self.kernel_shape[:2]  # Extract kernel dimensions
        return (X
                .reshape(self.input_shape[0], _n * _m, self.input_shape[2] // _n, self.input_shape[2] // _m)
                .transpose(0, 2, 3, 1))

    def channel_merge(self, X: np.ndarray[..., np.dtype[Any]]):
        """
        Merges channels after transformation to restore original channel order.

        Args:
            X (np.ndarray): Transformed data of shape [batch_size, height, width, channels].

        Returns:
            np.ndarray: Restored data of shape [batch_size, height, width, channels].
        """
        _m, _n = self.kernel_shape[:2]  # Extract kernel dimensions
        return (X
                .reshape(self.input_shape[0], self.input_shape[2] // _m, self.input_shape[2] // _n, _m, _n)
                .transpose((0, 1, 2, 4, 3))
                .transpose((0, 1, 3, 2, 4))
                .reshape(self.input_shape[:3]))
