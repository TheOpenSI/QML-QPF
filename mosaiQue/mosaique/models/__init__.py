# __init__.py
# This module defines custom transformations for image convolution.
# Author: Brian Recktenwall-Calvet
# Date: 12-18-2024
# Version: 1.0

import os
import pathlib
import mosaique.models.kernels
import numpy as np
from typing import Any, List, Optional


class ConvolutionLayer4x4:
    """
    A class representing a 4x4 convolution layer using a specified kernel.

    Attributes:
        _name (str): The name of the convolution layer.
        _kernel (kernels.Kernel2d4x4): An instance of the kernel used for convolution operations.

    Methods:
        fit(dataset): Fits the kernel to the provided dataset.
        transform(dataset): Applies the convolution transformation to the dataset.
        post_transform(dataset): Applies post-processing to the dataset after transformation.
        channel_merge(dataset): Merges channels in the dataset.
        save(dataset, variant): Saves a portion of the dataset to a .npy file based on the variant.
        open(variant): Loads a saved dataset from a .npy file based on the variant.
    """

    _name: str
    _kernel: kernels.Kernel2d4x4

    @property
    def name(self) -> str:
        """Gets the name of the convolution layer."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Sets the name of the convolution layer."""
        self._name = value

    @property
    def kernel(self) -> kernels.Kernel2d4x4:
        """Gets the kernel used for convolution operations."""
        return self._kernel

    @kernel.setter
    def kernel(self, value: kernels.Kernel2d4x4):
        """Sets the kernel used for convolution operations."""
        self._kernel = value

    def __init__(self, name: str, kernel_shape: Optional[List[int]] = None):
        """
        Initializes the ConvolutionLayer4x4 with a given name and kernel shape.

        Args:
            name (str): The name of the convolution layer.
            kernel_shape (Optional[List[int]]): The shape of the kernel (default is [2, 2]).

        Raises:
            ValueError: If kernel_shape is not a valid shape for the kernel.
        """
        if kernel_shape is None:
            kernel_shape = [2, 2]
        self.name = name
        self.kernel = kernels.Kernel2d4x4(kernel_shape)

    def fit(self, dataset: np.ndarray[..., np.dtype[Any]]):
        """
        Fits the kernel to the provided dataset.

        Args:
            dataset (np.ndarray): The input dataset used to fit the kernel.
        """
        self.kernel.fit(dataset)

    def transform(self, dataset: np.ndarray[..., np.dtype[Any]]) -> np.ndarray:
        """
        Applies the convolution transformation to the dataset.

        Args:
            dataset (np.ndarray): The input dataset to transform.

        Returns:
            np.ndarray: The transformed dataset after applying the convolution.
        """
        return self.kernel.transform(dataset)

    def post_transform(self, dataset: np.ndarray[..., np.dtype[Any]]) -> np.ndarray:
        """
        Applies post-processing to the dataset after transformation.

        Args:
            dataset (np.ndarray): The input dataset to post-process.

        Returns:
            np.ndarray: The post-processed dataset.
        """
        return self.kernel.post_transform(dataset)

    def channel_merge(self, dataset: np.ndarray[..., np.dtype[Any]]) -> np.ndarray:
        """
        Merges channels in the dataset, typically for multi-channel inputs.

        Args:
            dataset (np.ndarray): The input dataset whose channels will be merged.

        Returns:
            np.ndarray: The dataset with merged channels.
        """
        return self.kernel.channel_merge(dataset)

    def save(self, dataset: np.ndarray[..., np.dtype[Any]], variant: List[int]):
        """
        Saves a portion of the dataset to a .npy file based on the variant.

        Args:
            dataset (np.ndarray): The dataset to save.
            variant (List[int]): A list of indices specifying which channels to save.
        """
        variant_string = ''.join(map(str, variant))
        workdir = str(pathlib.Path().resolve()) + "/" + self.name
        os.makedirs(workdir, exist_ok=True)
        np.save(os.path.join(workdir, variant_string), dataset[:, :, :, variant])

    def open(self, variant: List[int]) -> np.ndarray:
        """
        Loads a saved dataset from a .npy file based on the variant.

        Args:
            variant (List[int]): A list of indices specifying which channels to load.

        Returns:
            np.ndarray: The loaded dataset.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        variant_string = ''.join(map(str, variant))
        workdir = str(pathlib.Path().resolve()) + "/" + self.name
        return np.load(os.path.join(workdir, variant_string + '.npy'))
