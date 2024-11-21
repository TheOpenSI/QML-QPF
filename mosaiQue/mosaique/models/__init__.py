import os
import pathlib
import mosaique.models.kernels
import numpy as np
from typing import Any, Callable

class ConvolutionLayer4x4:
    _name: str
    _kernel: kernels.Kernel2d4x4


    @property
    def name(self) -> str :
        return self._name
    @name.setter
    def name(self, value: str):
        self._name = value
    @property
    def kernel(self) -> kernels.Kernel2d4x4:
        return self._kernel
    @kernel.setter
    def kernel(self, value: kernels.Kernel2d4x4):
        self._kernel = value

    def __init__(self, name: str, kernel_shape:[int]=None):
        if kernel_shape is None:
            kernel_shape = [2, 2]
        self.name = name
        self.kernel = kernels.Kernel2d4x4(kernel_shape)

    def fit(self, dataset: np.ndarray[..., np.dtype[Any]]):
        self.kernel.fit(dataset)

    def transform(self, dataset: np.ndarray[..., np.dtype[Any]]):
        return self.kernel.transform(dataset)

    def post_transform(self, dataset: np.ndarray[..., np.dtype[Any]]):
        return self.kernel.post_transform(dataset)

    def save(self, dataset: np.ndarray[..., np.dtype[Any]], variant:[int]):
        variant_string = ''.join(map(str,variant))
        workdir = str(pathlib.Path().resolve()) + "/" + self.name
        os.makedirs(workdir, exist_ok=True)
        np.save(os.path.join(workdir, variant_string), dataset[:,:,:,variant])

    def open(self, variant:[int]):
        variant_string = ''.join(map(str,variant))
        workdir = str(pathlib.Path().resolve()) + "/" + self.name
        return np.load(os.path.join(workdir, variant_string + '.npy'))

