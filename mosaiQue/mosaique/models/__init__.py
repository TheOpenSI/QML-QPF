import kernels
import numpy as np
from typing import Any, Callable

class Custom:
    name: str
    kernel_operation: Callable[..., Any]
    kernel: kernels.Kernel2d

    def __init__(self, name: str, kernel_operation: Callable[..., Any], kernel_shape: [int]):
        self.kernel_operation = kernel_operation
        self.name = name
        self.kernel = kernels.Kernel2d(kernel_shape)

    def fit(self, dataset: np.ndarray[..., np.dtype[Any]]):
        self.kernel.fit(dataset)

    def transform(self, dataset: np.ndarray[..., np.dtype[Any]]):
            return self.kernel_operation(self.kernel.transform(dataset))


