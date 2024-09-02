from typing import Any
import numpy as np

class Kernel2d:
    input_shape :[int]
    kernel_shape :[int]

    def __init__(self, shape:[int]):
        self.kernel_shape = shape

    def fit(self, X: np.ndarray[..., np.dtype[Any]]):
        self.input_shape = X.shape


    def transform(self, X: np.ndarray[..., np.dtype[Any]]):
        # shape MxN
        _m, _n = self.kernel_shape[:2]
        return (X
                .reshape((self.input_shape[0],-1, _m, self.input_shape[2]//_n, _n))
                .transpose((0,1,3,2,4))
                .reshape((self.input_shape[0], np.prod(self.input_shape[1:3])//(_m*_n), (_m*_n)))
                )
