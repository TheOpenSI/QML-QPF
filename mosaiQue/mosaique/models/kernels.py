from typing import Any
import numpy as np

class Kernel2d4x4:
    _input_shape :[int]
    _kernel_shape :[int]

    @property
    def input_shape(self):
        return self._input_shape
    @input_shape.setter
    def input_shape(self, shape:[int]):
        self._input_shape = shape
    @property
    def kernel_shape(self):
        return self._kernel_shape
    @kernel_shape.setter
    def kernel_shape(self, shape:[int]):
        self._kernel_shape = shape

    def __init__(self, shape:[int]=None):
        if shape is None:
            shape = [2, 2]
        self.kernel_shape = shape

    def fit(self, X: np.ndarray[..., np.dtype[Any]]):
        self.input_shape = X.shape

    def transform(self, X: np.ndarray[..., np.dtype[Any]]):
        # shape MxN
        _m, _n = self.kernel_shape[:2]
        # Use slicing to extract all 2x2 patches from each 28x28 image
        # Reshape the array to extract all 2x2 patches
        # The number of 2x2 patches along height and width of each 28x28 image is 14
        return (X[:, :, :, np.newaxis]
                .reshape(self.input_shape[0], self.input_shape[2]//_m, _m, self.input_shape[2]//_n, _n)
                .transpose((0,1,3,4,2))
                .reshape(self.input_shape[0], -1, _m * _n)
                )

    def post_transform(self, X: np.ndarray[..., np.dtype[Any]]):
        # shape MxN
        _m, _n = self.kernel_shape[:2]
        return (X
                .reshape(self.input_shape[0], _n*_m, self.input_shape[2]//_n, self.input_shape[2]//_m)
                .transpose(0,2,3,1)
        )

    def channel_merge(self, X: np.ndarray[..., np.dtype[Any]]):
        # shape MxN
        _m, _n = self.kernel_shape[:2]
        return (X
                .reshape(self.input_shape[0], self.input_shape[2]//_m, self.input_shape[2]//_n, _m,_n)
                .transpose((0,1,2,4,3))
                .transpose((0, 1, 3, 2, 4))
                .reshape(self.input_shape[:3])
                )