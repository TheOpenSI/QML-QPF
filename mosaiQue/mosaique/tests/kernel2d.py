

from mosaique.models.kernels import Kernel2d
import numpy as np
from typing import Any


dataset: np.ndarray[..., np.dtype[Any]] =(
        np.ones(20)[:,None,None] *
        np.arange(28*28).reshape(28,28)[None,:,:]
)

kernel_shape = [2,2]

kernel = Kernel2d(kernel_shape)

kernel.fit(dataset[[0,1]])

data_blocks = kernel.transform(dataset[[0,1]])

print(data_blocks[0])