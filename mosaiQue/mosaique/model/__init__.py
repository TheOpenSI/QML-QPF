

import kernel
import numpy as np
from typing import Any, Callable

name: str = "model_name"
operation: Callable[..., Any]
dataset: np.ndarray[..., np.dtype[Any]] =(
        np.ones(20)[:,None,None] *
        np.arange(28*28).reshape(28,28)[None,:,:]
)
kernel.shape = [2,2]
datablocks = kernel.blocks2d(dataset[[0,1]])

print(datablocks[0])