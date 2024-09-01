from typing import Any
import numpy as np


shape :[int] = [2,2]

def blocks2d(input2d: np.ndarray[..., np.dtype[Any]]):
    # shape MxN
    _m, _n = shape[:2]
    return (input2d
            .reshape((input2d.shape[0],-1, _m, input2d.shape[2]//_n, _n))
            .transpose((0,1,3,2,4))
            )
