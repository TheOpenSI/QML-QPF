from numpy.ma.core import product

from mosaique.models.kernels import Kernel2d4x4
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from typing import Any

dataset: np.ndarray[..., np.dtype[Any]] =(
        np.ones(20)[:,None,None] *
        np.arange(28*28).reshape(28,28)[None,:,:]
)

dev = qml.device("default.qubit.tf", wires=4)
@qml.qnode(dev, interface='tf')
def qcnot_node(q1,q2,q3,q4):

    qml.RY(np.pi * q1, wires=0)
    qml.RY(np.pi * q2, wires=1)
    qml.RY(np.pi * q3, wires=2)
    qml.RY(np.pi * q4, wires=3)

    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[0, 3])

    # Measurement producing 4 classical output values
    return pnp.asarray([qml.expval(qml.PauliZ(wires=0)),qml.expval(qml.PauliZ(wires=1)),
                        qml.expval(qml.PauliZ(wires=2)),qml.expval(qml.PauliZ(wires=3))])


#data_blocks = kernel.inverse_transform(data_blocks)

#print(data_blocks[0])