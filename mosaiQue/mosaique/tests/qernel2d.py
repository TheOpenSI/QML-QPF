from mosaique.models.kernels import Kernel2d4x4
from mosaique.models.operation import OperationLayer
import pennylane as qml
import numpy as np
from typing import Any

dataset: np.ndarray[..., np.dtype[Any]] =(
        np.ones(20)[:,None,None] *
        np.arange(28*28).reshape(28,28)[None,:,:]
)

dev = qml.device("default.qubit.tf", wires=4)
@qml.qnode(dev, interface='tf')
def cnot(inputs):
    qml.AngleEmbedding(inputs[:, ...], wires=range(4), rotation='Y')

    #qml.CNOT(wires=[0, 1])
    #qml.CNOT(wires=[2, 3])

    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

kernel_shape = [2,2]

kernel = Kernel2d4x4(kernel_shape)

kernel.fit(dataset)

data_blocks = kernel.transform(dataset)[:,:,[0,3,2,1]]

#data_blocks = (245 * ((data_blocks+80)/(784)))

post_data = data_blocks.transpose((2,1,0))

post_data = post_data.transpose((2,1,0))

#post_data = post_data.reshape(20,14,14,4)

#post_data = np.asarray((((OperationLayer(cnot).pre_op(data_blocks)+1)*100)-160)) // 1

post_data = kernel.post_transform(post_data)[:,:,:,[0,3,2,1]]

print((post_data)[0])
