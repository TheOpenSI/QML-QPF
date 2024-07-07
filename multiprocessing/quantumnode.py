import pennylane as qml
from pennylane import numpy as np 

class QuantumNodes():

    def __init__(self, layer: int) :
        self.layer = layer

    @qml.qnode(dev, interface='tf')
    def qentcnot_node(inputs):
        inputs *= np.pi

        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])

        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

