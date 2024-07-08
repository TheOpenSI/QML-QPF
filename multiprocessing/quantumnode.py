import pennylane as qml
from pennylane import numpy as np 
from pennylane.templates import RandomLayers
import config as cf

def get_qcnot_node(dev):
    @qml.qnode(dev, interface='tf')
    def qcnot_node(inputs):
        inputs *= np.pi
        # Encoding of 4 classical input values
        #Further testing of the AngleEmbedding function is needed
        qml.AngleEmbedding(inputs, wires=range(cf.n_qubits), rotation='Y')
        # Filter from arxiv.org/abs/2308.14930

        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[0, 3])

        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(cf.n_qubits)]
    return qcnot_node

def get_qentcnot_node(dev):
    @qml.qnode(dev, interface='tf')
    def qentcnot_node(inputs):
        inputs *= np.pi

        qml.AngleEmbedding(inputs, wires=range(cf.n_qubits), rotation='Y')

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])

        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(cf.n_qubits)]
    return qentcnot_node

def get_qrand_node(dev):
    @qml.qnode(dev, interface='tf')
    def qrand_node(inputs):
        inputs *= np.pi
        # Encoding of 4 classical input values
        #Further testing of the AngleEmbedding function is needed
        qml.AngleEmbedding(inputs, wires=range(cf.n_qubits), rotation='Y')
        # Filter from arxiv.org/abs/2308.14930

        RandomLayers(cf.rand_params, wires=list(range(cf.n_qubits)))

        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(cf.n_qubits)]
    return qrand_node

