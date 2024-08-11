import pennylane as qml
from pennylane import numpy as np 
from pennylane.templates import RandomLayers
import config as cf



def get_ent_qcnot_node_variant(dev, cnot1_control, cnot1_target, cnot2_control, cnot2_target, cnot3_control, cnot3_target):
    @qml.qnode(dev, interface='tf')
    def qcnot_node(inputs):
        inputs *= np.pi
        # Encoding of 4 classical input values
        #Further testing of the AngleEmbedding function is needed
        qml.AngleEmbedding(inputs, wires=range(cf.n_qubits), rotation='Y')


        qml.CNOT(wires=[cnot1_control, cnot1_target])
        qml.CNOT(wires=[cnot2_control, cnot2_target])
        qml.CNOT(wires=[cnot3_control, cnot3_target])

        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(cf.n_qubits)]
    return qcnot_node

def get_qcnot_node_variant(dev, cnot1_control, cnot1_target, cnot2_control, cnot2_target):
    @qml.qnode(dev, interface='tf')
    def qcnot_node(inputs):
        inputs *= np.pi
        # Encoding of 4 classical input values
        #Further testing of the AngleEmbedding function is needed
        qml.AngleEmbedding(inputs, wires=range(cf.n_qubits), rotation='Y')
        # Filter from arxiv.org/abs/2308.14930

        qml.CNOT(wires=[cnot1_control, cnot1_target])
        qml.CNOT(wires=[cnot2_control, cnot2_target])

        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(cf.n_qubits)]
    return qcnot_node


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

def get_qrand_node_variant(dev, seed):
    @qml.qnode(dev, interface='tf')
    def qrand_node(inputs):
        inputs *= np.pi
        # Encoding of 4 classical input values
        #Further testing of the AngleEmbedding function is needed
        qml.AngleEmbedding(inputs, wires=range(cf.n_qubits), rotation='Y')
        # Filter from arxiv.org/abs/2308.14930
        [qml.Hadamard(j) for j in range(cf.n_qubits)]
        RandomLayers(cf.rand_params, wires=range(cf.n_qubits), ratio_imprim=0.5, imprimitive=qml.CH, seed=seed)
        RandomLayers(cf.rand_params, wires=range(cf.n_qubits), ratio_imprim=0.5, imprimitive=qml.CNOT, seed=seed)
        RandomLayers(cf.rand_params, wires=range(cf.n_qubits), ratio_imprim=0.5, imprimitive=qml.CH, seed=seed)
        [qml.Hadamard(j) for j in range(cf.n_qubits)]
        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(cf.n_qubits)]
    return qrand_node

