
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers


class Quantum():

    def __init__(self, n_channels, n_layers):
        # Should add a config file
        self.n_channels = n_channels
        self.n_layers = n_layers
        
        # Random circuit parameters
        self.rand_params = np.random.uniform(high=2 * np.pi, size=(self.n_layers, self.n_channels))

        # Define dev as an attribute (but don't initialize it yet)
        self.dev = None
    
        
    
       

    
    def circuit(self, phi):
        dev = qml.device("default.qubit", wires=self.n_channels)
        
        # Create a QNode using the initialized device
        @qml.qnode(dev)
        def qnode(phi):
            # Encoding of 4 classical input values
            for j in range(self.n_channels):
                qml.RY(np.pi * phi[j], wires=j)

            # Random quantum circuit
            RandomLayers(self.rand_params, wires=list(range(self.n_channels)))
            return [qml.expval(qml.PauliZ(j)) for j in range(self.n_channels)]

        # Measurement producing 4 classical output values
        return qnode(phi)