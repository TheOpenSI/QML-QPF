import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers

import math
# from quantum_circuit import Quantum


class Filters:
    rand_params = np.random.uniform(high=2 * np.pi, size=(1, 4))
    stride = 2

    def __init__(self, image, n_channels, n_layers):
        # Should add a config file
        self.image = image
        self.n_channels = n_channels
        self.n_layers = n_layers
        # Initialize a default.qubit device
        self.dev = qml.device("default.qubit", wires=4)

        return

    def geometry_filter(self):
        """Convolves the input image with by taking the trace of the rotation matrix applied for the image pixel"""

        # Get dimensions
        image_height, image_width = self.image.shape[0], self.image.shape[1]

        # modify to incorporate padding
        out = np.zeros((image_height // 2, image_width // 2, self.n_channels))
        # Loop over the coordinates of the top-left pixel of 2X2 squares
        for j in range(0, image_height, self.stride):
            for k in range(0, image_width, self.stride):
                # Process a squared 2x2 region of the image with a rotation matix and take the trace
                results = []
                for pixel in [self.image[j, k, 0], self.image[j, k + 1, 0], self.image[j + 1, k, 0], self.image[j + 1, k + 1, 0]]:
                    results.append(np.trace(self.rotation_y(pixel)))
                    # results.append(np.linalg.det(rotation_y(pixel)))
                    # results.append(pixel*np.linalg.det(rotation_y(pixel)))
                    # results.append(pixel*np.trace(rotation_y(pixel)))

                # print(results)
                # out[j // 2, k // 2, 0] = np.sum(results)
                # # Assign expectation values to different channels of the output pixel (j/2, k/2)
                for c in range(self.n_channels):
                    out[j // 2, k // 2, c] = results[c]

        return out

    def rotation_y(self, angle):
        cossine_angle = math.cos(np.pi*angle)
        sine_angle = math.sin(np.pi*angle)
        return (np.array([[cossine_angle, -sine_angle], [sine_angle, cossine_angle]]))

    # Function to perform classical convolution
    def convolution_filter(self, kernel, pooling=True, pool_size=2):
        # Get dimensions
        image_height, image_width = self.image.shape[0], self.image.shape[1]

        kernel_height, kernel_width = kernel.shape

        # Calculate output dimensions
        output_height = image_height - kernel_height + 1
        output_width = image_width - kernel_width + 1

        # Initialize output image
        output = np.zeros((output_height, output_width))

        # Perform convolution
        for y in range(output_height):
            for x in range(output_width):
                output[y, x] = np.sum(
                    self.image[y:y+kernel_height, x:x+kernel_width] * kernel)

        # perform max pooling
        if pooling:
            # Calculate output dimensions
            output_height = (image_height + pool_size - 1) // pool_size
            output_width = (image_width + pool_size - 1) // pool_size

            # Initialize output image
            output = np.zeros((output_height, output_width))

            # Perform max pooling
            for y in range(0, image_height, pool_size):
                for x in range(0, image_width, pool_size):
                    output[y//pool_size, x//pool_size] = np.max(
                        self.image[y:min(y+pool_size, image_height), x:min(x+pool_size, image_width)])

        return output

    def quantum_conv_filter(self, q_type, meas_last=False):
        image_height, image_width = self.image.shape[0], self.image.shape[1]

        # measure only the last qubit (useful if all qubits are entangled)
        if meas_last is True:
            self.n_channels = 1
        else:
            self.n_channels = 4

        out = np.zeros((image_height // 2, image_width // 2, self.n_channels))
        self.dev = qml.device('default.qubit', wires=4)

        for j in range(0, image_height, self.stride):
            for k in range(0, image_width, self.stride):
                q_results = self.circuit([self.image[j, k, 0],
                                          self.image[j + 1, k, 0],
                                          self.image[j, k + 1, 0],
                                          self.image[j + 1, k + 1, 0]],
                                         q_type,
                                         meas_last
                                         )

                for c in range(len(q_results)):
                    out[j // self.stride, k // self.stride, c] = q_results[c]
        return out

    def circuit(self, phi, q_type, meas_last):
        n_qubits = len(phi)

        @qml.qnode(self.dev)
        def qnode():
            for j in range(n_qubits):
                qml.RY(np.pi * phi[j], wires=j)

            if q_type == "random_layer":
                qml.templates.RandomLayers(
                    Filters.rand_params, wires=list(range(n_qubits)))

            # Filter from arxiv.org/abs/2308.14930
            elif q_type == "cnot":
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[0, 3])

            # filter with the missing CNOT gate added to the above filter to create full entanglement
            elif q_type == "full":
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[0, 3])
                qml.CNOT(wires=[3, 1])

            # filter with full entanglement using different permutation of CNOT gates than the above filter
            elif q_type == "full_asc":
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 3])

            elif q_type == "cz":
                qml.CZ(wires=[0, 1])
                qml.CZ(wires=[1, 2])
                qml.CZ(wires=[2, 3])

            if meas_last is True:    # measure only the last qubit if meas_last is set to True
                return [qml.expval(qml.PauliZ(n_qubits-1))]
            else:
                return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

        # print(qml.draw(qnode)())
        return qnode()

    # A generic filter that works for any given stride and kernel size

    def generic_quantum_conv_filter(self, kernel_size, meas_last=True):
        # Find padding needed

        # Determine output size
        image_height, image_width = self.image.shape[0], self.image.shape[1]
        output_height = (image_height - kernel_size)//self.stride + 1
        output_width = (image_width - kernel_size)//self.stride + 1

        self.n_channels = kernel_size**2
        self.dev = qml.device("default.qubit", wires=self.n_channels)

        if meas_last is True:
            out = np.zeros((output_height, output_width, 1))
        else:
            out = np.zeros((output_height, output_width, self.n_channels))

        # Apply filter - use inner loop to fetch kernel_size blocks from input
        for j in range(0, image_height-kernel_size+1, self.stride):
            for k in range(0, image_width-kernel_size+1, self.stride):
                q_results = self.fully_entangled_circuit(self.image[j:j+kernel_size, k:k+kernel_size, 0].flatten(),
                                                         meas_last
                                                         )

                for c in range(len(q_results)):
                    out[j // self.stride, k // self.stride, c] = q_results[c]
        return out

    def fully_entangled_circuit(self, phi, meas_last):
        n_qubits = len(phi)

        @qml.qnode(self.dev)
        def qnode():
            for j in range(n_qubits):
                qml.RY(np.pi * phi[j], wires=j)

            for i in range(n_qubits-1):
                qml.CNOT(wires=[i, i+1])

            if meas_last is True:
                return [qml.expval(qml.PauliZ(n_qubits - 1))]
            else:
                return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

        # print(qml.draw(qnode)())
        return qnode()
