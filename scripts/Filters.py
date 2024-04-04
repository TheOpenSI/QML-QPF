import pennylane as qml
from pennylane import numpy as np
import math
# from quantum_circuit import Quantum

class Filters():

    def __init__(self,):
    # Should add a config file
  
        return
    
    def classic_filter(self, image, n_channels):
        """Convolves the input image with by taking the trace of the rotation matrix applied for the image pixel"""
        out = np.zeros((14, 14, n_channels ))
        # Loop over the coordinates of the top-left pixel of 2X2 squares
        for j in range(0, 28, 2):
            for k in range(0, 28, 2):
                # Process a squared 2x2 region of the image with a rotation matix and take the trace
                results = []
                for pixel in [image[j, k, 0], image[j, k + 1, 0], image[j + 1, k, 0],image[j + 1, k + 1, 0]]:
                    results.append(np.trace(self.rotation_y(pixel)))
                    # results.append(np.linalg.det(rotation_y(pixel)))
                    # results.append(pixel*np.linalg.det(rotation_y(pixel)))
                    # results.append(pixel*np.trace(rotation_y(pixel)))
    
                # print(results)
                # out[j // 2, k // 2, 0] = np.sum(results)
                # # Assign expectation values to different channels of the output pixel (j/2, k/2)
                for c in range(n_channels):
                    out[j // 2, k // 2, c] = results[c]
                    
        return out

    def rotation_y(self, angle):
        cossine_angle = math.cos(np.pi*angle)
        sine_angle = math.sin(np.pi*angle)
        return(np.array([[cossine_angle, -sine_angle ],[ sine_angle ,cossine_angle]]))
    
    # def quanv(self, image, n_channels, n_layers):
    #     """Convolves the input image with many applications of the same quantum circuit."""
    #     out = np.zeros((14, 14, n_channels))
    #     #set the quantum device
    #     device = qml.device("default.qubit", wires=n_channels)
    #     qmc = Quantum(n_channels, n_layers, device)

    #     # Loop over the coordinates of the top-left pixel of 2X2 squares
    #     for j in range(0, 28, 2):
    #         for k in range(0, 28, 2):
    #             # Process a squared 2x2 region of the image with a quantum circuit
    #             q_results = qmc.circuit(
    #                 [
    #                     image[j, k, 0],
    #                     image[j, k + 1, 0],
    #                     image[j + 1, k, 0],
    #                     image[j + 1, k + 1, 0]
    #                 ]
    #             )
    #             # Assign expectation values to different channels of the output pixel (j/2, k/2)
    #             for c in range(4):
    #                 out[j // 2, k // 2, c] = q_results[c]
    #     return out
    
    