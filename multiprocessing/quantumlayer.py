
import tensorflow as tf
from tensorflow import keras
import pennylane as qml
import config as cf
from quantumnode import QuantumNodes

class QuantumLayers():
    def __init__(self) -> None:
        self.config = cf
        self.qentcnotlayer = QuantumLayers.QuantumLayer(self, QuantumNodes.qentcnot_node())

    
    class QuantumLayer(keras.layers.Layer):

        def __init__(self, container, q_node):
            self.config = container.config
            self.config.dev = qml.device("default.qubit.tf", wires=self.config.n_qubits)
            self.q_node = q_node


        def call(self, inputs):

            #14x14 flattened 2x2 squares
            get_subsections_14x14 = lambda im : tf.reshape(tf.unstack(tf.reshape(im,[14,2,14,2]), axis = 2),[14,14,4])

            #unpack 14x14 row by row
            list_squares_2x2 = lambda image_subsections: tf.reshape(tf.unstack(image_subsections, axis = 1), [196,4])


            #send 4 values to quantum function
            process_square_2x2 = lambda square_2x2 : self.q_node(square_2x2)

            #send all squares to the quantum function wrapper
            process_subsections = lambda squares: tf.vectorized_map(process_square_2x2,squares)

            #recompile the larger square
            separate_channels = lambda channel_stack: tf.reshape(channel_stack, [14,14,4])
            #each smaller square (channel) can be extracted as [:, :, channel]
            
            #apply function across batch
            preprocessing = lambda input: tf.vectorized_map(
                lambda image:(separate_channels(tf.transpose(process_subsections(list_squares_2x2(get_subsections_14x14(image)))))),
                input
            )

            return preprocessing(inputs)*-1