import pennylane as qml
import tensorflow as tf
from tensorflow import keras

import mosaique.config as cf
import mosaique.legacy.quantumnode as qn


@keras.utils.register_keras_serializable()
class QuantumLayer(keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        """Initialize quantum layer .
        """
        super().__init__(*args, **kwargs)
        dev = qml.device("default.qubit.tf", wires=cf.n_qubits)
        self.q_node = qn.get_qrand_node(dev)

    def prep_quantumlayer(self, q_node):
        """Set the quantum circuit being used in preprocessing .

        Args:
            q_node (function): quantum circuit selected for preprocessing
        """
        self.q_node = q_node

    def call(self, inputs):
        """preprocessing using the quantum network

        Args:
            inputs ([type]): [description]

        Returns:
            [type]: [description]
        """

        # 14x14 flattened 2x2 squares
        get_subsections_14x14 = lambda im: tf.reshape(tf.unstack(tf.reshape(im, [14, 2, 14, 2]), axis=2), [14, 14, 4])

        # unpack 14x14 row by row
        list_squares_2x2 = lambda image_subsections: tf.reshape(tf.unstack(image_subsections, axis=1), [196, 4])

        # send 4 values to quantum function
        process_square_2x2 = lambda square_2x2: self.q_node(square_2x2)

        # send all squares to the quantum function wrapper
        process_subsections = lambda squares: tf.vectorized_map(process_square_2x2, squares)

        # recompile the larger square
        separate_channels = lambda channel_stack: tf.reshape(channel_stack, [14, 14, 4])
        # each smaller square (channel) can be extracted as [:, :, channel]

        # apply function across batch
        preprocessing = lambda input: tf.vectorized_map(
            lambda image: (
                separate_channels(tf.transpose(process_subsections(list_squares_2x2(get_subsections_14x14(image)))))),
            input
        )

        return preprocessing(inputs) * -1


@keras.utils.register_keras_serializable()
class BasicLayer(keras.layers.Layer):

    def call(self, inputs):
        """Preprocessing without quantum network.
        splits input in to four channels

        Args:
            inputs ([type]): [description]

        Returns:
            [type]: [description]
        """

        rotate_14x14 = lambda subsection_14x14: tf.vectorized_map(lambda chanel: tf.transpose(chanel), subsection_14x14)

        get_subsections_14x14 = lambda im: tf.reshape(tf.unstack(tf.reshape(im, [14, 2, 14, 2]), axis=2), [14, 14, 4])

        preprocessing = lambda input: tf.vectorized_map(
            lambda image: (tf.transpose(rotate_14x14(tf.transpose(get_subsections_14x14(image))))),
            input
        )

        return preprocessing(inputs)
