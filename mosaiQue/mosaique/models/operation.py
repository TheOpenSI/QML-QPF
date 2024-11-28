import tensorflow as tf
from PIL.ImageChops import offset
from tensorflow import keras
import pennylane as qml

class QuantumLayer(keras.layers.Layer):
    _q_node: qml.QNode
    @property
    def q_node(self) -> qml.QNode:
        return self._q_node
    @q_node.setter
    def q_node(self, value):
        self._q_node = value

    def __init__(self, *args, **kwargs):
        """Initialize quantum layer .
        """
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        # does not work without setting a batch size
        return tf.stack(tf.vectorized_map(self.q_node, inputs),axis=1) - 1


class OperationLayer():
    _q_layer: QuantumLayer
    _kernel_operation: qml.QNode
    @property
    def q_layer(self) -> QuantumLayer:
        return self._q_layer
    @q_layer.setter
    def q_layer(self, value:QuantumLayer):
        self._q_layer = value
    @property
    def kernel_operation(self) -> qml.QNode:
        return self._kernel_operation
    @kernel_operation.setter
    def kernel_operation(self, value: qml.QNode):
        self._kernel_operation = value

    def __init__(self, kernel_operation):
        self.kernel_operation = kernel_operation
        self.q_layer = QuantumLayer()
        self.q_layer.q_node = self.kernel_operation

    @property
    def pre_op(self) -> keras.Model:
        """Initializes and returns a custom keras models used to preprocess images .

        Returns:
            keras.Model: [description]
        """
        this_model = keras.models.Sequential([
            keras.layers.Rescaling(scale=1. / 255.0),
            self.q_layer,
            keras.layers.Rescaling(scale=-127.5)
        ])
        this_model.compile(
            optimizer='adam',
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return this_model
