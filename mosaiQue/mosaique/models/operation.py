# operation.py
# This module defines custom Keras layers that integrate quantum computations using PennyLane.
# Author: Brian Recktenwall-Calvet
# Date: 12-18-2024
# Version: 1.0

import tensorflow as tf
from PIL.ImageChops import offset
from tensorflow import keras
import pennylane as qml

class QuantumLayer(keras.layers.Layer):
    """
    A custom Keras layer that integrates a PennyLane QNode for quantum computations.

    Attributes:
        _q_node (qml.QNode): The quantum node to execute quantum computations.
    """
    _q_node: qml.QNode

    @property
    def q_node(self) -> qml.QNode:
        """
        Getter for the QNode.

        Returns:
            qml.QNode: The QNode associated with this layer.
        """
        return self._q_node

    @q_node.setter
    def q_node(self, value):
        """
        Setter for the QNode.

        Args:
            value (qml.QNode): The QNode to set.
        """
        self._q_node = value

    def __init__(self, *args, **kwargs):
        """
        Initializes the quantum layer.

        Args:
            *args: Positional arguments for the parent class.
            **kwargs: Keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        """
        Executes the quantum layer on the input tensor.

        Args:
            inputs (tf.Tensor): The input tensor to process.

        Returns:
            tf.Tensor: The processed tensor after applying the QNode.
        """
        # The function assumes a batch size is set to use tf.vectorized_map
        return tf.stack(tf.vectorized_map(self.q_node, inputs), axis=1) - 1


class OperationLayer():
    """
    A class to manage a quantum layer and apply a kernel operation.

    Attributes:
        _q_layer (QuantumLayer): The quantum layer for processing inputs.
        _kernel_operation (qml.QNode): The QNode for the kernel operation.
    """
    _q_layer: QuantumLayer
    _kernel_operation: qml.QNode

    @property
    def q_layer(self) -> QuantumLayer:
        """
        Getter for the quantum layer.

        Returns:
            QuantumLayer: The quantum layer associated with this operation.
        """
        return self._q_layer

    @q_layer.setter
    def q_layer(self, value: QuantumLayer):
        """
        Setter for the quantum layer.

        Args:
            value (QuantumLayer): The quantum layer to set.
        """
        self._q_layer = value

    @property
    def kernel_operation(self) -> qml.QNode:
        """
        Getter for the kernel operation.

        Returns:
            qml.QNode: The QNode for the kernel operation.
        """
        return self._kernel_operation

    @kernel_operation.setter
    def kernel_operation(self, value: qml.QNode):
        """
        Setter for the kernel operation.

        Args:
            value (qml.QNode): The QNode to set as the kernel operation.
        """
        self._kernel_operation = value

    def __init__(self, kernel_operation):
        """
        Initializes the OperationLayer.

        Args:
            kernel_operation (qml.QNode): The kernel operation to use.
        """
        self.kernel_operation = kernel_operation
        self.q_layer = QuantumLayer()
        self.q_layer.q_node = self.kernel_operation

    @property
    def pre_op(self) -> keras.Model:
        """
        Initializes and returns a custom Keras model used to preprocess images.

        Returns:
            keras.Model: A compiled Keras model for image preprocessing.
        """
        this_model = keras.models.Sequential([
            keras.layers.Rescaling(scale=1. / 255.0),  # Normalize pixel values
            self.q_layer,  # Apply the quantum layer
            keras.layers.Rescaling(scale=-127.5)  # Scale the output
        ])
        this_model.compile(
            optimizer='adam',
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return this_model
