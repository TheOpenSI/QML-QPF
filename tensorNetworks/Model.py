import tensorflow as tf
from tensorflow import keras

class Model:
    def __init__(self, n_classes):
        """Initializes the Model class."""
        self.n_classes = n_classes
        self.model = self.create_model()

    def create_model(self):
        """Creates and returns a custom Keras model."""
        model = keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(self.n_classes, activation="softmax")
        ])
        return model

    def compile_model(self):
        """Compiles the Keras model."""
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train_model(self, train_images, train_labels, test_images, test_labels, batch_size, epochs=10):
        """Trains the Keras model."""
        return self.model.fit(train_images, 
                       train_labels, 
                       validation_data=(test_images, test_labels),
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=2)