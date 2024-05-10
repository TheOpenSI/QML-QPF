import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
        
    def prediction(self, my_model, test_images, test_labels):
        """Use this function to make prediction and return the data ready for generation of confusion matrix"""
        
        predicted = my_model.predict(test_images)
        actual = tf.stack(test_labels, axis=0)
        predicted = tf.concat(predicted, axis=0)
        predicted = tf.argmax(predicted, axis=1)
        labels = np.unique(test_labels)

        return actual, predicted, labels
        
        
    def plot_confusion_matrix(self, actual, predicted, labels, ds_type):
        """
        This function  is used to plot the confusion matrix 
        """
        cm = tf.math.confusion_matrix(actual, predicted)
        ax = sns.heatmap(cm, annot=True, fmt='g')
        sns.set(rc={'figure.figsize':(12, 12)})
        sns.set(font_scale=1.4)
        ax.set_title('Confusion matrix of Handwritten MNIST data set ' + ds_type)
        ax.set_xlabel('Predicted Action')
        ax.set_ylabel('Actual Action')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)