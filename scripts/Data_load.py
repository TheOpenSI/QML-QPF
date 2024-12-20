
from utils import read_configurations, choose_samples
from tensorflow import keras
from pennylane import numpy as np
import tensorflow as tf


class data_load():

    def __init__(self,):
        # Read configurations from config.json

        config = read_configurations('./experiment_params.json')

        self.n_epochs = config['EPOCHS']   # Number of optimization epochs
        self.n_layers = config['RNDM_LAYERS']    # Number of random layers
        self.n_train = config['TRAIN_IMAGES']    # Size of the train dataset
        self.n_test = config['TEST_IMAGES']     # Size of the test dataset
        self.n_channels = config['CHANNELS']  # Number of channels
        self.is_reduced = config['DATA_REDUCTION']  # its 0 (false) or 1 (True)
        self.reduction_ratio = config['REDUCTION_RATIO']  # its between 0 and 1

        self.SAVE_PATH = config['SAVE_PATH']  # Data saving folder
        # If False, skip quantum processing and load data from SAVE_PATH
        self.PREPROCESS = config['PREPROCESS']

        # represents number of classes in the data.
        self.n_classes = config["CLASSES"]

        # O means geometrical filter (rotation&trace), 1 means classical filter (conv&max_pooling), 2 means quantum (random), 3 means quantum (cnot)
        self.FILTER_TYPE = config["FILTER_TYPE"]

        self.batch_size = config["BATCH_SIZE"]

        return

    def data_mnist(self,):

        # Load the data set
        mnist_dataset = keras.datasets.mnist
        (train_images, train_labels), (test_images,
                                       test_labels) = mnist_dataset.load_data()

        if self.is_reduced:
            # Reduce dataset size
            train_images, train_labels, test_images, test_labels = choose_samples(
                train_images, train_labels, test_images, test_labels, self.reduction_ratio)

        # Normalize pixel values within 0 and 1
        train_images = train_images / 255
        test_images = test_images / 255

        # Add extra dimension for convolution channels
        train_images = np.array(
            train_images[..., tf.newaxis], requires_grad=False)
        test_images = np.array(
            test_images[..., tf.newaxis], requires_grad=False)

        return train_images, train_labels, test_images, test_labels
