
import json
from pennylane import numpy as np 
import tensorflow as tf

np.random.seed(0)           # Seed for NumPy random number generator
tf.random.set_seed(0)       # Seed for TensorFlow random number generator

def read_configurations(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


config = read_configurations('./config.json')

n_epochs = config['EPOCHS']   # Number of optimization epochs
n_layers = config['RNDM_LAYERS']    # Number of random layers
n_train = config['TRAIN_IMAGES']    # Size of the train dataset
n_test = config['TEST_IMAGES']     # Size of the test dataset
n_channels = config['CHANNELS']  # Number of channels
is_reduced = config['DATA_REDUCTION']  # its 0 (false) or 1 (True)
reduction_ratio = config['REDUCTION_RATIO']  # its between 0 and 1

SAVE_PATH = config['SAVE_PATH']  # Data saving folder
# If False, skip quantum processing and load data from SAVE_PATH
PREPROCESS = config['PREPROCESS']

# represents number of classes in the data.
n_classes = config["CLASSES"]

# O means geometrical filter (rotation&trace), 1 means classical filter (conv&max_pooling), 2 means quantum (random), 3 means quantum (cnot)
FILTER_TYPE = config["FILTER_TYPE"]

batch_size = config["BATCH_SIZE"]
n_batches = config["BATCH_SIZE"]

rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))

n_qubits = 4

memory_limit = 3000

datasets = {
    0 : "MNIST",
    1 : "FASHIONMNIST"
}

filters = {
    0 : "no_filter",
    1 : "qrand",
    2 : "qcnot",
    3 : "qentcnot"
}


