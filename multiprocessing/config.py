
from utils import read_configurations

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




