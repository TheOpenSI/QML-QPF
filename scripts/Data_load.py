
from utils import read_configurations, choose_samples
from tensorflow import keras

class data_load():

    def __init__(self,):
        # Read configurations from config.json

        config = read_configurations('config.json')

        self.n_epochs = config['EPOCHS']   # Number of optimization epochs
        self.n_layers = config['RNDM_LAYERS']    # Number of random layers
        self.n_train = config['TRAIN_IMAGES']    # Size of the train dataset
        self.n_test = config['TEST_IMAGES']     # Size of the test dataset
        self.n_channels = config['CHANNELS'] # Number of channels
        self.is_reduced = config['DATA_REDUCTION']
        self.reduction_ratio = config['REDUCTION_RATIO']

        self.SAVE_PATH = config['SAVE_PATH']  # Data saving folder
        self.PREPROCESS = config['PREPROCESS']           # If False, skip quantum processing and load data from SAVE_PATH
            
           
        return
    
    def data_mnist(self,):

        ## Load the data set
        mnist_dataset = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

        if self.is_reduced:
            # Reduce dataset size
            train_images, train_labels, test_images, test_labels = choose_samples(train_images, train_labels, test_images, test_labels, self.reduction_ratio)

        # Normalize pixel values within 0 and 1
        train_images = train_images / 255
        test_images = test_images / 255

        return train_images, train_labels, test_images, test_labels

