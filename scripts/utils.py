import json
from pennylane import numpy as np
import pennylane as qml

from Filters import Filters

kernel = np.random.rand(2, 2) # we can create a class including the different type of kernels, if required

def choose_samples(train_images, train_labels, test_images, test_labels, reduction_ratio):
    """
    Choose samples from the data based on specified ratios for training and testing sets.

    Parameters:
        train_images (numpy.ndarray): Training images.
        train_labels (numpy.ndarray): Corresponding labels for training images.
        test_images (numpy.ndarray): Testing images.
        test_labels (numpy.ndarray): Corresponding labels for testing images.
        train_ratio (float): Ratio of samples to keep for training set (0 to 1).
        test_ratio (float): Ratio of samples to keep for testing set (0 to 1).

    Returns:
        tuple: Tuple containing the selected samples for training and testing sets.
    """
    # Determine number of samples to keep
    n_train = int(len(train_images) * reduction_ratio)
    n_test = int(len(test_images) * reduction_ratio)
    print(n_train)

    # Select samples based on ratios
    train_images_selected = train_images[:n_train]
    train_labels_selected = train_labels[:n_train]
    test_images_selected = test_images[:n_test]
    test_labels_selected = test_labels[:n_test]

    return train_images_selected, train_labels_selected, test_images_selected, test_labels_selected

def read_configurations(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

## apply the selected filter on an image
def filter(image, type, n_channels, n_layers):
    f = Filters(image, n_channels, n_layers)
    if type == 0:
        filtered_image  = f.geometry_filter()
    elif type == 1:
        filtered_image = f.convolution_filter(kernel=kernel)
    elif type == 2:
        filtered_image = f.quantum_conv_filter("random_layer")
    elif type == 3:
        filtered_image = f.quantum_conv_filter("cnot")
    elif type == 4:
        filtered_image = f.convolution_filter(kernel=kernel, pooling=False)
    return filtered_image

## apply the filters for all images
def apply_filter(images, type, n_channels, n_layers):
    filtered_images = []
    print("fitlered pre-processing of images, with type = {}".format(type))
    
    for idx, img in enumerate(images):
        print("{}/{}        ".format(idx + 1, np.shape(images)[0]), end="\r")
        filtered_images.append(filter(img, type, n_channels, n_layers))
    
    filtered_images = np.asarray(filtered_images)

    return(filtered_images)

def load_data(image_file):
        """Loads the training data from .npy files."""
        train_images = np.load(image_file)
        return train_images