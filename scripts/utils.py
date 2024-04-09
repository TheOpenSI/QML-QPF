
import json

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