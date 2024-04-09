from pennylane import numpy as np
import pennylane as qml
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import json
from Filters import Filters
from Data_load import data_load

## Initialisation 

# n_epochs = 30   # Number of optimization epochs
# n_layers = 1    # Number of random layers
# n_train = 50    # Size of the train dataset
# n_test = 30     # Size of the test dataset
# n_channels = 4 # Number of channels

# SAVE_PATH = "../results/"  # Data saving folder
# PREPROCESS = True           # If False, skip quantum processing and load data from SAVE_PATH
# np.random.seed(0)           # Seed for NumPy random number generator
# tf.random.set_seed(0)       # Seed for TensorFlow random number generator

# ## Load the data set
# mnist_dataset = keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

# # Reduce dataset size
# train_images = train_images[:n_train]
# train_labels = train_labels[:n_train]
# test_images = test_images[:n_test]
# test_labels = test_labels[:n_test]

# # Normalize pixel values within 0 and 1
# train_images = train_images / 255
# test_images = test_images / 255


# # Add extra dimension for convolution channels
# train_images = np.array(train_images[..., tf.newaxis], requires_grad=False)
# test_images = np.array(test_images[..., tf.newaxis], requires_grad=False)

## the jupyther notebook always has a better result for the quantum filter (moving the whole quantum circuit to mains experiment script)

data = data_load()

train_images, train_labels, test_images, test_labels = data.data_mnist()

print(np.shape(train_images))

print(n)

dev = qml.device("default.qubit", wires=n_channels)
# Random circuit parameters
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, n_channels))

@qml.qnode(dev)
def circuit(phi):
    # Encoding of 4 classical input values
    for j in range(4):
        qml.RY(np.pi * phi[j], wires=j)

    # Random quantum circuit
    RandomLayers(rand_params, wires=list(range(n_channels)))

    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

def quanv(image):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = np.zeros((14, 14, n_channels))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = circuit(
                [
                    image[j, k, 0],
                    image[j, k + 1, 0],
                    image[j + 1, k, 0],
                    image[j + 1, k + 1, 0]
                ]
            )
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            for c in range(4):
                out[j // 2, k // 2, c] = q_results[c]
    return out


## apply the filters
filter = Filters()
def apply_filter(image, type ):
    filtered_images = []
    print("fitlered pre-processing of images:")
    for idx, img in enumerate(image):
        print("{}/{}        ".format(idx + 1, np.shape(image)[0]), end="\r")
        if type == 0:
            filtered_images.append(filter.classic_filter(image=img, n_channels=n_channels))
        elif type == 1:
            # filtered_images.append(filter.quanv(image=img, n_channels=n_channels, n_layers=n_layers))
            filtered_images.append(quanv(img))
    filtered_images = np.asarray(filtered_images)

    return(filtered_images)

 # apply the classical filter   

if PREPROCESS == True:
    filtered_train_images =  apply_filter(train_images, type=0)
    filtered_test_images = apply_filter(test_images, type=0)
# Save pre-processed images
    np.save(SAVE_PATH + "filtered_train_images.npy", filtered_train_images)
    np.save(SAVE_PATH + "filtered_test_images.npy", filtered_test_images)

 # apply the quantum filter   
    q_train_images = apply_filter(train_images, type=1)
    q_test_images = apply_filter(test_images, type=1)
    np.save(SAVE_PATH + "q_train_images.npy", q_train_images)
    np.save(SAVE_PATH + "q_test_images.npy", q_test_images)


def MyModel():
    """Initializes and returns a custom Keras model
    which is ready to be trained."""
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

classical_filtered_model = MyModel()
classical_filtered_history = classical_filtered_model.fit(
    filtered_train_images,
    train_labels,
    validation_data=(filtered_test_images, test_labels),
    batch_size=4,
    epochs=n_epochs,
    verbose=2,
)

classical_model = MyModel()
classical_history = classical_model.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    batch_size=4,
    epochs=n_epochs,
    verbose=2,
)

q_model = MyModel()

q_history = q_model.fit(
    q_train_images,
    train_labels,
    validation_data=(q_test_images, test_labels),
    batch_size=4,
    epochs=n_epochs,
    verbose=2,
)

#save the results as Json file
with open(SAVE_PATH + 'q_history.json','w') as json_file:
    json.dump(q_history.history, json_file)

with open(SAVE_PATH + 'classical_filtered_history.json','w') as json_file:
    json.dump(classical_filtered_history.history, json_file)

with open(SAVE_PATH + 'classical_history.json','w') as json_file:
    json.dump(classical_history.history, json_file)

print(f'Experiment results saved {SAVE_PATH}')