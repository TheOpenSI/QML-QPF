#import matplotlib.pyplot as plt
import json
from Filters import Filters
from Data_load import data_load

from Model import Model

from utils import *

import time

## Import the images 
data = data_load()
train_images, train_labels, test_images, test_labels = data.data_mnist()

types ={
    0: "geometrical",
    1: "classical_w_pooling",
    2: "quantum_random",
    3: "quantum_cnot",
    4: "classical_wo_pooling"
}

if data.PREPROCESS == True:

    for t, type in types.items():
        start_time = time.time()
        filtered_train_images = apply_filter(train_images, t, data.n_channels, data.n_layers)
        filtered_test_images = apply_filter(test_images, t, data.n_channels, data.n_layers)
        # Save pre-processed images
        np.save(data.SAVE_PATH + "filtered_train_images_{}.npy".format(type), filtered_train_images)
        np.save(data.SAVE_PATH + "filtered_test_images_{}.npy".format(type), filtered_test_images)
        end_time = time.time()
        # Calculate the time taken
        time_taken = end_time - start_time
        # Save time taken to a text file
        with open(data.SAVE_PATH + "time_for_{}.txt".format(type), "w") as file:
            file.write("Time taken for the experiment: {} seconds".format(time_taken))


    

# Training !!!!
n_classes = data.n_classes # 10 for mnist dataset

for t, type in types.items():
    # Instantiate a Model class
    my_model = Model(n_classes = data.n_classes) # 10 for mnist dataset)
    # Compile the model
    my_model.compile_model()
    
    train_images = load_data(data.SAVE_PATH + "filtered_train_images_{}.npy".format(type))
    test_images = load_data(data.SAVE_PATH + "filtered_test_images_{}.npy".format(type))
    
    # in case if you need to choose chunks of the filtered data, this can be done below here
    # ........


    # Train and validate the model (passing train_images and train_labels, test_images, test_labels)
    model_history = my_model.train_model(train_images,
                                            train_labels,
                                            test_images,
                                            test_labels,
                                            batch_size=data.batch_size,
                                            epochs = data.n_epochs)
    # save model history
    with open(data.SAVE_PATH + "model_{}.json".format(type),'w') as json_file:
        json.dump(model_history.history, json_file)


print(f'Experiment results saved {data.SAVE_PATH}')