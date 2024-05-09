# import matplotlib.pyplot as plt
import json
from Filters import Filters
from Data_load import data_load

from Model import Model

from utils import *

import time

# Import the filtered images
data = data_load()
train_images, train_labels, test_images, test_labels = data.data_mnist()

types = {
    5: "full",
    6: "full_asc"
}


for t, type in types.items():
    # Instantiate a Model class
    my_model = Model(n_classes=data.n_classes)  # 10 for mnist dataset)
    # Compile the model
    my_model.compile_model()

    train_images = load_data(
        data.SAVE_PATH + "filtered_train_images_{}.npy".format(type))
    test_images = load_data(
        data.SAVE_PATH + "filtered_test_images_{}.npy".format(type))

    # getting only the last channel to feed to the NN
    train_images = train_images[:,:,:,-1]
    test_images = test_images[:,:,:,-1]

    start_time = time.time()
    # Train and validate the model (passing train_images and train_labels, test_images, test_labels)
    model_history = my_model.train_model(train_images,
                                         train_labels,
                                         test_images,
                                         test_labels,
                                         batch_size=data.batch_size,
                                         epochs=data.n_epochs)
    end_time = time.time()
    time_taken = end_time - start_time
    # Save time taken to a text file
    with open(data.SAVE_PATH + "time_for_{}_last_channel.txt".format(type), "a") as file:
        file.write(
            "\nTime taken for training model using last channel: {} seconds".format(time_taken))
    # save model history
    with open(data.SAVE_PATH + "model_{}_last_channel.json".format(type), 'w') as json_file:
        json.dump(model_history.history, json_file)


print(f'Experiment results saved {data.SAVE_PATH}')
