import numpy as np
from Data_load import data_load

data = data_load()

types = {
    0: "geometrical",
    1: "classical_w_pooling",
    2: "quantum_random",
    3: "quantum_cnot",
    4: "classical_wo_pooling",
    5: "quantum_full",
    6: "quantum_full_asc",
    7: "quantum_no_cnot",
    8: "quantum_cz",
    9: "3x3_quantum_filter"
}

for t, type in types.items():
    filtered_imgs = np.load(
        data.SAVE_PATH + "filtered_train_images_{}.npy".format(type))
    print(type, filtered_imgs.shape)
