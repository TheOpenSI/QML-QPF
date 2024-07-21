
from tensorflow import keras
import pennylane as qml
from pennylane import numpy as np 
import config as cf
import quantumlayer as ql
import quantumnode as qn
import os



class Model():

    def __init__(self, data, filter, clock_start, workdir):
        workdir = workdir + "/output"
        self.data = data
        self.filter = filter
        self.model_name = cf.datasets[self.data]+cf.filters[self.filter]
        self.workdir = workdir + "/output"
        self.log_dir = workdir + "/" + clock_start + "/runs/" + cf.datasets[self.data] + "/" + self.model_name
        self.data_dir = workdir + "/" + clock_start + "/data/" + cf.datasets[self.data] + "/" + self.model_name
        self.model_dir = self.log_dir + "/train/" 
        self.visuals_dir = workdir + "/" + clock_start + "/visuals/" 
        if data == 0:
            self.class_labels = ["Zero","One","Two","Three", "Four",
                "Five", "Six", "Seven", "Eight", "Nine"]
        else:
            self.class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            write_steps_per_second=True,
            update_freq='batch',
            profile_batch=1,
            embeddings_freq=1,
            embeddings_metadata=None
        )

    
    def restore(self):
        #keras_embedding.ckpt-29.weights.h5
        self.pre_train_images = np.load(os.path.join(self.data_dir, 'filtered_train_images.npy'))
        self.pre_test_images = np.load(os.path.join(self.data_dir, 'filtered_test_images.npy'))
        self.q_model = self.Q_Model()
        self.q_model.predict(self.pre_test_images)
        self.q_model.load_weights(self.model_dir + f'keras_embedding.ckpt-{cf.n_epochs-1}.weights.h5')

    def load_history(self):
        import numpy as np
        bias = []
        weights = []
        self.restore()
        for num in range(cf.n_epochs):
            file = self.model_dir  + f'keras_embedding.ckpt-{num}.weights.h5'
            self.q_model.load_weights(file)
            layer = self.q_model.get_layer(index=2)
            extract = layer.get_weights()
            weights += [np.asarray(extract[0])]
            bias += [np.asarray(extract[1])]
        self.bias = np.asarray(bias)
        self.weights = np.asarray(weights)
        

    

    def save_filtered(self):
        os.makedirs(self.data_dir)
        np.save(os.path.join(self.data_dir, 'filtered_train_images'), self.pre_train_images)
        np.save(os.path.join(self.data_dir, 'filtered_test_images'), self.pre_test_images)

    
    def load_filtered(self):
        self.pre_train_images = np.load(os.path.join(self.data_dir, 'filtered_train_images.npy'))
        self.pre_test_images = np.load(os.path.join(self.data_dir, 'filtered_test_images.npy'))


    def set_data(self):
        if self.data == 1:
            self.mnist_dataset = keras.datasets.fashion_mnist
        else:
            self.mnist_dataset = keras.datasets.mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.mnist_dataset.load_data()

    def import_data(self, train_images, train_labels, test_images,test_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

    def prep(self):
        if self.filter == 0:
            self.qlayer = ql.BasicLayer()
        else:
            dev = qml.device("default.qubit.tf", wires=cf.n_qubits)
            if self.filter == 3:
                self.qnode = qn.get_qentcnot_node(dev)

            elif self.filter == 2:
                self.qnode = qn.get_qcnot_node(dev)

            else:
                self.qnode = qn.get_qrand_node(dev)

            self.qlayer = ql.QuantumLayer()
            self.qlayer.prep_quantumlayer(self.qnode)

            
        self.pre_model = self.Pre_Model()
    
    def pre_filter(self):
        self.pre_train_images = self.pre_model.predict(self.train_images,batch_size=cf.n_batches)
        self.pre_test_images = self.pre_model.predict(self.test_images,batch_size=cf.n_batches)
    
    @property
    @keras.utils.register_keras_serializable()
    def flatten(self):
        flattening = keras.models.Sequential([
            keras.layers.Rescaling(scale=1./127.5, offset=-1),
            keras.layers.Flatten()
        ])
        flattening.compile(
            optimizer='adam',
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return flattening


    @keras.utils.register_keras_serializable()
    def Pre_Model(self):
        """Initializes and returns a custom Keras model
        which is ready to be trained."""
        if self.filter == 0:
            this_model = keras.models.Sequential([
                self.qlayer
            ])
        else:
            this_model = keras.models.Sequential([
                keras.layers.Rescaling(scale=1./255.0),
                self.qlayer,
                keras.layers.Rescaling(scale=127.5, offset=127.5)
            ])                
        this_model.compile(
            optimizer='adam',
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return this_model
    

    #core model

    @keras.utils.register_keras_serializable()
    def Q_Model(self):
        """Initializes and returns a custom Keras model
        which is ready to be trained."""
        this_model = keras.models.Sequential([
            keras.layers.Rescaling(scale=1./127.5, offset=-1),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation="softmax")
        ])
        this_model.compile(
            optimizer='adam',
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return this_model
    def fit(self):
        self.q_model = self.Q_Model()
        self.q_history = self.q_model.fit(
            self.pre_train_images,
            self.train_labels,
            validation_data=(self.pre_test_images, self.test_labels),
            batch_size = cf.n_batches,
            epochs=cf.n_epochs,
            verbose=2,
            callbacks=[self.tensorboard_callback]
        )
    @property
    def visuals(self):
        from visualizations import Visualize
        this_visuals = Visualize(self)
        return this_visuals