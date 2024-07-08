
from tensorflow import keras
import pennylane as qml
import config as cf
import quantumlayer as ql
import quantumnode as qn



class model():

    def __init__(self, data, filter, clock_start, workdir):
        self.data = data
        self.filter = filter
        self.model_name = cf.datasets[self.data]+cf.filters[self.filter]
        self.log_dir = workdir + "/" + clock_start + "/runs/" + self.model_name

        self.tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=0,
            write_graph=True
            )

        self.set_data()
        self.prep()
        self.fit()

    def set_data(self):
        if self.data == 1:
            self.mnist_dataset = keras.datasets.fashion_mnist
        else:
            self.mnist_dataset = keras.datasets.mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.mnist_dataset.load_data()

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
        self.q_model = self.Q_Model()
        self.pre_train_images = self.pre_model.predict(self.train_images,batch_size=cf.n_batches)
        self.pre_test_images = self.pre_model.predict(self.test_images,batch_size=cf.n_batches)



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
        self.q_history = self.q_model.fit(
            self.pre_train_images,
            self.train_labels,
            validation_data=(self.pre_test_images, self.test_labels),
            batch_size = cf.n_batches,
            epochs=cf.n_epochs,
            verbose=2,
            callbacks=[self.tensorboard_callback]
        )

