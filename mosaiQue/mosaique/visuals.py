import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, FuncAnimation, PillowWriter
from matplotlib.image import AxesImage
import numpy as np
import tensorflow as tf
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core import Model
import os
import config as cf
from typing import Callable

def reassembled(input: np.ndarray) -> np.ndarray:
    """Reassemble the input tensors .

    Args:
        input (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """    
    reassembled_pt1 = lambda square: tf.reshape(
        tf.vectorized_map(lambda inp: tf.transpose(inp),tf.transpose(square)),[2,14,2,14])
    reassemble = lambda square: tf.reshape(
        tf.stack(tf.reshape(tf.stack(reassembled_pt1(square),axis=2),[2,28*14]),axis=1),[28,28])
    return reassemble(input)

class Visualize:
    this: 'Model'

    def __init__(self,model:'Model'):
        global this
        this = model
        this.bias = this.bias.reshape([30,1,10])

    def individual(self, history: np.ndarray) -> tuple[plt.Figure, Callable[[int],list[AxesImage]]]:
        """scene builder displays classes in individual matrices .

        Args:
            history (np.ndarray): array of matrices, matrices are shaped like the preprocessed dataset

        Returns:
            tuple[plt.Figure, function]: pass to animation
        """        
        figure, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
        def animate(index:int) -> list[AxesImage]:
            index -= 5
            if index < 0:
                index = 0
            if index > (cf.n_epochs - 1):
                index = (cf.n_epochs - 1)
            weighted_scaled = np.reshape(np.asarray(history[index]).T,(10,14,14,4))
            scene = []
            for i, title in enumerate(this.class_labels[:9]):
                y = ((i) % 3)
                x = ((i) // 3)
                ax[x][y].clear()
                ax[x][y].set_title(title, fontsize=16)
                im = ax[x][y].imshow(reassembled(weighted_scaled [i,:,:,:]))
                scene += [im]
                ax[x][y].axis('off')
            return scene
        return figure, animate
    
    def relative(self, history: np.ndarray) -> tuple[plt.Figure, Callable[[int],list[AxesImage]]]:
        """scene builder combines classes in a single matrix to view the classes at relative scale .

        Args:
            history (np.ndarray): array of matrices, matrices are shaped like the preprocessed dataset

        Returns:
            tuple[plt.Figure, function]: pass to animation
        """        
        figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

        def animate(indx: int) -> list[AxesImage]:
            indx -= 5
            if indx < 0:
                indx = 0
            if indx > (cf.n_epochs - 1):
                indx = (cf.n_epochs - 1)
            weighted_scaled = np.reshape(np.asarray(history[indx]).T,(10,14,14,4))
            ims = []
            for i, title in enumerate(this.class_labels[:9]):

                ims += [reassembled(weighted_scaled [i,:,:,:])]
            final = np.vstack([np.block([[ims[0],ims[1],ims[2]]]),
                            np.block([[ims[3],ims[4],ims[5]]]),
                            np.block([[ims[6],ims[7],ims[8]]])])
            ax.clear()
            ax.axis('off')
            return [ax.imshow(final)]
        return figure, animate
    
    def animating(self, figure: plt.Figure, animate: Callable[[int],list[AxesImage]], name: str):
        """builds the animation .

        Args:
            figure (plt.Figure): figure to write to
            animate (function): scene builder to animate
            name (str): name for tittle and save file
        """        
        figure.suptitle(cf.datasets[this.data] + this.model_name + " " + name)
        visuals_dir = this.visuals_dir + name + "/"
        os.makedirs(visuals_dir, exist_ok=True)
        file = visuals_dir + cf.datasets[this.data] + this.model_name + ".gif"
        print(file)                
        ani = FuncAnimation(figure, animate, interval=50, blit=True, repeat=True, frames=(cf.n_epochs + 10), repeat_delay= 50)    
        ani.save(file, dpi=300, writer=PillowWriter(fps=1))
    
    def filtering(self):
        start = this.train_images[0]
        image = this.pre_train_images[0]
        out = np.empty((14, 14, cf.n_channels))
        out[:] = np.nan
        display = np.empty((28,28))
        display[:] = np.nan
        figure, ax = plt.subplots(nrows=3, ncols=4, figsize=(12, 12))
        gs = ax[0,1].get_gridspec()
        [[a.remove() for a in x[1:3]] for x in ax[0:2]]
        bax = figure.add_subplot(gs[0:2,1:3])
        tittle = (cf.datasets[this.data] + this.model_name + " preprocessing")
        stage = []
        scene = []
        bax.set_title(tittle)
        scene += [bax.imshow(start[:,:])]
        scene += [ax[2][0].imshow(image[:,:,0])]
        scene += [ax[2][1].imshow(image[:,:,1])]
        scene += [ax[2][2].imshow(image[:,:,2])]
        scene += [ax[2][3].imshow(image[:,:,3])]
        
        for i in range(5):
            stage += [scene]

        for j in range(0, 28, 2):
            for k in range(0, 28, 2):
                scene = []
                display[j:j+2,k:k+2] = start[j:j+2,k:k+2]
                scene += [bax.imshow(display)]
                x = j*1
                y = k*1
                adv_x = x + 1
                adv_y = y + 1
                x = x//2
                y = y//2
                adv_x = adv_x//2
                adv_y = adv_y//2
                out[x, y, 0] = image[x, y, 0]
                scene += [ax[2][0].imshow(out[:,:,0])]
                out[x, adv_y, 1] = image[x, adv_y, 1]
                scene += [ax[2][1].imshow(out[:,:,1])]
                out[adv_x, y, 2] = image[adv_x, y, 2]
                scene += [ax[2][2].imshow(out[:,:,2])]
                out[adv_x, adv_y, 3] = image[adv_x, adv_y, 3]
                scene += [ax[2][3].imshow(out[:,:,3])]
                stage += [scene]

        for i in range(5):
            stage += [scene]

        tittle = (cf.datasets[this.data] + this.model_name + " reassembling")
        bax.set_title(tittle)
        end = reassembled(image)
        out = np.empty((14, 14, cf.n_channels ))
        out[:] = np.nan
        display = np.empty((28,28))
        display[:] = np.nan
        for j in range(0, 28, 2):
            for k in range(0, 28, 2):
                scene = []
                display[j:j+2,k:k+2] = end[j:j+2,k:k+2]
                scene += [bax.imshow(display)]
                x = j*1
                y = k*1
                adv_x = x + 1
                adv_y = y + 1
                x = x//2
                y = y//2
                adv_x = adv_x//2
                adv_y = adv_y//2
                out[x, y, 0] = image[x, y, 0]
                scene += [ax[2][0].imshow(out[:,:,0])]
                out[x, adv_y, 1] = image[x, adv_y, 1]
                scene += [ax[2][1].imshow(out[:,:,1])]
                out[adv_x, y, 2] = image[adv_x, y, 2]
                scene += [ax[2][2].imshow(out[:,:,2])]
                out[adv_x, adv_y, 3] = image[adv_x, adv_y, 3]
                scene += [ax[2][3].imshow(out[:,:,3])]
                stage += [scene]
        for i in range(5):
            stage += [scene]
        visuals_dir = this.visuals_dir + "filter/"
        os.makedirs(visuals_dir, exist_ok=True)
        file = visuals_dir + cf.datasets[this.data] + this.model_name + ".gif"
        print(file) 
        ani = ArtistAnimation(fig=figure, artists=stage)
        ani.save(file, writer=PillowWriter())


    def weights(self):
        figure, animate = self.individual(this.weights)
        self.animating(figure,animate, "weights")

    def relative_weights(self):

        figure, animate = self.relative(this.weights)
        self.animating(figure,animate,"relative_weights")
        
    def bias(self):
            
        figure, animate = self.individual(this.bias * this.weights)
        self.animating(figure,animate,"bias")

    def relative_bias(self):

        figure, animate = self.relative(this.bias * this.weights)
        self.animating(figure,animate,"relative_bias")

    def data_samples(self):

        sample = np.unique(this.test_labels, return_index=True)
        figure, _ = self.individual([this.pre_test_images[sample[1],...]])
        visuals_dir = this.visuals_dir + "individual_samples" + "/"
        os.makedirs(visuals_dir, exist_ok=True)
        file = visuals_dir + cf.datasets[this.data] + this.model_name + ".png"
        print(file) 
        figure.suptitle(cf.datasets[this.data] + this.model_name + " individual samples")
        figure.savefig(file)

    def relative_samples(self):
        """Plot the samples in single matrix to view the scale of the samples as a whole.
        """        
        sample = np.unique(this.test_labels, return_index=True)
        figure, _ = self.relative([this.pre_test_images[sample[1],...]])
        visuals_dir = this.visuals_dir + "relative_samples" + "/"
        os.makedirs(visuals_dir, exist_ok=True)
        file = visuals_dir + cf.datasets[this.data] + this.model_name + ".png"
        print(file) 
        figure.suptitle(cf.datasets[this.data] + this.model_name + " relative samples")
        figure.savefig(file)

    def manifold_umap(self):
        """Generate a UMAP plot .
        """        
        from cuml.manifold.umap import UMAP
        import umap
        import umap.plot
        import pandas as pd
        from bokeh.io import save
        flattened = this.flatten.predict(this.pre_train_images)
        category_labels = [this.class_labels[x] for x in this.train_labels]
        hover_df = pd.DataFrame(category_labels, columns=['category'])

        reducer = UMAP(
            random_state=42,
            n_components = 2,
            n_neighbors = cf.n_batches,
            n_epochs = cf.n_epochs
        )

        embedding = reducer.fit_transform(flattened,this.test_labels)
        visuals_dir = this.visuals_dir + "UMAP" + "/"
        os.makedirs(visuals_dir, exist_ok=True)
        file = visuals_dir + cf.datasets[this.data] + this.model_name + ".html"
        print(file) 
        title = cf.datasets[this.data] + this.model_name
        umap.plot.output_file(file, title = title)
        #figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
        bkp = umap.plot.interactive(reducer, labels=hover_df['category'],hover_data=hover_df,point_size=4)
        save(bkp)
        #figure.suptitle(cf.datasets[this.data] + this.model_name)
        #figure.savefig(file)

    def circuit(self):
        import pennylane as qml
        if hasattr(this,'q_node'):
            visuals_dir = this.visuals_dir + "circuits" + "/"
            os.makedirs(visuals_dir, exist_ok=True)
            file = visuals_dir + cf.datasets[this.data] + this.model_name + ".png"
            print(file)
            fig, ax = qml.draw_mpl(this.q_node,expansion_strategy="device")(np.asarray([0.1,0.1,0.1,0.1]))
            fig.suptitle(cf.datasets[this.data] + this.model_name)
            fig.savefig(file)