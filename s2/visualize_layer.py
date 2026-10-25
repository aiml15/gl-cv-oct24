"""
What: A small helper that shows what every convolution/pooling layer of a trained CNN "sees".
Why : Keeps the notebook clean - the notebook only has to call visualize_layer(img_path, model).
"""

import numpy as np                                                  # array handling (batch dim, scaling)
import matplotlib.pyplot as plt                                     # to display the images / feature maps
from tensorflow.keras.models import Model                           # to build a model that exposes inner layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # to read an image from disk


def visualize_layer(img_path, model, n_maps=8):
    """
    What: Plots the input image, then the first `n_maps` feature maps of each conv/pool layer.
    Why : Lets us actually see the edges/textures/shapes the network has learnt to detect.

    img_path : path of the image we want to push through the network
    model    : an already trained (and built) Keras model
    n_maps   : how many feature maps to show per layer (8 keeps the plot readable)
    """

    # What: ask the model itself which image size it expects, e.g. (128, 128)
    # Why : avoids hard-coding the size - works even if the model is rebuilt with another size
    target_size = tuple(model.inputs[0].shape[1:3])

    # What: read the image from disk and resize it to the size the model expects
    img = load_img(img_path, target_size=target_size)

    # What: convert the PIL image into a numpy array of shape (128, 128, 3)
    x = img_to_array(img)

    # What: bring the pixel values from 0-255 down to 0-1
    # Why : the model was trained on rescaled (1/255) images, so the input must match
    x = x / 255.0

    # What: add a leading batch dimension -> (1, 128, 128, 3)
    # Why : Keras always predicts on batches of images, never on a single image
    x = np.expand_dims(x, axis=0)

    # What: show the original image first, as a reference for the feature maps below
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.title("Input image")
    plt.axis("off")                                                 # hide pixel-index ticks, they add nothing
    plt.show()

    # What: keep only the layers whose output is still image-like (4D: batch, height, width, filters)
    # Why : Flatten/Dense outputs are plain vectors - there is no picture to look at
    image_layers = [layer for layer in model.layers if len(layer.output.shape) == 4]

    # What: one model, same input as the trained model, but with ONE output per layer above
    # Why : a single predict() call then hands us the feature maps of every layer at once
    viz_model = Model(inputs=model.inputs, outputs=[layer.output for layer in image_layers])

    # What: push our one image through the network and collect every layer's output
    feature_maps = viz_model.predict(x, verbose=0)

    # What: if the model had only one such layer, predict() returns an array instead of a list
    # Why : wrapping it keeps the loop below identical in both cases
    if not isinstance(feature_maps, list):
        feature_maps = [feature_maps]

    # What: go layer by layer and plot that layer's first few feature maps side by side
    for layer, fmap in zip(image_layers, feature_maps):

        # What: never ask for more maps than the layer actually produced
        n = min(n_maps, fmap.shape[-1])

        # What: one row of n small images
        fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))

        # What: with n == 1 matplotlib returns a single axis, not an array - make it always an array
        axes = np.atleast_1d(axes)

        # What: title tells us which layer this row belongs to and the size of its output
        fig.suptitle(f"{layer.name}   ->   output shape {fmap.shape[1:]}")

        for i in range(n):
            # What: fmap[0] drops the batch dim, [:, :, i] picks the i-th filter's feature map
            axes[i].imshow(fmap[0][:, :, i])
            axes[i].axis("off")

        plt.show()
