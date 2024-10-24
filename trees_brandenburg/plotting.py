from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import torch

# needed to undo axis swapping and normalization applied beforehand
def make_image_plottable(img: torch.Tensor) -> np.ndarray:
    img /= 2
    img += 0.5
    npimg = img.numpy().transpose((1, 2, 0))
    return npimg


def plot_images(data: torch.Tensor, reverse_label_encoding: Dict[int, str], nrows: int = 3, ncols: int = 5, **args) -> Tuple[plt.figure, plt.axis]:
    dataiter = iter(data)
    images, labels = next(dataiter)
    fig, axis = plt.subplots(nrows, ncols, **args)
    for i, ax in enumerate(axis.flat):
        with torch.no_grad():
            image, label = images[i], labels[i]
            ax.imshow(make_image_plottable(image))
            ax.set(title = f"{reverse_label_encoding[label.item()]}")
    
    return fig, ax


def plot_validation(data: torch.Tensor, model: torch.nn.Module, reverse_label_encoding: Dict[int, str], nrows: int = 3, ncols: int = 5, **args) -> Tuple[plt.figure, plt.axis]:
    dataiter = iter(data)
    images, labels = next(dataiter)
    fig, axis = plt.subplots(nrows, ncols, **args)
    with torch.inference_mode():
        for ax, image, label in zip(axis.flat, images, labels):
            ax.imshow(make_image_plottable(image))
            image_tensor = image.unsqueeze_(0)
            output_ = model(image_tensor).argmax()
            k = output_.item()==label.item()
            ax.set_title(str(reverse_label_encoding[label.item()])+":" +str(k))

    return fig, ax