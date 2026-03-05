""" Some helpers """
import torch

import numpy as np
import matplotlib.pyplot as plt

# This provides the mapping from the integer labels to a descriptive name
CLASSES = ["Buildings", "Forest", "Mountains", "Glacier", "Sea", "Street"]


def compute_accuracy(model, data_loader):
    """
    Compute accuracy of <model> over some data returned by <data_loader>
    
    Args:
        model         ... A PyTorch model predicting image labels
        data_loader   ... Dataloader returning images & GT labels 
    """

    # Fix a random seed
    np.random.seed(6174)
    torch.manual_seed(6174)

    # Set the model to eval mode
    model.eval()

    total, correct = 0, 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def show_images(ims, gt_labels, pred_labels=None):
    fig, ax = plt.subplots(1, len(ims), figsize=(20, 20))
    for id in range(len(ims)):
        im = ims[id]
        im = im / 2 + 0.5     # unnormalize
        im_np = im.numpy()

        ax[id].imshow(np.transpose(im_np, (1, 2, 0)))

        if pred_labels is None:
            im_title = f'GT: {CLASSES[gt_labels[id]]}'
        else:
            im_title = f'GT: {CLASSES[gt_labels[id]]}   Pred: {CLASSES[pred_labels[id]]}'
        ax[id].set_title(im_title)
    plt.show()