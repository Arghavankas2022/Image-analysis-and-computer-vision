import torch
import numpy as np
import random
import matplotlib.pyplot as plt


def seed_everything(seed:int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def show_images(ims, gt_labels, pred_labels=None):
    fig, ax = plt.subplots(1, len(ims), figsize=(12, 12))
    for i in range(len(ims)):
        im = ims[i]
        im = im / 2 + 0.5     # unnormalize
        im_np = im.numpy()

        ax[i].imshow(np.transpose(im_np, (1, 2, 0)))

        if pred_labels is None:
            im_title = f'GT: {gt_labels[i]}'
        else:
            im_title = f'GT: {gt_labels[i]}  '
            im_title += f' Pred: {pred_labels[i]}'
        ax[i].set_title(im_title)
    plt.show()


def show_class_accs(accs_dict, class_names=None, title=" "):
    accs, names = list(), list()
    for key, value in accs_dict.items():
        name = class_names[key] if class_names is not None else str(key)
        names.append(name)
        accs.append(value)

    y_pos = np.arange(len(names))

    fig, ax = plt.subplots()

    hbars = ax.barh(y_pos, accs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Accuracy')
    ax.set_title(title)

    plt.show()