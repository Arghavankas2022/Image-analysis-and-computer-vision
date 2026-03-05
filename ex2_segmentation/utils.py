import time
from pathlib import Path

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_image(im_path):
    """
    This function loads an image from file and returns it as NumPy array.
    The NumPy array will have shape Height x Witdh x Channels and data of type unit8 in the range [0, 255]
    """
    im = Image.open(im_path)
    return np.asarray(im)


def load_sample(sample_path):
    """
    Args:
        sample_path ... Expected to be Path() object to a sample directory
        
    Returns:
        A dictionary with following keys, values:
        'img'         ... The RGB image as dtype uint8 and in range [0, 255]
        'scribble_fg' ... Foreground scribble - Pixels with intensity 255 are considered FG
        'scribble_bg' ... Background scribble - Pixels with intensity 255 are considered BG
        'mask_true'   ... The ground truth segmentation mask
        
    """
    fname = sample_path / 'mask.jpg'
    mask_true = load_image(sample_path / 'mask.jpg')
    
    img = load_image(sample_path / 'im_rgb.jpg')
    
    scribble_fg = load_image(sample_path / 'mask_fg.jpg')
    scribble_bg = load_image(sample_path / 'mask_bg.jpg')

    # Remove weird downsampling artifacts
    scribble_fg = np.where(scribble_fg >= 200, 255, 0)
    scribble_bg = np.where(scribble_bg >= 200, 255, 0)


    return {
        'img': img,
        'scribble_fg': scribble_fg,
        'scribble_bg': scribble_bg,
        'mask_true': mask_true
    }


def show_image(im, title=None):
    plt.imshow(im)
    plt.title(title)
    plt.show()


def show_sample(sample_dd, show_scribble=False, mask_pred=None):
    """
    Args:
        sample_dd ... A dictionary as returned by load_sample()
        mask_pred ... Binary NumPy array with predicted segmentation 
    """
    num_images = 1 + (sample_dd['mask_true'] is not None) + (mask_pred is not None)
    if show_scribble:
        num_images += 2

    fig, ax = plt.subplots(1, num_images, figsize=(10, 10))

    ax[0].imshow(sample_dd['img'])
    ax[0].set_title('RGB Image')

    idx = 1
    if sample_dd['mask_true'] is not None:
        ax[idx].imshow(sample_dd['mask_true'], interpolation='nearest')
        ax[idx].set_title('True Mask')
        idx += 1

    if show_scribble:
        ax[idx].imshow(sample_dd["scribble_fg"], interpolation="nearest")
        ax[idx].set_title("FG Scribble")
        idx += 1
        
        ax[idx].imshow(sample_dd["scribble_bg"], interpolation="nearest")
        ax[idx].set_title("BG Scribble")
        idx += 1

    if mask_pred is not None:
        ax[idx].imshow(mask_pred, interpolation='nearest')
        ax[idx].set_title('Pred Mask')
    plt.show()


def compute_iou(true, pred):
    """
    Calculate intersection over union metric for binary segmentations
    
    Args:
        true ... Ground truth segmentation mask. Supposed to be boolean array
        pred ... Predicted segmentation mask. Supposed to be boolean array
    """
    assert true.dtype == bool, "Expected boolean arrays"
    assert pred.dtype == bool, "Expected boolean arrays"
    
    intersection = (true * pred).sum()
    union = true.sum() + pred.sum() - intersection
    return intersection / union


def evaluate_segmentation(segmenter, sample_dirs, seed, display=False):
    """ Evaluate segmentation algorithm on images in sample_dirs
    
    Args:
        segmenter   ... Instance of ImageSegmenter
        sample_dirs ... List of paths to sample images
        seed        ... Provide a random seed to be used for evaluation
        display     ... If True show results
        
    Returns:
        mean_iou  ... IoU metric averaged over images
        mean_time ... Computation time averaged over images
    """
    # Run evaluation with provided random seed
    segmenter.rng = np.random.default_rng(seed)

    iou_score_all = []
    time_all = []
    for sample_path in sample_dirs:

        # Load validation sample
        sample_dd = load_sample(sample_path)

        # Run your segmentation algorithm
        t1 = time.time()
        mask_pred = segmenter.segment_image(sample_dd)
        t2 = time.time()

        iou_score = compute_iou(
            sample_dd['mask_true'].astype(bool),
            mask_pred.astype(bool)
        )
        iou_score_all.append(iou_score)
        time_all.append(t2 - t1)

        if display:
            # Visualize your prediction
            show_sample(sample_dd, mask_pred=mask_pred)
            print(f'IoU score is: {iou_score:0.3f}')

    mean_iou = sum(iou_score_all) / len(iou_score_all)
    mean_time = sum(time_all) / len(time_all)
    
    return mean_iou, mean_time
