"""
PyTorch interface to our dataset
No need to modify anything here. In fact you shouldn't.
"""
import h5py
from PIL import Image
from io import BytesIO
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import Dataset


def load_image(hdf5_file, im_path):
    """
    Helper function for loading images stored in a HDF5 file.

    HDF5 enables a compact way to store images, without needing to store each
    image as a separate file.
    """
    return np.array(
        Image.open(BytesIO(bytearray(bytes(hdf5_file[im_path][()]))))
    )


class ImageDataset(Dataset):
    """ PyTorch Dataset for loading our data """

    def __init__(self, annotations_file, img_file, transform=None):
        """
        Initialize dataset
        
        Args:
            annotations_file  ... Specify the file that stores the classification labels.
            img_file          ... Specify the file where all images are stored.
            transform         ... Specif the transformations to apply to the loaded image.
                                  Note that we do not want to apply data augmentations on
                                  validation and test images, since it is a regularization
                                  technique only used in training.
        """
        with open(annotations_file) as fp:
            self.img_labels = [line.strip().split(',') for line in fp.readlines()]
        self.img_file = h5py.File(img_file, mode="r")
        self.transform = transform

    def __len__(self):
        """ Return length of dataset """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """ Get image and labels at the given index """
        image = load_image(self.img_file, self.img_labels[idx][0])
        if self.transform:
            image = self.transform(image)
        if len(self.img_labels[idx]) > 1:
            label = int(self.img_labels[idx][1])
            return image, label
        return image
