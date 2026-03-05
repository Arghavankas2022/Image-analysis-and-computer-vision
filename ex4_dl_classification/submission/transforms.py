import torchvision.transforms as transforms

def get_transforms_train():
    """Return the transformations applied to images during training."""
    
    transform = transforms.Compose(
        [
      
            transforms.ToPILImage(),
            
            transforms.RandomResizedCrop(size=50, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

            transforms.ToTensor(),  
            
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    return transform


def get_transforms_val():
    """Return the transformations applied to images during validation.

    Note: You do not need to change this function. We want consistent
    validation, so we do not apply random augmentations.
    """
    transform = transforms.Compose(
        [

            transforms.ToTensor(),  # convert image to a PyTorch Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
        ]
    )
    return transform