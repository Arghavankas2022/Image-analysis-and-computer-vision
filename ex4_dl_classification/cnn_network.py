import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResidualBlock(nn.Module):
    """
    A simple ResNet block with two conv layers and a skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        # Save the input for the skip connection
        identity = self.shortcut(x)
        
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add the skip connection
        out = out + identity
        
        # Final ReLU
        out = F.relu(out)
        return out

class CNN(nn.Module):
    def __init__(self):
        """Initialize layers."""
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Stack of Residual Blocks
        self.layers = nn.Sequential(
            # 32x50x50 -> 32x50x50
            ResidualBlock(in_channels=32, out_channels=32),
            
            # 32x50x50 -> 64x25x25 
            ResidualBlock(in_channels=32, out_channels=64, stride=2),
            
            # 64x25x25 -> 128x13x13 
            # Note: 25x25 / 2 = 12.5, rounds up to 13
            ResidualBlock(in_channels=64, out_channels=128, stride=2),
            
            # 128x13x13 -> 256x7x7 
            ResidualBlock(in_channels=128, out_channels=256, stride=2)
        )
        
        # Final pooling and Classifier
        # This brings the 7x7 image to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 256 features -> 128
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 6)     # 128 -> 6 classes
        )

    def forward(self, x):
        """Forward pass of network."""
        x = self.stem(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten (B, 256, 1, 1) -> (B, 256)
        x = self.classifier(x)
        return x

    def write_weights(self, fname):
        """ Store learned weights of CNN """
        torch.save(self.state_dict(), fname)

    def load_weights(self, fname):
        """
        Load weights from file in fname.
        The evaluation server will look for a file called checkpoint.pt
        """
        ckpt = torch.load(fname, weights_only=True)
        self.load_state_dict(ckpt)


def get_loss_function():
    """Return the loss function to use during training. We use
       the Cross-Entropy loss for now.
    
    See https://pytorch.org/docs/stable/nn.html#loss-functions.
    """
    return nn.CrossEntropyLoss()


def get_optimizer(network, lr=0.001,momentum=0.9,weight_decay=1e-4):
    """Return the optimizer to use during training.
    
    network specifies the PyTorch model.

    See https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer.
    
    """
    return optim.SGD(
        network.parameters(), 
        lr=lr, 
        momentum=momentum  
    )

    # The fist parameter here specifies the list of parameters that are
    # learnt. In our case, we want to learn all the parameters in the network
    return optim.SGD(network.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
