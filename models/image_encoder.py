import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    """CNN-based image encoder for food images"""
    
    def __init__(self, embedding_dim=768, pretrained=True):
        """
        Args:
            embedding_dim (int): Dimension of the output embeddings
            pretrained (bool): Whether to use pretrained weights
        """
        super(ImageEncoder, self).__init__()
        
        # Load pretrained ResNet
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Replace with a new FC layer that outputs embedding_dim features
        self.fc = nn.Linear(self.resnet.fc.in_features, embedding_dim)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Image embeddings of shape (batch_size, embedding_dim)
        """
        # Extract features
        x = self.features(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Project to embedding dimension
        x = self.fc(x)
        
        return x