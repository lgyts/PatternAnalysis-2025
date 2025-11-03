# modules.py
# Siamese network modules: encoder and classifier.
# Author: s4778251



import torch
import torch.nn as nn
import torchvision.models as models
from params import OUT_DIM, HIDDEN_DIMS, NEGATIVE_SLOPE, DROPOUT_P


class SiameseEncoder(nn.Module):
    """Feature extraction network for Siamese training.

    This encoder uses a ResNet-50 backbone pretrained on ImageNet and projects the
    resulting feature vector into a normalized embedding space. It is typically
    trained using triplet loss to ensure semantically similar samples are close
    together while dissimilar ones are farther apart.
    """

    def __init__(self, out_dim=OUT_DIM, pretrained=True):
        """
        Args:
            out_dim (int): Dimensionality of the output embedding vector.
            pretrained (bool): Whether to initialize the ResNet-50 backbone with
                ImageNet-pretrained weights.
        """
        super().__init__()
        
        # Load the ResNet50 backbone and remove its final classification layer
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        
        # Linear projection to the target embedding dimension
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input batch of images with shape (B, 3, H, W).

        Returns:
            torch.Tensor: L2-normalized embeddings of shape (B, out_dim).
        """
        feat = self.backbone(x)           # Extract features via ResNet
        emb = self.proj(feat)             # Project into embedding space
        emb = nn.functional.normalize(emb, p=2, dim=1)  # Normalize to unit length
        return emb


class BinaryClassifier(nn.Module):
    """Four-layer MLP classifier for binary prediction.

    This model takes precomputed embeddings (e.g., from SiameseEncoder) and maps
    them through a sequence of fully connected layers with LeakyReLU activation
    and dropout regularization. The output layer produces two logits for binary
    classification.
    """

    def __init__(self, in_dim=OUT_DIM, hidden=HIDDEN_DIMS, num_classes=2,
                 negative_slope=NEGATIVE_SLOPE, p=DROPOUT_P):
        """
        Args:
            in_dim (int): Input feature dimension (should match encoder output).
            hidden (tuple[int]): Sizes of hidden layers.
            num_classes (int): Number of output classes (2 for binary tasks).
            negative_slope (float): Slope for LeakyReLU activation.
            p (float): Dropout probability for regularization.
        """
        super().__init__()
        layers = []
        last = in_dim
        
        # Build MLP layers dynamically from hidden size sequence
        for h in hidden:
            layers += [
                nn.Linear(last, h),
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                nn.Dropout(p)
            ]
            last = h
        
        # Final classification layer without activation 
        layers += [nn.Linear(last, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the classifier.

        Args:
            x (torch.Tensor): Input feature batch (B, in_dim).

        Returns:
            torch.Tensor: Output logits (B, num_classes).
        """
        return self.net(x)
