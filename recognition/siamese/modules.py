import torch
import torch.nn as nn
import torchvision.models as models

class SiameseEncoder(nn.Module):
    # use ResNet50 as backbone for feature extraction
    def __init__(self, out_dim=1000, pretrained=True):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.proj(feat)
        emb = nn.functional.normalize(emb, p=2, dim=1)  # L2 normalize
        return emb