import torch
import torch.nn as nn
import torchvision.models as models

class SiameseEncoder(nn.Module):
    # use ResNet50 as backbone for feature extraction
    def __init__(self, out_dim=512, pretrained=True):
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
    
class BinaryClassifier(nn.Module):
    # 4 layer MLP for binary classification, with LeakyReLU activations
    def __init__(self, in_dim=512, hidden=(256, 64, 32), num_classes=2,
                 negative_slope=0.01, p=0.4):  # add p
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [
                nn.Linear(last, h),
                nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
                nn.Dropout(p)  # add dropout
            ]
            last = h
        layers += [nn.Linear(last, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)