"""Class Anchor Clustering (CAC) model and training objective."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CACModel(nn.Module):
    """Project backbone embeddings into K class logits without a bias term."""

    def __init__(self, backbone, feature_dim=448, num_classes=50, feature=False):
        super(CACModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.feature = feature

        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, x):
        feat = self.backbone(x)

        if self.feature:
            return feat

        logits = self.classifier(feat)
        return logits


class PaperCACLoss(nn.Module):
    """Implement Eqs. (17) and (18) on the K-dimensional class logits.

    Cross-entropy uses the logits directly. A second term penalizes the mean
    squared Euclidean distance to the target class's fixed anchor, defined
    as its one-hot class vector multiplied by the anchor magnitude.
    """

    def __init__(self, num_classes, magnitude=10.0, anchor_scale=0.1):
        super().__init__()
        self.anchor_scale = anchor_scale
        self.register_buffer("centres", magnitude * torch.eye(num_classes))

    def forward(self, logits, targets):
        cross_entropy = F.cross_entropy(logits, targets)
        squared_distance = (logits - self.centres[targets]).square().sum(dim=1)
        anchor_loss = squared_distance.mean()
        return cross_entropy + self.anchor_scale * anchor_loss, anchor_loss
