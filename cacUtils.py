import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. CAC 模型包装器 (适配 PlanktonNet)
# ==========================================
class CACModel(nn.Module):
    def __init__(self, backbone, feature_dim=448, num_classes=50, feature=False):
        super(CACModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.feature = feature

        # 关键层：将 Backbone 的 448 维特征映射到 50 维类别锚点空间
        # Bias=False 是必须的，因为 CAC 的锚点是固定的，这有助于纯粹的距离学习
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, x):
        # 1. 提取骨干特征 (Batch, 448)
        feat = self.backbone(x)

        if self.feature:
            return feat

        # 2. 映射到类别锚点空间 (Batch, 50)
        logits = self.classifier(feat)
        return logits


# ==========================================
# 2. CAC Loss (核心损失函数)
# ==========================================
class PaperCACLoss(nn.Module):
    """Eqs. (17)–(18), using the K-dimensional projection as the CE logits."""

    def __init__(self, num_classes, magnitude=10.0, anchor_scale=0.1):
        super().__init__()
        self.anchor_scale = anchor_scale
        self.register_buffer("centres", magnitude * torch.eye(num_classes))

    def forward(self, logits, targets):
        cross_entropy = F.cross_entropy(logits, targets)
        squared_distance = (logits - self.centres[targets]).square().sum(dim=1)
        anchor_loss = squared_distance.mean()
        return cross_entropy + self.anchor_scale * anchor_loss, anchor_loss
