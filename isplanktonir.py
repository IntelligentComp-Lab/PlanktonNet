import torch.nn as nn
import torch.nn.functional as F


class CosineClassifier(nn.Module):
    """
    PlanktonNet backbone + 余弦相似度分类头。

    feature_dim: backbone 输出维度 (PlanktonNet=448)
    num_classes: 已知类别数
    temperature: 温度缩放因子 (控制 logits 的动态范围)
    """
    def __init__(self, backbone, backbone_dim, feature_dim=512, num_classes=50,
                 temperature=16.0):
        super().__init__()
        self.backbone = backbone
        self.temperature = temperature

        self.proj = nn.Linear(backbone_dim, feature_dim, bias=False)
        self.bn = nn.BatchNorm1d(feature_dim)
        self.prototype = nn.Linear(feature_dim, num_classes, bias=False)

        nn.init.xavier_normal_(self.proj.weight)
        nn.init.xavier_normal_(self.prototype.weight)

    def forward(self, x, return_features=False):
        feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat.flatten(1)
        feat = self.proj(feat)
        feat_bn = self.bn(feat)
        feat_norm = F.normalize(feat_bn, p=2, dim=1)
        proto_norm = F.normalize(self.prototype.weight, p=2, dim=1)
        cosine = F.linear(feat_norm, proto_norm)
        logits = self.temperature * cosine
        if return_features:
            return logits, feat_norm
        return logits

    def cosine_similarity(self, x):
        """Return raw cosine similarity (no temperature)."""
        feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat.flatten(1)
        feat = self.bn(self.proj(feat))
        feat_norm = F.normalize(feat, p=2, dim=1)
        proto_norm = F.normalize(self.prototype.weight, p=2, dim=1)
        return F.linear(feat_norm, proto_norm)
