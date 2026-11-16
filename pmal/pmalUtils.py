import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pairwise_euclidean_distance
from tqdm import tqdm
import math
import numpy as np
import logging

_logger = logging.getLogger('pmal')

# ==========================================
# 0. Backbone Factory (shared by train & test)
# ==========================================
def create_pmal_backbone(model_name, feature_dim, drop_rate=0.0, drop_path_rate=0.0, img_size=224):
    """
    Create backbone that outputs exactly `feature_dim`-dim features.
    Supports 'plankton_net' and any timm model name.
    Returns (backbone, backbone_out_dim) where backbone(x) -> (B, feature_dim).
    """
    from .PlanktonNet import PlanktonNet

    if model_name == 'plankton_net':
        _logger.info(f"Creating PlanktonNet backbone (output dim={feature_dim})")
        backbone = PlanktonNet(pretrained=False, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        return backbone, feature_dim

    _logger.info(f"Creating timm backbone: {model_name}")
    import timm

    model_kwargs = dict(pretrained=False, num_classes=0, drop_rate=drop_rate)
    if drop_path_rate > 0:
        model_kwargs['drop_path_rate'] = drop_path_rate

    try:
        timm_backbone = timm.create_model(model_name, **model_kwargs)
    except TypeError:
        if 'drop_path_rate' in model_kwargs:
            model_kwargs.pop('drop_path_rate')
        timm_backbone = timm.create_model(model_name, **model_kwargs)

    with torch.no_grad():
        dummy = torch.zeros(2, 3, img_size, img_size)
        backbone_out_dim = timm_backbone(dummy).shape[1]
    _logger.info(f"Detected backbone output dim: {backbone_out_dim}")

    if backbone_out_dim == feature_dim:
        return timm_backbone, feature_dim

    _logger.info(f"Adding projection: {backbone_out_dim} -> {feature_dim}")
    proj = nn.Linear(backbone_out_dim, feature_dim, bias=False)
    backbone = nn.Sequential(timm_backbone, proj)
    return backbone, feature_dim


# ==========================================
# 1. PMAL 模型包装器 (输出原始特征，不归一化)
# ==========================================
class PMALModel(nn.Module):
    def __init__(self, backbone, feature_dim=448, num_classes=50, feature=False):
        super(PMALModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.feature = feature

        # 映射层：448 -> 50 (与 CAC 保持一致，便于计算欧氏距离)
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, x):
        # 1. 提取骨干特征
        feat = self.backbone(x)

        if self.feature:
            return feat

        # 2. 映射到类别空间 (Batch, 50)
        # 注意：这里输出的是未归一化的 Logits/Features
        mapped_feat = self.classifier(feat)
        return mapped_feat


# ==========================================
# 2. Euclidean PMAL Loss (核心改进)
# ==========================================
class PMALoss(nn.Module):
    def __init__(self, num_classes, magnitude=10.0, alpha=1.0, beta=1.0, device="cuda"):
        super(PMALoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # 分类权重
        self.beta = beta  # 边界权重 (建议设大一点，如 1.0)
        self.device = device

        # 1. 原型 (Prototypes): 可学习，初始化为 CAC 的固定锚点
        # 这样能继承 CAC 的几何优势，同时允许微调
        self.prototypes = nn.Parameter(magnitude * torch.eye(num_classes, device=device))

        # 2. 自适应半径 (Limits): 可学习
        # 初始化为 magnitude 的一半，给一个初始范围
        self.limits = nn.Parameter(torch.ones(num_classes, device=device) * (magnitude / 2.0))

        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, label):
        # input: (B, num_classes)
        # prototypes: (K, num_classes)

        # 1. 计算欧氏距离 (Batch, K)
        dists = pairwise_euclidean_distance(input, self.prototypes)

        # 2. 分类 Loss (基于距离的 CrossEntropy)
        # 距离越小越好 -> Logits 应该取负距离 (-dists)
        # 这样 Softmax 概率最大的就是距离最小的
        loss_cls = self.ce(-dists, label)

        # 3. 自适应限制 Loss (Limit Loss)
        # 获取正确类别的距离
        anchor_dists = torch.gather(dists, 1, label.view(-1, 1)).squeeze()  # (B,)
        # 获取正确类别的 limit
        anchor_limits = self.limits[label]  # (B,)

        # 目标：Dist < Limit
        # Hinge Loss: max(0, Dist - Limit + margin)
        # margin 设为 0.1，强迫样本还要往里缩一点
        loss_limit = F.relu(anchor_dists - anchor_limits + 0.1).mean()

        # 4. 正则化: 限制 Limit 不要无限变大
        loss_reg = self.limits.mean()

        # 总 Loss
        total_loss = self.alpha * loss_cls + self.beta * (loss_limit + 0.1 * loss_reg)

        return total_loss, self.limits.mean()


# ==========================================
# 3. PMAL Wrapper (兼容优化器)
# ==========================================
class PMALWrapper(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x, labels=None):
        feat = self.model(x)
        if labels is not None:
            # 训练模式：计算 Loss
            loss, avg_limit = self.loss_fn(feat, labels)
            return loss, avg_limit
        else:
            # 验证/测试模式：返回特征
            return feat


# ==========================================
# 4. PMAL 推理器 (基于归一化欧氏距离)
# ==========================================
class PMALClassif:
    def __init__(self, wrapper, device):
        self.model = wrapper.model
        self.loss_fn = wrapper.loss_fn
        self.model.eval()
        self.device = device

    def predict_score(self, loader):
        """
        Score = Distance / Limit
        Score < 1.0 -> Known
        Score > 1.0 -> Unknown
        """
        all_scores = []

        # 获取当前的 Prototype 和 Limit
        prototypes = self.loss_fn.prototypes.detach()
        limits = self.loss_fn.limits.detach().abs() + 1e-6  # 防止除0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Scoring"):
                if isinstance(batch, (list, tuple)):
                    imgs, _ = batch[0], batch[1]

                outputs = self.model(imgs.to(self.device))

                # 计算到所有原型的欧氏距离
                dists = pairwise_euclidean_distance(outputs, prototypes)

                # 归一化距离: dist / limit
                # (B, K)
                norm_dists = dists / limits.view(1, -1)

                # 取最小的归一化距离作为 Score
                min_scores, _ = norm_dists.min(dim=1)
                all_scores.append(min_scores.cpu())

        return torch.cat(all_scores).numpy()