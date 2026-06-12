import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pairwise_euclidean_distance
from tqdm import tqdm


# ==========================================
# 1. CAC Model Wrapper
# ==========================================
class CACModel(nn.Module):
    """Wraps a feature backbone with a linear projection into the class-anchor space.

    The projection (bias=False) maps backbone features to a K-dimensional
    anchor space. Bias is disabled so the anchors remain purely linear,
    essential for distance-based learning.
    """

    def __init__(self, backbone, feature_dim=448, num_classes=50, feature=False):
        super(CACModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.feature = feature

        # Linear projection: feature_dim -> num_classes (anchor space)
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, x):
        # Extract backbone features: (B, feature_dim)
        feat = self.backbone(x)

        if self.feature:
            return feat

        # Project to anchor space: (B, num_classes)
        logits = self.classifier(feat)
        return logits


# ==========================================
# 2. CAC Loss
# ==========================================
class CACLoss(nn.Module):
    """Class Anchor Clustering Loss.

    Pulls samples toward their fixed orthogonal class anchors and pushes
    them away from all other anchors.

    Loss = tuplet_loss + anchor_scale * anchor_distance

    where:
        tuplet_loss = log(sum_j exp(d_true - d_j))
        anchor_distance = mean_i ||z_i - a_{y_i}||_2
    """

    def __init__(self, num_classes, magnitude=10.0, anchor_scale=0.1, device="cuda"):
        super(CACLoss, self).__init__()
        self.num_classes = num_classes
        self.anchor_scale = anchor_scale
        self.magnitude = magnitude
        self.device = device

        # Fixed orthonormal anchors: magnitude * I
        # Shape: (num_classes, num_classes)
        self.centres = magnitude * torch.eye(num_classes, device=device)

    def forward(self, input, label):
        # input: (B, num_classes) — output of CACModel

        # 1. Euclidean distance from each sample to all anchors
        dists = pairwise_euclidean_distance(input, self.centres)  # (B, num_classes)

        # 2. Distance to the true class anchor
        anchor_dists = torch.gather(dists, 1, label.view(-1, 1))  # (B, 1)

        # 3. Tuplet loss: log(sum_j exp(d_true - d_j))
        tuplet = (anchor_dists - dists).exp().sum(dim=1).log()

        # 4. Total loss = tuplet + lambda * anchor distance
        loss = tuplet.mean() + self.anchor_scale * anchor_dists.mean()

        return loss, anchor_dists.mean()


# ==========================================
# 3. CAC Inference Classifier (for OSR evaluation)
# ==========================================
class CACClassif:
    """Inference-time classifier: recomputes class centers from training
    features and scores samples via: score = d * (1 - softmin(d))

    Using training-set feature means as centers (instead of fixed identity
    anchors) yields better test-time results.

    Lower score → more likely known class.
    Higher score → more likely unknown class.

    Note: test_cac_osr.py uses standalone functions for the same logic
    combined with threshold search and metric computation. This class
    provides the predict_score() API for programmatic use.
    """

    def __init__(self, model, train_loader, device):
        self.model = model
        self.model.eval()
        self.device = device

        print("Initializing CAC Centers from Train Set (Gallery)...")
        self.centres = self._calculate_centers(train_loader)

    def _calculate_centers(self, loader):
        """Compute per-class mean feature vectors over the training loader."""
        features_sum = torch.zeros(self.model.num_classes, self.model.num_classes, device=self.device)
        counts = torch.zeros(self.model.num_classes, device=self.device)

        with torch.no_grad():
            for batch in tqdm(loader, desc="Updating Centers"):
                if isinstance(batch, (list, tuple)):
                    imgs, labels = batch[0], batch[1]
                else:
                    imgs, labels = batch, None

                outputs = self.model(imgs.to(self.device))

                # Accumulate per-class sums
                for i in range(len(labels)):
                    lab = labels[i]
                    feat = outputs[i]
                    features_sum[lab] += feat
                    counts[lab] += 1

        counts = torch.clamp(counts, min=1)
        final_centres = features_sum / counts.view(-1, 1)
        return final_centres

    def predict_score(self, loader):
        """Compute CAC rejection scores.

        score = d * (1 - softmin(d))

        Returns:
            numpy array of per-sample minimum scores (lower = more likely known)
        """
        all_scores = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Scoring"):
                if isinstance(batch, (list, tuple)):
                    imgs, _ = batch[0], batch[1]

                outputs = self.model(imgs.to(self.device))

                # 1. Distance to each class center
                dists = pairwise_euclidean_distance(outputs, self.centres)

                # 2. Rejection score: d * (1 - softmin(d))
                probs = F.softmin(dists, dim=1)
                scores = dists * (1 - probs)

                # Take the minimum score (closest class) per sample
                min_scores, _ = scores.min(dim=1)
                all_scores.append(min_scores.cpu())

        return torch.cat(all_scores).numpy()
