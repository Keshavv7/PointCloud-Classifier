import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PointNetLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0, reg_weight=0.001):
        """
        Loss function for PointNet with focal loss, balanced cross-entropy,
        and optional regularization.

        :param alpha: Class weights for balanced cross-entropy. Can be None, float, or list.
        :param gamma: Focal loss parameter to focus on hard examples.
        :param reg_weight: Weight for the orthogonal regularization term.
        """
        super(PointNetLoss, self).__init__()
        self.gamma = gamma
        self.reg_weight = reg_weight

        # Handle alpha for class balancing
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float)
        elif isinstance(alpha, (list, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        else:
            self.alpha = None

        # Initialize cross-entropy loss with class weights
        self.cross_entropy_loss = None

    def forward(self, predictions, targets, transform_matrix=None):
        """
        Compute the loss.

        :param predictions: Model outputs (logits).
        :param targets: Ground truth class indices.
        :param transform_matrix: Transformation matrix for regularization.
        """
        device = predictions.device

        # Transfer alpha to the same device as predictions
        if self.alpha is not None:
            weight = self.alpha.to(device)
        else:
            weight = None

        # Initialize cross-entropy loss with the correct device
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weight)

        # Cross-Entropy Loss
        ce_loss = self.cross_entropy_loss(predictions, targets)

        # Focal Loss Component
        if self.gamma > 0:
            probs = F.softmax(predictions, dim=1)
            target_probs = probs.gather(1, targets.view(-1, 1)).view(-1)
            focal_weight = (1 - target_probs) ** self.gamma
            ce_loss = focal_weight * ce_loss

        # Regularization Term
        reg_loss = 0
        if self.reg_weight > 0 and transform_matrix is not None:
            I = torch.eye(transform_matrix.size(1)).to(device)
            batch_size = transform_matrix.size(0)
            mat_diff = torch.bmm(transform_matrix, transform_matrix.transpose(1, 2)) - I
            reg_loss = torch.mean(torch.norm(mat_diff, dim=(1, 2)))
            reg_loss *= self.reg_weight

        # Combine Losses
        total_loss = ce_loss.mean() + reg_loss
        return total_loss