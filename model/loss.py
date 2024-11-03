# model/loss.py
import torch.nn as nn

class PointNetLoss(nn.Module):
    def __init__(self):
        super(PointNetLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, target, reg_loss):
        return self.cross_entropy(pred, target) + 0.001 * reg_loss
