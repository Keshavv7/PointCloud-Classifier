# model/loss.py
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

# Focal Loss + Regularization Loss
class PointNetLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0, reg_weight=0, size_average=True):
        super(PointNetLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reg_weight = reg_weight
        self.size_average = size_average

        # Sanitize inputs
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,(list, np.ndarray)): self.alpha = torch.Tensor(alpha)

        # Define Balanced Cross Entropy Loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.alpha)


    def forward(self, pred, target, A=None):
        # Get batch size
        bs = pred.size[0]

        # Get Balanced Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(pred, target)
            
        # Get predicted class probabilities for the true class
        pn = F.softmax(pred)
        pn = pn.gather(1, target.view(-1, 1)).view(-1)

        # Get Regularization loss
        if self.reg_weight > 0:
            I = torch.eye(64).unsqueeze(0).repeat(A.shape[0], 1, 1)
            if A.is_cuda: 
                I = I.cuda()
            reg_loss = torch.linalg.norm(I - torch.bmm(A, A.transpose(2,1)))
            reg_loss = (self.reg_weight * reg_loss)/bs
        else: 
            reg_loss = 0
        
        # Compute total loss
        loss = ((1 - pn)**self.gamma * ce_loss)

        if self.size_average:
            return loss.mean() + reg_loss
        
        return loss.sum() + reg_loss
