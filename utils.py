# utils.py
import torch

def calculate_accuracy(pred, target):
    pred = pred.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct



I = torch.eye(8)
print(I)
print(I.shape)
I = I.unsqueeze(0)
print(I)
print(I.shape)
I = I.repeat(5, 1, 1)
print(I)
print(I.shape)