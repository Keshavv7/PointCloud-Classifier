# utils.py
import torch

def calculate_accuracy(pred, target):
    pred = pred.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct
