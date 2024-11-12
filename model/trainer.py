# trainer.py
import torch
import torch.optim as optim
from config import Config
from model.loss import PointNetLoss
from utils import calculate_accuracy

class Trainer:
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = PointNetLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        self.device = Config.DEVICE

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Training loop
            pass

    def test(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                # Evaluation loop
                pass
        print('Test Accuracy:', correct / len(self.test_loader.dataset))
