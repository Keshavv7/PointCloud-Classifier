# main.py
import torch
from config import Config
from data_loader import get_dataloader
from model.pointnet import PointNet
from trainer import Trainer

def main():
    train_loader = get_dataloader(root_dir='path/to/ModelNet10', batch_size=Config.BATCH_SIZE, num_points=Config.NUM_POINTS, split='train')
    test_loader = get_dataloader(root_dir='path/to/ModelNet10', batch_size=Config.BATCH_SIZE, num_points=Config.NUM_POINTS, split='test')

    model = PointNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    trainer = Trainer(model, train_loader, test_loader)

    for epoch in range(1, Config.EPOCHS + 1):
        trainer.train(epoch)
        trainer.test()

if __name__ == '__main__':
    main()
