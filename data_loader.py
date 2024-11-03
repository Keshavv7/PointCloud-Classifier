# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class ModelNet10Dataset(Dataset):
    def __init__(self, root_dir, num_points=2048, split='train'):
        self.root_dir = os.path.join(root_dir, split)
        self.num_points = num_points
        self.data, self.labels = self.load_data()

    def load_data(self):
        # Implement data loading for ModelNet40
        data, labels = [], []
        # Load .npy files or other formats specific to ModelNet40
        for path in os.listdir(self.root_dir):
            # Append data and labels here
            pass
        return np.array(data), np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pointcloud = self.data[idx]
        label = self.labels[idx]
        if pointcloud.shape[0] > self.num_points:
            pointcloud = pointcloud[:self.num_points]
        return torch.FloatTensor(pointcloud), torch.LongTensor([label])

def get_dataloader(root_dir, batch_size, num_points, split):
    dataset = ModelNet10Dataset(root_dir, num_points, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
