# model/pointnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    ''' T-Net learns a Transformation matrix with a specified dimension d'''
    def __init__(self, d, num_points = 2000):
        super(TNet, self).__init__()
        self.d = d

        # For the Shared MLPs
        self.conv1 = nn.Conv1d(d, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # For the FC Layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, d * d)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # For extracting major features
        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

    def forward(self, x):
        # Implement forward function with T-Net structure
        bs = x.shape[0]

        # Pass the points through the shared MLP layers
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))

        # Max Pooling to extract major features in 1024D vector representation
        x = self.max_pool(x).view(bs, -1)

        # Pass the feature vector through FC layers for extracting transformation matrix
        x = self.bn4(F.relu(self.fc1(x)))
        x = self.bn5(F.relu(self.fc2(x)))
        x = self.fc3(x)

        # Add with identity matrix for regularization
        identity_matrix = torch.eye(self.d, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            identity_matrix = identity_matrix.cuda()
        
        transformation_matrix = x.view(-1, self.d, self.d) + identity_matrix

        return transformation_matrix 
        pass

class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PointNet, self).__init__()
        self.tnet1 = TNet(d=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        # Additional layers
        # Final classifier layer

    def forward(self, x):
        # Define forward pass
        x = F.relu(self.bn1(self.conv1))
        x = F.relu(self.bn2(self.conv2))
        pass
