# model/pointnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet import TNet

class PointNet(nn.Module):
    '''
    This is the main portion of Point Net before the classification and segmentation heads.
    The main function of this network is to obtain the local and global point features, 
    which can then be passed to each of the heads to perform either classification or
    segmentation. The forward pass through the backbone includes both T-nets and their 
    transformations, the shared MLPs, and the max pool layer to obtain the global features.

    The forward function either returns the global or combined (local and global features)
    along with the critical point index locations and the feature transformation matrix. The
    feature transformation matrix is used for a regularization term that will help it become
    orthogonal. (i.e. a rigid body transformation is an orthogonal transform and we would like
    to maintain orthogonality in high dimensional space). "An orthogonal transformations preserves
    the lengths of vectors and angles between them"
    ''' 
    def __init__(self, num_points=2500, num_global_feats=1024, local_feat=True):
        ''' Initializers:
                num_points - number of points in point cloud
                num_global_feats - number of Global Features for the main 
                                   Max Pooling layer
                local_feat - if True, forward() returns the concatenation 
                             of the local and global features
        '''
        super(PointNet, self).__init__()

        # Initializing basic variables
        self.num_points = num_points
        self.num_global_feats = num_global_feats
        self.local_feat = local_feat
        self.features_cls = None
        self.features_seg = None

        # Initializing Spatial Transformer Networks (T-nets)
        self.tnet1 = TNet(d=3, num_points=self.num_points)
        self.tnet2 = TNet(d=64, num_points=self.num_points)

        # Initializing SharedMLP 01
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        # Initializing SharedMLP 02
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, self.num_global_feats, kernel_size=1)

        # Initializing Batch norms for both shared MLPs
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.num_global_feats)

        # max pool to get the global features
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)

    def forward(self, x):
        # Get batch size
        bs = x.shape[0]     # Shape of x = (batch_size, dim, num_points)

        # Retrieve transform matrix using TNet-01
        A_input = self.tnet1(x)

        # First transformation of x using batch matrix multiplication
        x = torch.bmm(x.transpose(2,1), A_input).transpose(2,1)

        # Pass x through the first Shared MLP
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))

        # Retrieve Feature transform matrix using TNet-02
        A_feat = self.tnet2(x)

        # Second transformation of x using batch matrix multiplication
        x = torch.bmm(x.transpose(2,1), A_feat).transpose(2,1)

        # Capture local features for segmentation head
        local_features = x.clone()

        # Pass x through the second Shared MLP
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))

         # get global feature vector and critical indexes
        global_features, critical_indexes = self.max_pool(x)
        global_features = global_features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)

        features = torch.cat((local_features, 
                    global_features.unsqueeze(-1).repeat(1, 1, self.num_points)), 
                    dim=1)
        
        self.features_seg = features    # Shape = (batch_size, num_global_feat + num_local_feat, num_points)
        self.features_cls = global_features # Shape = (batch_size, num_global_feat)

        if self.local_feat:     # For Segmentation
            return features, critical_indexes, A_feat
        
        # For Classification
        return global_features, critical_indexes, A_feat