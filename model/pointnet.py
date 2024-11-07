# model/pointnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    ''' T-Net learns a Transformation matrix with a specified dimension d'''
    def __init__(self, d, num_points = 2500):
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
        bs = x.shape[0]     # Shape of x = (batch_size, dim, num_points)

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

        if self.local_feat:
            return features, critical_indexes, A_feat
        
        return global_features, critical_indexes, A_feat

# Test 
def main():
    test_data = torch.rand(32, 3, 2500)

    ## test T-net
    tnet = TNet(d=3)
    transform = tnet(test_data)
    print(f'T-net output shape: {transform.shape}')

    ## test the pointnet
    pointnet_seg = PointNet(local_feat=True)
    features_seg, index_seg, A_feat_seg = pointnet_seg(test_data)
    print(f'The shape of features extracted by Pointnet for segmentation: {features_seg.shape}')
    print(f'Shape of indices: {index_seg.shape}')
    print(f'Shape of feature_matrix: {A_feat_seg.shape}')

    pointnet_cls = PointNet(local_feat=False)
    features_cls, index_cls, A_feat_cls = pointnet_cls(test_data)
    print(f'The shape of features extracted by Pointnet for classification: {features_cls.shape}')
    print(f'Shape of indices: {index_cls.shape}')
    print(f'Shape of feature_matrix: {A_feat_cls.shape}')



if __name__ == '__main__':
    main()