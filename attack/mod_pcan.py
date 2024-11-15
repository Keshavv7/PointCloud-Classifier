import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        
        # Initialize trainable parameters
        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        # Attention network layers
        self.attention_conv1 = nn.Conv1d(feature_size, 128, 1)
        self.attention_bn1 = nn.BatchNorm1d(128)
        self.attention_conv2 = nn.Conv1d(128, 32, 1)
        self.attention_bn2 = nn.BatchNorm1d(32)
        self.attention_conv3 = nn.Conv1d(32, 1, 1)
        
        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        # x shape: [batch_size, feature_size, num_points]
        batch_size = x.size(0)
        
        # Compute attention weights
        attention = F.relu(self.attention_bn1(self.attention_conv1(x)))
        attention = F.relu(self.attention_bn2(self.attention_conv2(attention)))
        attention = self.attention_conv3(attention)  # [batch_size, 1, num_points]
        attention = torch.sigmoid(attention)
        
        # Reshape x for VLAD processing
        x = x.transpose(1, 2).contiguous()  # [batch_size, num_points, feature_size]
        
        # Compute soft-assignments
        activation = torch.matmul(x, self.cluster_weights)  # [batch_size, num_points, cluster_size]
        
        if self.add_batch_norm:
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, self.max_samples, self.cluster_size)
        else:
            activation = activation + self.cluster_biases
            
        activation = self.softmax(activation)
        
        # Apply attention weights
        attention = attention.transpose(1, 2)  # [batch_size, num_points, 1]
        attention = attention.expand(-1, -1, self.cluster_size)  # [batch_size, num_points, cluster_size]
        activation = torch.mul(activation, attention)

        # Compute VLAD
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2
        
        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        # Normalize and reshape
        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        # Final fully connected layer
        vlad = torch.matmul(vlad, self.hidden1_weights)
        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad, attention.squeeze(-1)


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)
        activation = x * gates

        return activation


class ModifiedPCAN(nn.Module):
    def __init__(self, input_feature_size=1088, num_points=2500, output_dim=256):
        super(ModifiedPCAN, self).__init__()
        self.net_vlad = NetVLADLoupe(
            feature_size=input_feature_size,
            max_samples=num_points,
            cluster_size=64,
            output_dim=output_dim,
            gating=True,
            add_batch_norm=True,
            is_training=True
        )

    def forward(self, x):
        # x should be the output from your PointNet
        # Expected shape: [batch_size, feature_size, num_points]
        x, attention_weights = self.net_vlad(x)
        return x, attention_weights
    
from ..model.pointnet import PointNet

# Initialize your PointNet
pointnet = PointNet(local_feat=True)

# Initialize the modified PCAN
pcan = ModifiedPCAN(
    input_feature_size=1088,  # Matches your PointNet output feature size
    num_points=2500,          # Matches your number of points
    output_dim=256           # Desired output dimension
)

# Forward pass
features, indices, A_feat = pointnet(test_data)
global_descriptor, attention_weights = pcan(features)

print(f'Global descriptor shape: {global_descriptor.shape}')
print(f'Attention weights shape: {attention_weights.shape}')