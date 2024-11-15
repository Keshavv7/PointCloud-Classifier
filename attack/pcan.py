import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..model.pointnet import PointNet

class PCAN_Attention(nn.Module):
    def __init__(self, feature_size=1086, num_points=2500, cluster_size=64):
        """
        PCAN Attention module adapted for PointNet features
        Args:
            feature_size: Dimension of input features (default: 1086 from PointNet)
            num_points: Number of points in the point cloud (default: 2500)
            cluster_size: Number of clusters for attention (default: 64)
        """
        super(PCAN_Attention, self).__init__()
        
        self.feature_size = feature_size
        self.num_points = num_points
        self.cluster_size = cluster_size
        
        # Feature compression network
        self.compress = nn.Sequential(
            nn.Conv1d(feature_size, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input features (batch_size, feature_size, num_points)
        Returns:
            attention_weights: Attention weights for each point (batch_size, num_points, 1)
            weighted_features: Features weighted by attention (batch_size, feature_size, num_points)
        """
        # Compress features
        compressed = self.compress(x)  # (batch_size, 128, num_points)
        
        # Calculate attention weights
        attention_weights = self.attention(compressed)  # (batch_size, 1, num_points)
        
        # Apply attention to original features
        weighted_features = x * attention_weights  # (batch_size, feature_size, num_points)
        
        # Reshape attention weights for output
        attention_weights = attention_weights.transpose(1, 2)  # (batch_size, num_points, 1)
        
        return attention_weights, weighted_features

def main():
    # Example usage with your PointNet
    batch_size = 32
    num_points = 2500
    feature_size = 1086  # 64 (local) + 1024 (global) = 1086
    
    # Create sample input (simulating PointNet features)
    sample_input = torch.randn(batch_size, feature_size, num_points)
    
    # Initialize PCAN attention module
    pcan = PCAN_Attention(feature_size=feature_size, num_points=num_points)
    
    # Get attention weights and weighted features
    attention_weights, weighted_features = pcan(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Weighted features shape: {weighted_features.shape}")


    # First, get features from PointNet
    pointnet = PointNet(local_feat=True)
    feat, ind, A = pointnet(x)  # x is your input point cloud
    feature_map = feat  # Shape: (batch_size, 1086, num_points)

    # Initialize PCAN attention module
    pcan = PCAN_Attention(feature_size=1086, num_points=2500)

    # Get attention weights and weighted features
    attention_weights, weighted_features = pcan(feature_map)

    # attention_weights will have shape (batch_size, num_points, 1)
    # weighted_features will have shape (batch_size, 1086, num_points)

if __name__ == "__main__":
    main()