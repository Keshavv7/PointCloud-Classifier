import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PointNetSetAbstractionMsg(nn.Module):
    """PointNet Set Abstraction Layer with Multi-Scale Grouping"""
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        xyz: input points position data, [B, C, N]
        points: input points data, [B, D, N]
        """
        xyz = xyz.permute(0, 2, 1)  # [B,N,C]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B,N,D]

        B, N, C = xyz.shape
        S = self.npoint
        
        # Sample points
        fps_idx = torch.arange(S, device=xyz.device).repeat(B, 1)  # [B,S]
        new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, C))  # [B,S,C]

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            # Ball query
            dist = torch.cdist(new_xyz, xyz)  # [B,S,N]
            idx = dist.topk(k=K, dim=-1, largest=False)[1]  # [B,S,K]
            
            grouped_xyz = torch.gather(xyz.view(B, 1, N, C).expand(-1,S,-1,-1), 2, 
                                     idx.unsqueeze(-1).expand(-1,-1,-1,C))  # [B,S,K,C]
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            
            if points is not None:
                grouped_points = torch.gather(points.view(B, 1, N, -1).expand(-1,S,-1,-1), 2,
                                           idx.unsqueeze(-1).expand(-1,-1,-1,points.size(-1)))  # [B,S,K,D]
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B,D,K,S]
            
            # Point Feature Embedding
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            
            new_points = torch.max(grouped_points, 2)[0]  # [B,D,S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: input points position data, [B, C, N]
        xyz2: sampled input points position data, [B, C, S]
        points1: input points data, [B, D, N]
        points2: input points data, [B, D, S]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            dists = torch.cdist(xyz1, xyz2)
            dists, idx = torch.topk(dists, k=3, dim=-1, largest=False)
            weight = 1.0 / (dists + 1e-10)
            weight = weight / torch.sum(weight, dim=-1, keepdim=True)
            
            interpolated_points = torch.zeros_like(points1)
            for i in range(B):
                points = points2[i].permute(1, 0)  # [S, D]
                interp = torch.matmul(weight[i], points[idx[i]])  # [N, D]
                interpolated_points[i] = interp.permute(1, 0)  # [D, N]

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points

class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        
        self.softmax = nn.Softmax(dim=-1)
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

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x, attention_weights):
        x = x.transpose(1, 2).contiguous()  # [B,N,D]
        
        activation = torch.matmul(x, self.cluster_weights)  # [B,N,K]
        
        if self.add_batch_norm:
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, self.max_samples, self.cluster_size)
        else:
            activation = activation + self.cluster_biases
            
        activation = self.softmax(activation)
        
        # Apply attention weights
        activation = torch.mul(activation, attention_weights)
        
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2
        
        activation = torch.transpose(activation, 2, 1)
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)
        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad

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
        return x * gates

class PCAN(nn.Module):
    def __init__(self, input_dim=1088, output_dim=256, num_points=2500):
        super(PCAN, self).__init__()
        
        # Point Contextual Attention Network
        self.sa1 = PointNetSetAbstractionMsg(
            256, [0.1, 0.2, 0.4], [16, 32, 64], 
            input_dim, [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            None, None, None, 320, [256, 512, 1024]
        )
        self.fp2 = PointNetFeaturePropagation(1024 + 320, [256, 256])
        self.fp1 = PointNetFeaturePropagation(256, [256, 128])
        
        # Attention scoring
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 1, 1)
        
        # NetVLAD
        self.net_vlad = NetVLADLoupe(
            feature_size=input_dim,
            max_samples=num_points,
            cluster_size=64,
            output_dim=output_dim,
            gating=True,
            add_batch_norm=True
        )
        
    def forward(self, xyz, features):
        """
        xyz: point coordinates [B,3,N]
        features: point features from PointNet [B,D,N]
        """
        # Point contextual attention network
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)
        
        # Generate attention weights
        x = F.relu(self.bn1(self.conv1(l0_points)))
        scores = self.conv2(x)
        attention_weights = torch.sigmoid(scores)
        attention_weights = attention_weights.expand(-1, -1, 64)  # Expand for NetVLAD
        
        # Apply NetVLAD with attention
        global_desc = self.net_vlad(features, attention_weights)
        
        return global_desc, attention_weights
    


# Initialize models
pointnet = PointNet(local_feat=True)  # Your existing PointNet
pcan = PCAN(input_dim=1088, output_dim=256, num_points=2500)  # The new PCAN model

# Forward pass
xyz = test_data.transpose(1,2)  # [B,3,N]
features, _, _ = pointnet(test_data)  # Get local features from PointNet of the shape (batchsize, num_features, num_points)
global_descriptor, attention_weights = pcan(xyz, features)

print(f'Global descriptor shape: {global_descriptor.shape}')  # [B, 256]
print(f'Attention weights shape: {attention_weights.shape}')  # [B, 1, N]