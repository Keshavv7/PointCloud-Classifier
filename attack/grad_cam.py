import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..model.pointnet import PointNet

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.features = None
        self.gradients = None

        # Register hook to capture the feature map and gradients
        self.target_layer.register_forward_hook(self.save_features)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_features(self, module, input, output):
        self.features = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_attention_map(self, target_class):
        # Compute gradients of target class with respect to the features
        self.model.zero_grad()
        target_class_score = self.model.features_cls[0, target_class]  # Assuming batch_size = 1
        target_class_score.backward()

        # Compute weights for the attention map
        gradients = self.gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2])  # Mean over the points and feature dimensions

        # Weight the feature map with the gradients
        weighted_features = self.features * pooled_gradients.view(1, -1, 1)
        attention_map = torch.mean(weighted_features, dim=1)  # Mean over feature channels

        # Apply ReLU to ignore negative values (we only want the positive parts)
        attention_map = F.relu(attention_map)

        # Normalize the attention map
        attention_map = torch.squeeze(attention_map)
        attention_map = (attention_map - torch.min(attention_map)) / (torch.max(attention_map) - torch.min(attention_map))

        return attention_map


# Initialize PointNet model
model = PointNet(num_points=2500, num_global_feats=1024, local_feat=True)

# Select a target layer for Grad-CAM (conv5 layer in this case)
grad_cam = GradCAM(model, model.conv5)

# Example point cloud input (shape: [batch_size, num_features, num_points])
point_cloud_input = torch.rand(1, 3, 2500)  # Batch size = 1, num_points = 2500, num_features = 3

# Forward pass to get features
output = model(point_cloud_input)

# Assuming you want the attention map for the 0th class (class index 0)
attention_map = grad_cam.generate_attention_map(target_class=0)

print(attention_map.shape)  # The attention map shape should be (num_points,)
