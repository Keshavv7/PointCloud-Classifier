import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..model.pointnet import PointNet


# Grad-CAM class for generating attention maps for 3D point clouds
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def get_attention_map(self, x, class_idx=None):
        # Step 1: Get the feature map and the gradients
        features, logits = self.model(x)

        # Step 2: Set the gradient hook for the feature map
        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]  # Gradients of the output wrt the feature map

        # Hook the gradients to the last fully connected layer
        hook = self.model.fc2.register_backward_hook(save_gradients)
        
        # Step 3: Backpropagate to get the gradients for the class we are interested in
        if class_idx is None:
            class_idx = logits.argmax(dim=1)  # Default to the predicted class
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(logits)
        one_hot_output[0][class_idx] = 1
        logits.backward(gradient=one_hot_output, retain_graph=True)

        # Step 4: Pool the gradients and feature maps to get the attention map
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2], keepdim=True)
        weighted_features = features * pooled_gradients
        attention_map = torch.sum(weighted_features, dim=1, keepdim=True)
        attention_map = F.relu(attention_map)  # Only positive parts are important
        
        # Normalize the attention map to [0, 1]
        attention_map = F.interpolate(attention_map, size=x.size(2), mode='linear', align_corners=False)
        attention_map = attention_map.squeeze(0).cpu().detach().numpy()
        
        # Step 5: Return the attention map (scaled and normalized)
        return attention_map

# Main execution
if __name__ == '__main__':
    # Initialize the model and Grad-CAM
    model = PointNet(local_feat=True)
    grad_cam = GradCAM(model)
    
    # Simulate a batch of point cloud data (batch_size x num_features x num_points)
    # Here num_points is 2048 for ShapeNet dataset and num_features is 64
    x = torch.randn(1, 3, 2048)  # Simulate 1 point cloud with 2048 points, each with 3 coordinates (x, y, z)
    
    # Get the attention map for a specific class (default: predicted class)
    attention_map = grad_cam.get_attention_map(x)
    
    print(f"Attention map shape: {attention_map.shape}")