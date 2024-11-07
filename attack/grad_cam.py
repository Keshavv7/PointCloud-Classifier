import torch
import torch.nn.functional as F

class PointNetGradCAM:
    def __init__(self, model, target_layer):
        """
        Initializes the Grad-CAM method for PointNet.
        Args:
            model: The trained PointNet model.
            target_layer: The layer to extract gradients from (e.g., feature layer before max pooling).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Register hooks to get gradients and activations from the target layer
        self._register_hooks()

    def _register_hooks(self):
        """
        Registers hooks to capture gradients and activations in the target layer.
        """
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output

        # Attach hooks to target layer
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_attention_map(self, input_point_cloud, target_class):
        """
        Generates a Grad-CAM based attention map for the given input and target class.
        Args:
            input_point_cloud: Input 3D point cloud data of shape (batch_size, num_points, num_features).
            target_class: Class index for which the attention map is generated.
        
        Returns:
            Binary attention map of shape (batch_size, num_features) indicating important regions.
        """
        # Forward pass
        output = self.model(input_point_cloud)  # Shape: (batch_size, num_classes)
        loss = output[:, target_class].sum()    # Sum over batch for a specific target class

        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        # Apply global average pooling on the gradients to obtain importance weights
        weights = torch.mean(self.gradients, dim=2, keepdim=True)  # Shape: (batch_size, num_features, 1)

        # Compute weighted activations
        weighted_activations = weights * self.activations  # Shape: (batch_size, num_features, num_points)
        
        # Sum across feature dimension to get attention map
        attention_map = torch.sum(weighted_activations, dim=1)  # Shape: (batch_size, num_points)

        # Normalize attention map to [0, 1] and binarize based on threshold
        attention_map = F.relu(attention_map)  # Remove negative values
        attention_map -= attention_map.min((1), keepdim=True)[0]  # Min-max normalization
        attention_map /= (attention_map.max((1), keepdim=True)[0] + 1e-5)
        
        # Binarize attention map with a threshold (e.g., 0.2)
        binary_attention_map = (attention_map > 0.2).float()  # Shape: (batch_size, num_points)

        return binary_attention_map


# Example usage
# Assume `model` is a PointNet model with a layer `model.feature_layer` before pooling
# and input_point_cloud is a tensor of shape (batch_size, num_points, num_features)

model = None  # Load your trained PointNet model
target_layer = model.features_cls  # Define the target layer from PointNet

# Initialize Grad-CAM for PointNet
grad_cam = PointNetGradCAM(model, target_layer)

# Generate binary attention map
input_point_cloud = torch.randn(8, 1024, 3)  # Example input: batch of 8 point clouds, 1024 points each, 3 features (x, y, z)
target_class = 1  # Example target class index

# Obtain binary attention map for the given input and target class
binary_attention_map = grad_cam.generate_attention_map(input_point_cloud, target_class)
print("Binary Attention Map Shape:", binary_attention_map.shape)  # Should be (batch_size, num_points)
