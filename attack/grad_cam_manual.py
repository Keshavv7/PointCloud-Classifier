import torch
import numpy as np
from model.pointnet import *

class PointNetGradCAM:
    def __init__(self, model):
        """
        Initialize GradCAM for PointNet
        Args:
            model: PointNetClassHead instance
            target_layer: Name of the target layer for GradCAM visualization
        """
        self.model = model
        self.gradients = None
        self.features = None
        
        # # Register hooks
        # def save_gradients(grad):
        #     print("Gradient hook called!")
        #     self.gradients = grad.detach()

        # Register a backward hook on the PointNet module
        def backward_hook(module, grad_input, grad_output):
            # Capture the gradient of the local feature map
            if len(grad_input) > 0 and grad_input[0] is not None:
                self.gradients = grad_input[0].detach()
            
        def forward_hook(module, input, output):
            # Store the local features
            if isinstance(module, PointNet):
                self.features = module.local_feature_map.detach()
            
        # Find the specific layer to hook
        for name, module in self.model.pointnet.named_children():
            if name == 'bn3':  # Hook right after the second convolution
                module.register_backward_hook(backward_hook)

                # Register hook on the main model's forward pass
                # self.model.pointnet.register_full_backward_hook(backward_hook)
                self.model.pointnet.register_forward_hook(forward_hook)


    
    def generate_cam(self, input_points, target_class=None):
        """
        Generate GradCAM attention map for the input points
        Args:
            input_points: Input point cloud tensor (B, 3, N)
            target_class: Target class index. If None, uses the model's prediction
        Returns:
            cam: GradCAM attention map (B, N)
            pred_class: Predicted class
        """
        
        input_points.requires_grad = True
        batch_size = input_points.size(0)

        # Forward pass
        logits, _, _ = self.model(input_points)
        
        # If target_class is not specified, use the predicted class
        if target_class is None:
            target_class = logits.argmax(dim=1)
        
        # Create one-hot encoding for each sample in the batch
        one_hot = torch.zeros((batch_size, logits.size(-1)), device=input_points.device)
        one_hot.scatter_(1, target_class.view(-1, 1), 1.0)

        # Zero the gradients before backprop
        self.model.zero_grad()
        
        # Backward pass
        logits.backward(gradient=one_hot, retain_graph=True)

        # Add these prints in generate_cam() after the backward pass:
        print("Features shape:", self.features.shape if self.features is not None else None)
        print("Gradients shape:", self.gradients.shape if self.gradients is not None else None)

        # Ensure we have gradients and features
        if self.gradients is None or self.features is None:
            raise ValueError("Gradients or features are None. Check if hooks are properly registered.")
        
        # Get gradients and features
        gradients = self.gradients
        features = self.features
        
        # Calculate weights (global average pooling of gradients)
        alpha = torch.mean(gradients, dim=(2))  # Shape: (B, C)        
        # Reshape weights for broadcasting
        alpha = alpha.view(batch_size, -1, 1)  # Shape: (B, C, 1)        
        # Generate weighted combination of feature maps
        weighted_features = alpha * features  # Shape: (B, C, N)
        # Sum along the channel dimension
        cam = torch.sum(weighted_features, dim=1)  # Shape: (B, N)
        # Apply ReLU to focus on features that have a positive influence
        cam = torch.relu(cam)
        
        # Normalize CAM
        if cam.sum() != 0:  # Check if CAM is not all zeros
            cam_min = cam.min(dim=1, keepdim=True)[0]
            cam_max = cam.max(dim=1, keepdim=True)[0]
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-7)

        
        return cam, target_class

def visualize_attention(points, attention_weights, threshold=0.5):
    """
    Helper function to create a visualization-friendly format of points colored by attention weights
    Args:
        points: Input points (B, 3, N)
        attention_weights: Attention weights from GradCAM (B, N)
        threshold: Threshold for highlighting high-attention points
    Returns:
        highlighted_points: Points with an additional channel for attention weights (B, 4, N)
    """
    batch_size = points.size(0)
    num_points = points.size(2)
    
    # Create output tensor with additional channel for attention weights
    highlighted_points = torch.cat([
        points,
        attention_weights.unsqueeze(1)
    ], dim=1)
    
    return highlighted_points

def save_attention_visualization(points, attention_weights, output_file="attention_viz.txt"):
    """
    Save the points with their attention weights to a file for visualization
    Args:
        points: Input points (B, 3, N)
        attention_weights: Attention weights from GradCAM (B, N)
        output_file: Path to save the visualization data
    """
    # Combine points with attention weights
    points = points.transpose(2, 1).detach().cpu().numpy()  # (B, N, 3)
    attention_weights = attention_weights.detach().cpu().numpy()  # (B, N)

    # Iterate through each batch and save individually
    with open(output_file, 'w') as f:
        f.write('x y z attention\n')  # Write header
        for b in range(points.shape[0]):  # Process each batch
            batch_points = points[b]  # (N, 3)
            batch_weights = attention_weights[b]  # (N,)
            
            # Combine points and weights
            viz_data = np.column_stack([batch_points, batch_weights])  # (N, 4)
            
            # Save each batch to the file
            np.savetxt(f, viz_data, fmt='%.6f')


# Example usage
def main():
    # Create model and sample input
    model = PointNetClassHead(num_classes=10)
    # model.load_state_dict(torch.load('path_to_model_weights.pth'))

    # Set model to eval mode
    #model.eval()

    sample_input = torch.rand(5, 3, 2500)

    print("Running GradCAM...\n\n")
    
    # Initialize GradCAM
    grad_cam = PointNetGradCAM(model)

    # Generate attention map
    with torch.enable_grad():  # Ensure gradients are enabled
        attention_map, pred_class = grad_cam.generate_cam(sample_input)
    
    # Visualize results
    highlighted_points = visualize_attention(sample_input, attention_map)
    
     # Save visualization data
    save_attention_visualization(sample_input, attention_map)
    
    print(f"Predicted class: {pred_class}")
    print(f"Attention map shape: {attention_map.shape}")
    print(f"Highlighted points shape: {highlighted_points.shape}")
    print(f"Visualization data saved to 'attention_viz.txt'")

    # The highlighted_points tensor can now be used for visualization
    # It has shape (B, 4, N) where the 4th channel contains attention weights
    
if __name__ == "__main__":
    main()