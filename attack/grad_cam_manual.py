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

        # Set model to evaluation mode
        self.model.eval()
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        
        # Ensure consistent computation
        with torch.no_grad():
            input_points.requires_grad = True
        
        #input_points.requires_grad = True
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

    return
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d


def visualize_point_cloud_with_attention(points, attention_weights, title_prefix="Point Cloud"):
    """
    Visualize each point cloud in the batch with attention weights using Open3D.
    
    Args:
        points: Input points (B, 3, N) tensor
        attention_weights: Attention weights (B, N) tensor
        title_prefix: Prefix for visualization window titles
    """
    points = points.detach().cpu().numpy()  # Convert points to numpy (B, 3, N)
    attention_weights = attention_weights.detach().cpu().numpy()  # Convert weights to numpy (B, N)

    batch_size = points.shape[0]

    for b in range(batch_size):
        # Extract point cloud and attention weights for this batch
        point_cloud = points[b].T  # Shape: (N, 3)
        weights = attention_weights[b]  # Shape: (N,)

        # Normalize attention weights to [0, 1] for color mapping
        normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())
        
        # Create an Open3D point cloud
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud)

        # Map normalized weights to colors (e.g., "jet" colormap)
        colormap = plt.cm.get_cmap("jet")  # Use "jet" colormap for vivid colors
        colors = colormap(normalized_weights)[:, :3]  # Get RGB values (ignore alpha)
        o3d_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the point cloud
        print(f"Visualizing batch {b + 1} point cloud with attention weights...")
        o3d.visualization.draw_geometries([o3d_cloud], window_name=f"{title_prefix} {b + 1}")



from model.training import PointCloudDataset
import json
# Example usage
def main():
     # Load configuration
    with open('runs/20241125_233927/config.json', 'r') as f:
        config = json.load(f)

    # Create model and sample input
    model = PointNetClassHead(
        num_points=config['num_points'],
        num_classes=config['num_classes']
    )
    checkpoint = torch.load('runs/20241125_233927/checkpoints/final_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(torch.load('path_to_model_weights.pth'))

    # Set model to eval mode
    #model.eval()

   

    # Initialize test dataset and loader
    test_dataset = PointCloudDataset(
        root_dir=config['data_dir'],
        n_points=config['num_points'],
        train=False,
        augment=False
    )

    # Get three data points from the dataset
    points_list = []
    labels_list = []
    for i in range(1,4):  # Adjust batch size by changing this loop range
        points, label = test_dataset[i]
        points_list.append(points.unsqueeze(0))  # Add batch dimension to each point cloud
        labels_list.append(label.unsqueeze(0))  # Add batch dimension to each label

    # Combine the individual data points into a batch
    batch_points = torch.cat(points_list, dim=0)  # Shape: (3, 3, num_points)
    batch_labels = torch.cat(labels_list, dim=0)  # Shape: (3, num_classes)
    batch_targets = torch.argmax(batch_labels, dim=1)

    print(f"Batch points shape: {batch_points.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f'Batch targets: {batch_targets}')

    sample_input = torch.rand(5, 3, 1024)

    print(f"Sample input shape: {sample_input.shape}")

    sample_input = batch_points

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

    #plot_3d_attention(sample_input, attention_map, title="GradCAM 3D Attention Map")

    # Visualize attention map on point clouds using Open3D
    visualize_point_cloud_with_attention(sample_input, attention_map, title_prefix="GradCAM Attention")
    
    print(f"Predicted class: {pred_class}")
    print(f"Attention map shape: {attention_map.shape}")
    print(f"Highlighted points shape: {highlighted_points.shape}")
    print(f"Visualization data saved to 'attention_viz.txt'")

    # The highlighted_points tensor can now be used for visualization
    # It has shape (B, 4, N) where the 4th channel contains attention weights
    
if __name__ == "__main__":
    main()