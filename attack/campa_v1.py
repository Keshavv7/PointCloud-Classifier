import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from model.pointnet import PointNetClassHead
from attack.grad_cam_manual import PointNetGradCAM
from model.training import PointCloudDataset, visualize_point_cloud_with_label
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CAMPA3DAttack:
    def __init__(self, model, grad_cam, config=None):
        """
        Initialize the CAMPA-3D Adversarial Attack

        Args:
            model (nn.Module): Target point cloud classification model
            grad_cam (PointNetGradCAM): GradCAM module for generating attention maps
            config (dict): Configuration parameters for the attack
        """
        self.model = model
        self.grad_cam = grad_cam
        self.config = {
            'max_iterations': 10000,
            'learning_rate': 0.01,
            'epsilon': 500,  # Maximum perturbation magnitude
            'confidence': 0.5,  # Confidence threshold for successful attack
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        }
        
        if config:
            self.config.update(config)
        
        self.model.to(self.config['device'])
        self.model.eval()

        # Set random seeds for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)

    def generate_adversarial_point_clouds(self, point_clouds, target_classes=None):
        """
        Generate adversarial point clouds using CAMPA-3D method

        Args:
            point_clouds (torch.Tensor): Input point clouds (B, 3, N)
            target_classes (torch.Tensor, optional): Target misclassification classes

        Returns:
            torch.Tensor: Adversarial point clouds
        """
        point_clouds = point_clouds.to(self.config['device'])

        point_clouds.requires_grad_(True)
        
        if target_classes is None:
            # If no target classes specified, choose random different classes
            current_preds = self.model(point_clouds)[0]
            current_labels = torch.argmax(current_preds, dim=1)
            target_classes = torch.randint(
                0, self.model.fc3.out_features, 
                size=(point_clouds.size(0),), 
                device=self.config['device']
            )
            target_classes[target_classes == current_labels] += 1
            target_classes %= self.model.fc3.out_features
        
        target_classes = target_classes.to(self.config['device'])
        
        print(f'Target classes: {target_classes}')

        adversarial_clouds = point_clouds.clone().detach().requires_grad_(True)

        best_loss = float('inf')
        best_adv_clouds = None

        for iteration in range(self.config['max_iterations']):
            # Ensure adversarial_clouds requires gradients for each iteration
            if not adversarial_clouds.requires_grad:
                adversarial_clouds.requires_grad_(True)

            # Generate attention map
            attention_map, _ = self.grad_cam.generate_cam(adversarial_clouds, target_classes)
            
            # Compute model outputs
            logits, _, _ = self.model(adversarial_clouds)
            
            # Compute attack loss
            loss = self._compute_attack_loss(
                logits, 
                target_classes, 
                attention_map
            )

            # Zero out previous gradients
            if adversarial_clouds.grad is not None:
                adversarial_clouds.grad.zero_()
        
            loss.backward(retain_graph=True)
            grad = adversarial_clouds.grad
            # Compute gradients
            #grad = torch.autograd.grad(loss, adversarial_clouds, retain_graph=True)[0]
            # print(f'Gradients shape: {grad.shape}')
            
            # Normalize gradients
            grad = grad.contiguous()
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1)
            grad_normalized = grad / (grad_norm + 1e-8)

            with torch.no_grad():
                # Apply attention-weighted perturbation
                delta = self.config['learning_rate'] * (
                    grad_normalized * attention_map.unsqueeze(1)
                )
                
                # Update adversarial point cloud
                adversarial_clouds = torch.clamp(
                    adversarial_clouds + delta, 
                    min=point_clouds - self.config['epsilon'],
                    max=point_clouds + self.config['epsilon']
                )
            
            # Track best adversarial example
            if loss < best_loss:
                best_loss = loss
                best_adv_clouds = adversarial_clouds.clone().detach()

            if (iteration % 400 == 0 or iteration == self.config['max_iterations']-1):
                print(f'Iter: {iteration+1}, Loss: {loss}')
            
            # Check attack success
            adv_logits, _, _ = self.model(adversarial_clouds)
            if self._check_attack_success(adv_logits, target_classes):
                break
        
        return best_adv_clouds

    def _compute_attack_loss(self, logits, target_classes, attention_map):
        """
        Compute loss for adversarial attack

        Args:
            logits (torch.Tensor): Model output logits
            target_classes (torch.Tensor): Target misclassification classes
            attention_map (torch.Tensor): Point-wise attention weights

        Returns:
            torch.Tensor: Attack loss
        """
        # Cross-entropy loss towards target classes
        ce_loss = F.cross_entropy(logits, target_classes, reduction='none')

        # Entropy-based loss to encourage misclassification
        entropy_loss = -torch.mean(F.log_softmax(logits, dim=1).sum(dim=1))

        # Attention-weighted component
        attention_loss = attention_map.mean(dim=1)
        
        # Combined loss with tunable weights
        total_loss = (
            ce_loss +  # Target class loss
            0.5 * entropy_loss +  # Entropy encourages uncertainty
            0.2 * attention_loss  # Attention guides perturbation
        )
        
        return total_loss.mean()

    def _check_attack_success(self, logits, target_classes):
        """
        Check if the attack successfully misclassifies the point cloud

        Args:
            logits (torch.Tensor): Model output logits
            target_classes (torch.Tensor): Target misclassification classes

        Returns:
            bool: Whether the attack is successful
        """
        # Compute confidence for target classes
        target_confidences = F.softmax(logits, dim=1)[
            torch.arange(logits.size(0)), 
            target_classes
        ]
        
        return (target_confidences > self.config['confidence']).all()

def visualize_adversarial_attack(model, grad_cam, points, target_classes=None):
    """
    Visualize adversarial point cloud generation process

    Args:
        model (nn.Module): Classification model
        grad_cam (PointNetGradCAM): GradCAM module
        points (torch.Tensor): Original point clouds
        target_classes (torch.Tensor, optional): Target misclassification classes
    """
    attack = CAMPA3DAttack(model, grad_cam)
    points = points.to('cuda')
    
    # Original predictions
    with torch.no_grad():
        orig_logits, _, _ = model(points)
        orig_preds = torch.argmax(orig_logits, dim=1)
    
    # Generate adversarial point clouds
    adv_points = attack.generate_adversarial_point_clouds(points, target_classes)
    
    # Adversarial predictions
    with torch.no_grad():
        adv_logits, _, _ = model(adv_points)
        adv_preds = torch.argmax(adv_logits, dim=1)
    
    print("Original Predictions:", orig_preds)
    print("Adversarial Predictions:", adv_preds)
    
    return adv_points, orig_preds, adv_preds


def display_points(dataset, point_clouds, labels):
    batch_num = range(point_clouds.size(0))  # Adjust to iterate over valid batch indices
    for b in batch_num:
        point_cloud = point_clouds[b]
        label = labels[b]
        class_idx = torch.argmax(label).item()
        label_name = dataset.classes[class_idx]
        points_numpy = point_cloud.T.cpu().numpy()

        visualize_point_cloud_with_label(points_numpy, label_name)

    return


def display_comparison(dataset, original_points, adv_points, orig_preds, adv_preds, labels):
    import open3d as o3d
    """
    Display original and adversarial point clouds side by side using Open3D.

    Args:
        dataset (PointCloudDataset): Dataset object for class names
        original_points (torch.Tensor): Original point clouds (B, 3, N)
        adv_points (torch.Tensor): Adversarial point clouds (B, 3, N)
        orig_preds (torch.Tensor): Predictions for original point clouds (B,)
        adv_preds (torch.Tensor): Predictions for adversarial point clouds (B,)
        labels (torch.Tensor): True labels (B,)
    """
    batch_size = original_points.size(0)

    for b in range(batch_size):
        # Convert point clouds to numpy
        orig_cloud = original_points[b].T.cpu().numpy()  # Transpose to shape (N, 3)
        adv_cloud = adv_points[b].T.cpu().numpy()  # Transpose to shape (N, 3)
        
        # Prepare labels
        orig_label = labels[b]
        orig_pred_label = dataset.classes[orig_preds[b].item()]
        adv_pred_label = dataset.classes[adv_preds[b].item()]
        true_label_name = dataset.classes[torch.argmax(orig_label).item()]
        
        # Create Open3D point clouds
        orig_pcd = o3d.geometry.PointCloud()
        orig_pcd.points = o3d.utility.Vector3dVector(orig_cloud)
        orig_pcd.paint_uniform_color([0, 0, 1])  # Blue for original cloud
        
        adv_pcd = o3d.geometry.PointCloud()
        adv_pcd.points = o3d.utility.Vector3dVector(adv_cloud)
        adv_pcd.paint_uniform_color([1, 0, 0])  # Red for adversarial cloud
        
        # Create visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Comparison for Sample {b}")
        
        # Add point clouds to visualization
        vis.add_geometry(orig_pcd)
        vis.add_geometry(adv_pcd)
        
        # Add coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0])
        vis.add_geometry(coord_frame)
        
        # Set up camera view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.6)
        ctr.set_front([0.4, -0.4, -0.8])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        
        # Render and add annotations
        opt = vis.get_render_option()
        opt.point_size = 3.0
        
        # Create custom annotation
        annotation_text = (
            f"True Label: {true_label_name}\n"
            f"Original Pred: {orig_pred_label}\n"
            f"Adversarial Pred: {adv_pred_label}"
        )
        
        # Render initial view
        vis.update_geometry(orig_pcd)
        vis.update_geometry(adv_pcd)
        vis.update_renderer()
        
        # Run visualization
        vis.run()
        vis.destroy_window()
    
    return


# Usage example (requires existing PointNetClassHead and PointNetGradCAM)
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

    grad_cam = PointNetGradCAM(model)


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

    sample_input = batch_points

    target_classes = torch.tensor([7, 7, 7])
    
    # Run adversarial attack visualization
    adv_points, orig_preds, adv_preds = visualize_adversarial_attack(model, grad_cam, sample_input,target_classes=target_classes)

    print(f'Perturbed Clouds shape: {adv_points.shape}')

    display_points(test_dataset, adv_points, batch_labels)

    # Visualize comparison of original and adversarial clouds
    #display_comparison(test_dataset, sample_input, adv_points, orig_preds, adv_preds, batch_labels)


if __name__ == "__main__":
    main()