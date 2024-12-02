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
from torch.utils.data import DataLoader

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
        self.grad_scale_factor = 2
        self.iter_step = 50
        self.config = {
            'max_iterations': 500,
            'learning_rate': 0.001,
            'epsilon': 5,  # Maximum perturbation magnitude
            'confidence': 0.5,  # Confidence threshold for successful attack
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        }
        self.max_point_dist = 1.5
        
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

        # Compute per-point distance constraints
        #point_dist_constraint = torch.full_like(adversarial_clouds, self.max_point_dist)


        best_loss = float('inf')
        best_adv_clouds = None

        for iteration in range(self.config['max_iterations']):
            # Zero out previous gradients
            self.model.zero_grad()

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
            
            # Normalize gradients
            grad = grad.contiguous()
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1)
            grad_normalized = grad / (grad_norm + 1e-8)

            # Adaptive learning rate decay
            current_lr = self.config['learning_rate'] * (0.99 ** (iteration // 100))

            with torch.no_grad():
                # Apply attention-weighted perturbation
                delta = current_lr * (
                    self.grad_scale_factor * grad_normalized * attention_map.unsqueeze(1) 
                )

                # Point-wise distance constraint
                point_wise_dist = torch.norm(delta, dim=1)  # Shape: (B, N)
                
                # Create a mask for points exceeding max distance
                dist_mask = point_wise_dist > self.max_point_dist
                
                # Scale down perturbations that exceed max distance
                # Use broadcasting to handle tensor shapes correctly
                scale_factors = torch.where(
                    dist_mask, 
                    self.max_point_dist / (point_wise_dist + 1e-8), 
                    torch.ones_like(point_wise_dist)
                ).unsqueeze(1)
                
                delta *= scale_factors

                # Adaptive perturbation scaling
                delta_norm = torch.norm(delta.view(delta.size(0), -1), dim=1).view(-1, 1, 1)
                delta_scaled = delta * (self.config['epsilon'] / (delta_norm + 1e-8))
                
                # Update adversarial point cloud
                adversarial_clouds = torch.clamp(
                    adversarial_clouds + delta_scaled, 
                    min=point_clouds - self.config['epsilon'],
                    max=point_clouds + self.config['epsilon']
                )
            
            # Track best adversarial example
            if loss < best_loss:
                best_loss = loss
                best_adv_clouds = adversarial_clouds.clone().detach()

            if (iteration % self.iter_step == 0 or iteration == self.config['max_iterations']-1):
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

        # Cross-entropy loss: Want to MINIMIZE the logits for the original class
        # and MAXIMIZE the logits for the target class
        log_probs = F.log_softmax(logits, dim=1)
        target_log_prob = log_probs[torch.arange(logits.size(0)), target_classes]

        # Original class log probability (want to minimize)
        original_preds = torch.argmax(logits, dim=1)
        original_log_prob = log_probs[torch.arange(logits.size(0)), original_preds]
        
        # We want this to be a NEGATIVE loss (high probability for target class)
        misclassification_loss = original_log_prob.mean() + (-2*target_log_prob.mean()) 
        
        # Optional: Add an entropy term to encourage uncertainty
        entropy_loss = -torch.mean(torch.sum(torch.exp(log_probs) * log_probs, dim=1))

        # Attention-guided loss component
        attention_loss = torch.mean(attention_map)
        
        # Combined loss
        total_loss = 8 * misclassification_loss + 0.5 * entropy_loss + 2.5 * attention_loss
        
        return total_loss

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

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model_performance(model, dataloader, device=None):
    """
    Evaluate model performance on a given dataset
    
    Args:
        model (nn.Module): Classification model
        dataloader (torch.utils.data.DataLoader): Dataset loader
        device (torch.device, optional): Computing device
    
    Returns:
        dict: Performance metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for points, labels in dataloader:
            points, labels = points.to(device), labels.to(device)
            
            # Get true labels
            true_label_indices = torch.argmax(labels, dim=1)
            true_labels.extend(true_label_indices.cpu().numpy())
            
            # Get predictions
            logits, _, _ = model(points)
            preds = torch.argmax(logits, dim=1)
            predicted_labels.extend(preds.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    class_report = classification_report(true_labels, predicted_labels)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

def evaluate_attack_robustness(model, attack, dataloader, target_classes=None, device=None):
    """
    Evaluate robustness of adversarial attack
    
    Args:
        model (nn.Module): Classification model
        attack (CAMPA3DAttack): Adversarial attack method
        dataloader (torch.utils.data.DataLoader): Dataset loader
        target_classes (torch.Tensor, optional): Target misclassification classes
        device (torch.device, optional): Computing device
    
    Returns:
        dict: Attack performance metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    original_labels = []
    original_preds = []
    adversarial_preds = []
    misclassification_rates = []
    
    for points, labels in dataloader:
        points, labels = points.to(device), labels.to(device)
        
        # Get original predictions
        with torch.no_grad():
            orig_logits, _, _ = model(points)
            orig_pred = torch.argmax(orig_logits, dim=1)
            original_preds.extend(orig_pred.cpu().numpy())
        
        # Generate target classes if not provided
        if target_classes is None:
            target_classes = torch.randint(
                0, model.fc3.out_features, 
                size=(points.size(0),), 
                device=device
            )
            for i in range(len(target_classes)):
                if target_classes[i] == orig_pred[i]:
                    target_classes[i] = (target_classes[i] + 1) % model.fc3.out_features
        
        # Generate adversarial examples
        adv_points = attack.generate_adversarial_point_clouds(points, target_classes)
        
        # Get adversarial predictions
        with torch.no_grad():
            adv_logits, _, _ = model(adv_points)
            adv_pred = torch.argmax(adv_logits, dim=1)
            adversarial_preds.extend(adv_pred.cpu().numpy())
        
        # Calculate misclassification rate
        misclassification = (adv_pred != orig_pred).float().mean().item()
        misclassification_rates.append(misclassification)
        
        original_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())
    
    return {
        'original_labels': original_labels,
        'original_predictions': original_preds,
        'adversarial_predictions': adversarial_preds,
        'misclassification_rate': np.mean(misclassification_rates),
        'attack_success_rate': accuracy_score(
            original_labels, 
            adversarial_preds
        )
    }

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

    attack = CAMPA3DAttack(model, grad_cam)

    # Initialize test dataset and loader
    test_dataset = PointCloudDataset(
        root_dir=config['data_dir'],
        n_points=config['num_points'],
        train=False,
        augment=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False
    )

    # Evaluate original model performance
    print("Original Model Performance:")
    orig_metrics = evaluate_model_performance(model, test_loader)
    print(f"Accuracy: {orig_metrics['accuracy']:.4f}")
    print("Confusion Matrix:\n", orig_metrics['confusion_matrix'])
    print("Classification Report:\n", orig_metrics['classification_report'])

    # Evaluate attack performance
    print("\nAdversarial Attack Performance:")
    attack_metrics = evaluate_attack_robustness(model, attack, test_loader)
    print(f"Misclassification Rate: {attack_metrics['misclassification_rate']:.4f}")
    print(f"Attack Success Rate: {attack_metrics['attack_success_rate']:.4f}")



if __name__ == "__main__":
    main()