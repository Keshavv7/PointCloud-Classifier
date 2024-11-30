# trainer.py
import torch
from model.loss import PointNetLoss
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import json
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from model.training import PointCloudDataset, PointNetClassHead, setup_logging, evaluate_model, visualize_point_cloud_with_label

def test_model():
    """Function to test the trained model"""
    # Load configuration
    with open('runs/20241125_233927/config.json', 'r') as f:
        config = json.load(f)

    # Set up logging
    logger = setup_logging('test_logs')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Initialize test dataset and loader
    test_dataset = PointCloudDataset(
        root_dir=config['data_dir'],
        n_points=config['num_points'],
        train=False,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    # Initialize model
    model = PointNetClassHead(
        num_points=config['num_points'],
        num_classes=config['num_classes']
    )
    model.to(device="cuda")

    # Load trained model
    checkpoint = torch.load('runs/20241125_233927/checkpoints/final_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info('Loaded trained model')

    # Evaluate
    accuracy = evaluate_model(model, test_loader, device, logger)
    print(f'Accuracy: {accuracy}')
    logger.info(f'Test Accuracy: {accuracy}')

    # Test first item
    points, label = test_dataset[345]
    print(f"Points shape: {points.shape}")
    print(f"Label shape: {label.shape}")

    # Add a batch dimension to the points
    points = points.unsqueeze(0)  # Shape becomes (1, 3, 1024)
    print(f"Points with batch dimension shape: {points.shape}")


    # Move points and model to the same device
    points = points.to(device)
    label = label.to(device)
    model.to(device)

    # Forward pass
    outputs, _, _ = model(points)  # Shape: (batch_size=1, num_classes)
    _, predicted = outputs.max(1)  # Get the class index with the highest score
    print("Predicted class index: ", predicted.item())

    # Compare with the label
    class_idx = torch.argmax(label).item()  # Get ground truth class index
    label_name = test_dataset.classes[class_idx]
    predicted_name = test_dataset.classes[predicted.item()]

    # Log and visualize
    print(f"Ground truth: {label_name}")
    print(f"Predicted: {predicted_name}")

    # Convert points to numpy format for visualization
    points_numpy = points.squeeze(0).T.cpu().numpy()  # Back to (n_points, 3) format

    # Visualize the point cloud with the predicted and true labels
    visualize_point_cloud_with_label(points_numpy, f"True: {label_name}, Predicted: {predicted_name}")

    return 

accuracy = test_model()


