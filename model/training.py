import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import trimesh
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json
import wandb  
from model.pointnet import *

class PointCloudDataset(Dataset):
    def __init__(self, root_dir, n_points=1024, train=True, augment=True):
        self.n_points = n_points
        self.augment = augment and train
        
        # Load and preprocess all files
        self.samples = []
        self.labels = []
        
        split = 'train' if train else 'test'
        self.classes = sorted(os.listdir(root_dir))
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name, split)
            for file_name in os.listdir(class_path):
                self.samples.append(os.path.join(class_path, file_name))
                self.labels.append(class_idx)

        # Convert labels to numpy array for proper indexing
        self.labels = np.array(self.labels, dtype=np.int64)
    
    def normalize_point_cloud(self, points):
        # Center the point cloud
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Scale to unit sphere
        furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / furthest_distance
        
        return points
    
    def random_point_dropout(self, points):
        points_idx = np.arange(points.shape[0])
        np.random.shuffle(points_idx)
        return points[points_idx[:int(0.7 * points.shape[0])]]
    
    def random_scale_point_cloud(self, points, scale_low=0.8, scale_high=1.25):
        scale = np.random.uniform(scale_low, scale_high)
        return points * scale
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
            # Ensure idx is an integer scalar
            if isinstance(idx, torch.Tensor):
                idx = idx.item()

            # Load point cloud
            mesh = trimesh.load(self.samples[idx])
            points = np.array(mesh.sample(self.n_points))
            
            # Normalize point cloud
            points = self.normalize_point_cloud(points)
            
            # Data augmentation for training
            if self.augment:
                # Random rotation around z-axis
                theta = np.random.uniform(0, 2*np.pi)
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                points = points @ rotation_matrix
                
                # Random jittering
                points += np.random.normal(0, 0.02, size=points.shape)
                
                # Random scaling
                points = self.random_scale_point_cloud(points)
                
                # Random point dropout
                if np.random.random() > 0.5:
                    points = self.random_point_dropout(points)
                    # Resample to maintain consistent point count
                    if points.shape[0] < self.n_points:
                        idx = np.random.choice(points.shape[0], self.n_points, replace=True)
                        points = points[idx]
            
            # Ensure we have exactly n_points
            if points.shape[0] != self.n_points:
                idx = np.random.choice(points.shape[0], self.n_points, replace=True)
                points = points[idx]
            
            # Convert to tensor
            points = torch.FloatTensor(points)
            # Create one-hot encoded label
            label = torch.zeros(len(self.classes))
            # print("Element of labels: ", self.labels[idx])
            # print("Shape of labels: ", (self.labels[idx]).shape)
            # label[int(self.labels[idx])] = 1
            label[self.labels[idx]] = 1
            
            # Ensure points tensor is of shape (3, N)
            points = points.transpose(0, 1) if points.shape[0] != 3 else points
            
            return points, label
            
        except Exception as e:
            print(f"Error loading file {self.samples[idx]}: {str(e)}")
            # Return a zero tensor of correct shape in case of error
            return torch.zeros(3, self.n_points), torch.zeros(len(self.classes))


def train_pointnet(model, train_loader, val_loader, num_epochs=20, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (points, targets) in enumerate(train_loader):
            points, targets = points.to(device), targets.to(device)
            
            # Convert one-hot to class indices
            targets = torch.argmax(targets, dim=1)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _, transform_matrix = model(points)
            
            # Calculate main loss
            loss = criterion(outputs, targets)
            
            # Add orthogonal regularization loss for the feature transform
            if transform_matrix is not None:
                I = torch.eye(64).to(device)
                batch_size = transform_matrix.size(0)
                mat_diff = torch.bmm(transform_matrix, transform_matrix.transpose(1, 2)) - I
                reg_loss = torch.mean(torch.norm(mat_diff, dim=(1, 2)))
                loss += 0.001 * reg_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for points, targets in val_loader:
                points, targets = points.to(device), targets.to(device)
                targets = torch.argmax(targets, dim=1)
                
                outputs, _, _ = model(points)
                loss = criterion(outputs, targets)
                
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }, f'checkpoints/pointnet_best.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
        
        print(f'Epoch {epoch}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%') 


def setup_logging(log_dir):
    """Set up logging configuration"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_config(config, log_dir):
    """Save configuration to JSON file"""
    with open(f'{log_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=4)

def main():
    # Configuration
    config = {
        'data_dir': 'datasets/ModelNet10',
        'log_dir': 'runs/' + datetime.now().strftime('%Y%m%d_%H%M%S'),
        'num_points': 1024,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 0.05,
        'num_workers': 4,
        'num_classes': 10,  # ModelNet10 has 10 classes
        'use_wandb': False,  # Set to True if using wandb
    }

    # Set up logging
    logger = setup_logging(config['log_dir'])
    save_config(config, config['log_dir'])

    # Create checkpoint directory
    checkpoint_dir = Path(config['log_dir']) / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Initialize datasets
    train_dataset = PointCloudDataset(
        root_dir=config['data_dir'],
        n_points=config['num_points'],
        train=True,
        augment=True
    )
    
    val_dataset = PointCloudDataset(
        root_dir=config['data_dir'],
        n_points=config['num_points'],
        train=False,
        augment=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    logger.info(f'Train dataset size: {len(train_dataset)}')
    logger.info(f'Validation dataset size: {len(val_dataset)}')

    # Initialize model
    model = PointNetClassHead(
        num_points=config['num_points'],
        num_classes=config['num_classes']
    ).to(device)

    # Optional: Initialize wandb
    if config['use_wandb']:
        wandb.init(
            project="pointnet-classification",
            config=config
        )
        wandb.watch(model)

    # Load checkpoint if exists
    checkpoint_path = checkpoint_dir / 'latest.pth'
    start_epoch = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f'Resumed from epoch {start_epoch}')

    # Train the model
    try:
        train_pointnet(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['num_epochs'],
            device=device,
            checkpoint_dir=checkpoint_dir,
            logger=logger,
            use_wandb=config['use_wandb']
        )
    except KeyboardInterrupt:
        logger.info('Training interrupted by user')
    except Exception as e:
        logger.exception(f'Training failed with error: {str(e)}')
    finally:
        # Save final model
        torch.save({
            'epoch': config['num_epochs'],
            'model_state_dict': model.state_dict(),
        }, checkpoint_dir / 'final_model.pth')
        logger.info('Saved final model')

        if config['use_wandb']:
            wandb.finish()

def evaluate_model(model, test_loader, device, logger):
    """Evaluate the model on the test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for points, targets in test_loader:
            points = points.to(device)
            targets = torch.argmax(targets, dim=1).to(device)
            
            outputs, _, _ = model(points)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    logger.info(f'Test accuracy: {accuracy:.2f}%')
    return accuracy

def test():
    """Function to test the trained model"""
    # Load configuration
    with open('runs/latest/config.json', 'r') as f:
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
    ).to(device)

    # Load trained model
    checkpoint = torch.load('runs/latest/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info('Loaded trained model')

    # Evaluate
    accuracy = evaluate_model(model, test_loader, device, logger)
    return accuracy

def test_dataset():
    """Test the dataset implementation"""
    dataset = PointCloudDataset(
        root_dir='datasets/ModelNet10',
        n_points=1024,
        train=True,
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test first item
    points, label = dataset[0]
    print(f"Points shape: {points.shape}")
    print(f"Label shape: {label.shape}")
    
    # Test data loader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    for batch_idx, (points, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: Points shape: {points.shape}, Labels shape: {labels.shape}")
        break

if __name__ == '__main__':
    test_dataset()
    #main()
    # Uncomment to run testing after training
    # test()