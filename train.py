"""
Training script for Liveness Detection Models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.cnn_model import LivenessCNN
from models.efficientnet_model import EfficientNetLiveness


class LivenessDataset(Dataset):
    """
    Custom dataset for liveness detection
    Directory structure:
    data/
    ├── train/
    │   ├── live/
    │   └── fake/
    ├── val/
    │   ├── live/
    │   └── fake/
    └── test/
        ├── live/
        └── fake/
    """
    
    def __init__(self, data_dir: str, split: str = 'train', model_type: str = 'cnn'):
        self.data_dir = Path(data_dir)
        self.split = split
        self.model_type = model_type
        
        # Image size based on model
        self.img_size = 300 if model_type == 'cnn' else 224
        
        # Set transforms
        if split == 'train':
            self.transform = self._get_train_transform()
        else:
            self.transform = self._get_val_transform()
        
        # Load image paths
        self.images = []
        self.labels = []
        self._load_images()
    
    def _get_train_transform(self):
        """Training augmentation transforms"""
        return transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.4),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.2), value='random'),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_val_transform(self):
        """Validation transforms"""
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_images(self):
        """Load image paths and labels"""
        split_dir = self.data_dir / self.split
        
        # Load live images (label 1)
        live_dir = split_dir / 'live'
        if live_dir.exists():
            for img_path in live_dir.glob('*'):
                if img_path.is_file():
                    self.images.append(str(img_path))
                    self.labels.append(1)
        
        # Load fake images (label 0)
        fake_dir = split_dir / 'fake'
        if fake_dir.exists():
            for img_path in fake_dir.glob('*'):
                if img_path.is_file():
                    self.images.append(str(img_path))
                    self.labels.append(0)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image, label


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return epoch_loss, accuracy, precision, recall, f1


def train_model(model_type='cnn', data_dir='./data', num_epochs=50, batch_size=32, 
                learning_rate=0.001, save_path='./weights'):
    """
    Train a liveness detection model
    
    Args:
        model_type: 'cnn' or 'efficientnet'
        data_dir: Path to data directory
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_path: Path to save model weights
    """
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    if model_type == 'cnn':
        model = LivenessCNN(in_size=300)
    else:
        model = EfficientNetLiveness(pretrained=True)
    
    model = model.to(device)
    
    # Data loaders
    train_dataset = LivenessDataset(data_dir, split='train', model_type=model_type)
    val_dataset = LivenessDataset(data_dir, split='val', model_type=model_type)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    best_val_acc = 0
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_prec, val_recall, val_f1 = validate(model, val_loader, criterion, device)
        
        print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
        
        # Save best model
        if val_acc > best_val_acc or val_f1 > best_val_f1:
            best_val_acc = max(best_val_acc, val_acc)
            best_val_f1 = max(best_val_f1, val_f1)
            
            Path(save_path).mkdir(exist_ok=True)
            save_file = Path(save_path) / f'{model_type}_best.pth'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_f1': val_f1
            }, save_file)
            
            print(f'✓ Best model saved to {save_file}')
        
        scheduler.step()
    
    print('\n✓ Training complete!')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train liveness detection model')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'efficientnet'],
                       help='Model type to train')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--save-path', type=str, default='./weights',
                       help='Path to save weights')
    
    args = parser.parse_args()
    
    train_model(
        model_type=args.model,
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path
    )
