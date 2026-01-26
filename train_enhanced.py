#!/usr/bin/env python
"""
Enhanced Liveness Detection Model Training
Uses improved preprocessing with LBP and frequency features
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm


class LivenessDataset(Dataset):
    """Enhanced dataset with liveness features"""
    
    def __init__(self, image_paths, labels, augment=True):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        import cv2
        from utils.liveness_features import LivenessPreprocessor
        
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Preprocess
        preprocessor = LivenessPreprocessor()
        main_tensor, lbp_tensor, freq_tensor = preprocessor.preprocess_with_liveness_features(image)
        
        return {
            'main': main_tensor.squeeze(0),
            'lbp': lbp_tensor.squeeze(0),
            'freq': freq_tensor.squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class EnhancedLivenessModel(nn.Module):
    """
    Enhanced model combining:
    1. Visual CNN (main image)
    2. LBP texture classifier
    3. Frequency domain classifier
    """
    
    def __init__(self):
        super().__init__()
        
        # Main visual branch (from LivenessCNN)
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        
        self.visual_features = nn.Sequential(
            block(3, 32),    # 300 -> 150
            block(32, 64),   # 150 -> 75
            block(64, 128),  # 75 -> 37
            block(128, 256), # 37 -> 18
            block(256, 256), # 18 -> 9
        )
        
        self.visual_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        
        # LBP texture branch (256D features from histogram)
        self.lbp_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        
        # Frequency domain branch
        freq_dim = 64 + 256 + 256  # Low + mid + high frequency
        self.freq_classifier = nn.Sequential(
            nn.Linear(freq_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
    
    def forward(self, main, lbp, freq):
        # Visual branch
        visual = self.visual_features(main)
        visual = self.visual_avgpool(visual)
        visual = torch.flatten(visual, 1)
        visual_out = self.visual_classifier(visual)
        
        # LBP branch
        lbp_out = self.lbp_classifier(lbp)
        
        # Frequency branch
        freq_out = self.freq_classifier(freq)
        
        # Fuse all predictions
        fused = torch.cat([visual_out, lbp_out, freq_out], dim=1)
        final_out = self.fusion(fused)
        
        return final_out, (visual_out, lbp_out, freq_out)


def train_enhanced_model(train_paths, train_labels, val_paths=None, val_labels=None,
                        epochs=50, batch_size=32, device='cuda'):
    """
    Train enhanced liveness model
    
    Args:
        train_paths: List of training image paths
        train_labels: List of training labels (0=Fake, 1=Live)
        val_paths: List of validation image paths (optional)
        val_labels: List of validation labels (optional)
        epochs: Number of training epochs
        batch_size: Batch size
        device: 'cuda' or 'cpu'
    """
    
    # Create model
    model = EnhancedLivenessModel().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Create datasets
    train_dataset = LivenessDataset(train_paths, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_paths is not None:
        val_dataset = LivenessDataset(val_paths, val_labels, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    else:
        val_loader = None
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            main = batch['main'].to(device)
            lbp = batch['lbp'].to(device)
            freq = batch['freq'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(main, lbp, freq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    main = batch['main'].to(device)
                    lbp = batch['lbp'].to(device)
                    freq = batch['freq'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs, _ = model(main, lbp, freq)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}")
            print(f"           Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc
                }, 'weights/enhanced_liveness_model.pt')
                print(f"✓ Model saved with validation accuracy: {val_acc:.4f}")
        else:
            print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}")
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'weights/enhanced_liveness_model.pt')
    
    return model


if __name__ == "__main__":
    print("Enhanced Liveness Detection Model Training")
    print("=" * 60)
    print("\nTo train the model, prepare your dataset and call:")
    print("  train_enhanced_model(train_paths, train_labels, val_paths, val_labels)")
    print("\nExample:")
    print("  train_paths = [...list of image paths...]")
    print("  train_labels = [...list of 0/1 labels...]")
    print("  val_paths = [...validation image paths...]")
    print("  val_labels = [...validation labels...]")
    print("\n  model = train_enhanced_model(train_paths, train_labels, val_paths, val_labels)")
