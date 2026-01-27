"""
Training script for Temporal Liveness Transformer

Demonstrates:
1. Feature extraction pipeline (CNN + handcrafted features)
2. Frame windowing for video sequences
3. Heavy augmentation (blur, compression, downscale) for robustness
4. Training loop with temporal consistency regularization
5. Validation with confidence calibration

Key principle: Train on degraded videos so model learns temporal consistency
rather than relying on frame sharpness (which fails with motion blur, low quality).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from typing import Tuple, List, Optional
from pathlib import Path

from models.temporal_transformer import TemporalLivenessTransformer, TemporalLivenessLoss
from models.efficientnet_model import EfficientNetLiveness, load_efficientnet_model
from utils.liveness_features import LivenessPreprocessor


class VideoLivenessDataset(Dataset):
    """
    Dataset for video-level liveness detection.
    
    Loads video frames, extracts liveness features, and packages them
    as windowed sequences for transformer training.
    """
    
    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        window_size: int = 12,
        stride: int = 6,
        max_frames: int = None,
        augment: bool = True,
    ):
        """
        Args:
            video_paths: List of paths to video files
            labels: List of binary labels (0=spoof, 1=live)
            window_size: Number of frames per window (8-16 recommended)
            stride: Stride for sliding window
            max_frames: Maximum frames to load per video (None = all)
            augment: Apply heavy augmentation for robustness
        """
        self.video_paths = video_paths
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.max_frames = max_frames
        self.augment = augment
        
        self.preprocessor = LivenessPreprocessor(model_type='cnn')
        
        # Pre-load and cache frames/features
        self.windows = []
        self._build_windows()
    
    def _build_windows(self):
        """Pre-process videos and create windowed sequences."""
        for video_path, label in zip(self.video_paths, self.labels):
            frames = self._load_video_frames(video_path)
            
            if frames is None or len(frames) < self.window_size:
                continue
            
            # Create sliding windows
            for start_idx in range(0, len(frames) - self.window_size + 1, self.stride):
                end_idx = start_idx + self.window_size
                window_frames = frames[start_idx:end_idx]
                
                # Extract features for window
                try:
                    features = self._extract_window_features(window_frames)
                    self.windows.append({
                        'features': features,
                        'label': label,
                        'frame_count': len(window_frames)
                    })
                except Exception as e:
                    print(f"Error processing window: {e}")
                    continue
    
    def _load_video_frames(self, video_path: str) -> Optional[List[np.ndarray]]:
        """Load frames from video file."""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                
                if self.max_frames and len(frames) >= self.max_frames:
                    break
            
            cap.release()
            return frames if frames else None
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None
    
    def _extract_window_features(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Extract CNN + handcrafted features for each frame."""
        window_features = []
        
        for frame in frames:
            # Get liveness features
            face_img, lbp, freq, moire, depth = self.preprocessor.preprocess_with_liveness_features(frame)
            
            # For CNN features: would extract from EfficientNet
            # For now, flatten the face tensor
            face_flat = face_img.flatten()  # This is placeholder
            
            # Concatenate all features
            # Adjust dimensions based on actual feature extractor
            all_features = torch.cat([
                face_flat.unsqueeze(0),  # CNN embedding (mock)
                lbp,
                freq,
                moire,
                depth
            ], dim=1)
            
            window_features.append(all_features)
        
        # Stack frames: (T, feature_dim)
        window_tensor = torch.cat(window_features, dim=0)
        return window_tensor
    
    def _augment_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply heavy degradations to teach temporal consistency."""
        if not self.augment:
            return frame
        
        # Randomly apply augmentations
        if np.random.random() < 0.5:
            # Motion blur
            size = np.random.randint(5, 15)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            frame = cv2.filter2D(frame, -1, kernel)
        
        if np.random.random() < 0.5:
            # Gaussian blur
            kernel_size = np.random.choice([3, 5, 7])
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        
        if np.random.random() < 0.5:
            # JPEG compression
            quality = np.random.randint(30, 80)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        
        if np.random.random() < 0.5:
            # Downscale -> upscale
            h, w = frame.shape[:2]
            new_h, new_w = h // 2, w // 2
            frame = cv2.resize(frame, (new_w, new_h))
            frame = cv2.resize(frame, (w, h))
        
        if np.random.random() < 0.3:
            # Random frame dropping (simulates low FPS)
            # Mark frame as "dropped" by reducing intensity
            frame = (frame * 0.8).astype(np.uint8)
        
        return frame
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            - features: (T, D) - sequence of frame features
            - label: scalar - video label
            - frame_count: scalar - actual frame count (for padding mask)
        """
        window = self.windows[idx]
        features = window['features'].float()  # (T, D)
        label = torch.tensor(window['label'], dtype=torch.float)
        frame_count = torch.tensor(window['frame_count'], dtype=torch.long)
        
        # Pad to fixed window size if needed
        if features.shape[0] < self.window_size:
            pad_size = self.window_size - features.shape[0]
            padding = torch.zeros(pad_size, features.shape[1])
            features = torch.cat([features, padding], dim=0)
        
        return features, label, frame_count


def train_temporal_transformer(
    model: TemporalLivenessTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device = 'cpu',
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
):
    """
    Training loop for temporal liveness transformer.
    
    Key points:
    1. Video-level labels (apply to all frames in sequence)
    2. Binary cross entropy loss + temporal consistency regularization
    3. Validation uses confidence calibration
    """
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = TemporalLivenessLoss(consistency_weight=0.1)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # ===== Training Phase =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (features, labels, frame_counts) in enumerate(train_loader):
            features = features.to(device)  # (B, T, D)
            labels = labels.to(device)      # (B,)
            frame_counts = frame_counts.to(device)  # (B,)
            
            # Forward pass
            liveness_scores, frame_logits, attn_weights = model(features, frame_counts)
            
            # Compute loss
            loss, class_loss, consistency_loss = loss_fn(
                liveness_scores.squeeze(),
                frame_logits,
                labels,
                attn_weights
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            predictions = (liveness_scores.squeeze() > 0.5).long()
            train_correct += (predictions == labels.long()).sum().item()
            train_total += labels.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Class Loss: {class_loss.item():.4f} | "
                      f"Consistency Loss: {consistency_loss.item():.4f}")
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # ===== Validation Phase =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels, frame_counts in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                frame_counts = frame_counts.to(device)
                
                # Forward pass
                liveness_scores, frame_logits, attn_weights = model(features, frame_counts)
                
                # Loss
                loss, _, _ = loss_fn(
                    liveness_scores.squeeze(),
                    frame_logits,
                    labels,
                    attn_weights
                )
                
                val_loss += loss.item()
                
                # Confidence calibration: if variance is high, reduce confidence
                frame_variance = torch.var(attn_weights, dim=1)
                confidence_adjusted = liveness_scores.squeeze() * (1 - 0.3 * frame_variance.squeeze())
                
                predictions = (confidence_adjusted > 0.5).long()
                val_correct += (predictions == labels.long()).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'temporal_transformer_best.pt')
            print(f"Model saved with validation accuracy: {val_acc:.2f}%")
    
    return model


if __name__ == '__main__':
    # Example usage (requires actual video data)
    print("Temporal Liveness Transformer Training Script")
    print("=" * 50)
    print("This script demonstrates:")
    print("1. Feature extraction (CNN + handcrafted)")
    print("2. Video windowing for sequences")
    print("3. Heavy augmentation (blur, compression, etc.)")
    print("4. Training with temporal consistency loss")
    print("5. Validation with confidence calibration")
    print("=" * 50)
    
    # Initialize model
    model = TemporalLivenessTransformer(
        cnn_embedding_dim=1280,  # EfficientNet-B3 global average pool
        lbp_dim=768,
        freq_dim=785,
        moire_dim=29,
        depth_dim=16,
        embedding_dim=256,
        num_transformer_layers=2,
        num_heads=4,
        dropout=0.1,
    )
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # To train with actual data:
    # 1. Collect video paths and labels
    # video_paths = [...]  # List of video file paths
    # labels = [...]        # Binary labels
    #
    # 2. Create datasets
    # train_dataset = VideoLivenessDataset(video_paths[:split], labels[:split], augment=True)
    # val_dataset = VideoLivenessDataset(video_paths[split:], labels[split:], augment=False)
    #
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    #
    # 3. Train
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = train_temporal_transformer(model, train_loader, val_loader, device, num_epochs=50)
