"""
Temporal Transformer Module for Liveness Detection
Fuses CNN embeddings with handcrafted liveness features across video frames
to stabilize confidence and improve detection of real faces under low-quality conditions.

Architecture:
1. Per-frame feature embedding: CNN + handcrafted features → 256D
2. Temporal transformer encoder: 2 layers, 4 heads, learnable positional embeddings
3. Temporal attention pooling: Learn frame importance weights
4. Classification head: Final sigmoid output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalLivenessTransformer(nn.Module):
    """
    Lightweight Transformer for temporal liveness fusion.
    
    Takes per-frame features (CNN embedding + LBP + frequency + moire + depth)
    and produces a video-level liveness probability via temporal attention.
    
    Args:
        cnn_embedding_dim: CNN backbone embedding dimension (e.g., 1280 for EfficientNet-B3)
        lbp_dim: LBP feature dimension (e.g., 768 = 256*3 scales)
        freq_dim: Frequency feature dimension (e.g., 785 = 256*3 + 16 + 1)
        moire_dim: Moiré pattern feature dimension (e.g., 29)
        depth_dim: Pseudo-depth feature dimension (e.g., 16)
        embedding_dim: Hidden dimension for frame embeddings (default 256)
        num_transformer_layers: Number of transformer encoder layers (default 2)
        num_heads: Number of attention heads (default 4)
        dropout: Dropout rate (default 0.1)
    """
    
    def __init__(
        self,
        cnn_embedding_dim: int = 1280,
        lbp_dim: int = 768,
        freq_dim: int = 785,
        moire_dim: int = 29,
        depth_dim: int = 16,
        embedding_dim: int = 256,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        
        # Calculate concatenated feature dimension
        total_feature_dim = cnn_embedding_dim + lbp_dim + freq_dim + moire_dim + depth_dim
        
        # ===== 1. Per-Frame Feature Embedding =====
        # Project all features to embedding_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(total_feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
        
        # ===== 2. Learnable Positional Embeddings =====
        # Support up to 16 frames per window
        self.positional_embedding = nn.Parameter(
            torch.randn(1, 16, embedding_dim) * 0.02
        )
        
        # ===== 3. Temporal Transformer Encoder =====
        # Multi-head attention operates across frames (temporal dimension only)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )
        
        # ===== 4. Temporal Attention Pooling =====
        # Learn importance weights for each frame
        # Key insight: Real faces have consistent micro-motion and stable features
        # Attention pooling emphasizes frames with good cues, suppresses noisy frames
        self.attention_weights = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.GELU(),
            nn.Linear(embedding_dim // 4, 1),
            nn.Softmax(dim=1)
        )
        
        # ===== 5. Classification Head =====
        # Maps pooled features to liveness probability [0, 1]
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()  # Output probability
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        frame_embeddings: torch.Tensor,
        frame_count: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for temporal liveness fusion.
        
        Args:
            frame_embeddings: Tensor of shape (B, T, D) where:
                - B: batch size
                - T: number of frames (≤ 16)
                - D: concatenated feature dimension
            frame_count: Optional tensor of shape (B,) with actual frame counts
                       for masking padded frames. If None, all frames are used.
        
        Returns:
            tuple of:
            - liveness_score: (B, 1) - video-level liveness probability
            - frame_logits: (B, T) - logits for each frame (for temporal consistency loss)
            - attention_weights: (B, T, 1) - learned importance weights per frame
        """
        batch_size, seq_len, _ = frame_embeddings.shape
        
        # ===== Step 1: Project frame features to embedding space =====
        # Each frame's features (CNN + LBP + freq + moire + depth) → 256D
        frame_embeddings = self.feature_projection(frame_embeddings)  # (B, T, 256)
        
        # ===== Step 2: Add positional embeddings =====
        # Learnable position encodings tell transformer frame order
        # This is critical: allows transformer to learn temporal patterns
        pos_emb = self.positional_embedding[:, :seq_len, :]
        frame_embeddings = frame_embeddings + pos_emb  # (B, T, 256)
        
        # ===== Step 3: Apply transformer encoder =====
        # Multi-head self-attention across frames learns:
        # - Which frames agree (consistency)
        # - Which frames are noisy (blur, artifacts)
        # - Temporal patterns (micro-motion signature of real faces)
        if frame_count is not None:
            # Create attention mask for padded frames
            mask = self._create_padding_mask(frame_count, seq_len, device=frame_embeddings.device)
            transformer_out = self.transformer_encoder(frame_embeddings, src_key_padding_mask=mask)
        else:
            transformer_out = self.transformer_encoder(frame_embeddings)  # (B, T, 256)
        
        # ===== Step 4: Compute temporal attention weights =====
        # Learn which frames matter most
        # Real faces: stable, consistent features → high weights
        # Blurry/artifact frames: noisy features → low weights
        attn_weights = self.attention_weights(transformer_out)  # (B, T, 1)
        
        # Apply softmax already in attention_weights module
        # Zero out padded frames
        if frame_count is not None:
            for b in range(batch_size):
                attn_weights[b, frame_count[b]:, :] = 0.0
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # ===== Step 5: Temporal attention pooling =====
        # Weighted average of frame embeddings
        # Real live videos: high-confidence stable frames dominate
        # Spoof videos: inconsistent frames, lower average confidence
        pooled = torch.sum(attn_weights * transformer_out, dim=1)  # (B, 256)
        
        # ===== Step 6: Classification head =====
        # Final probability prediction
        liveness_score = self.classifier(pooled)  # (B, 1)
        
        # Also compute raw logits per frame (for consistency regularization)
        frame_logits = torch.sum(attn_weights * torch.logit(torch.clamp(liveness_score, 1e-6, 1-1e-6) + 0.1), dim=1)
        
        return liveness_score, frame_logits, attn_weights
    
    def _create_padding_mask(
        self,
        frame_counts: torch.Tensor,
        max_seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create attention mask for padded frames.
        
        Args:
            frame_counts: (B,) - actual frame count per sequence
            max_seq_len: Maximum sequence length
            device: Device to create mask on
        
        Returns:
            Mask of shape (B, max_seq_len) where True = ignore (padding)
        """
        batch_size = frame_counts.shape[0]
        mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)
        
        for b in range(batch_size):
            if frame_counts[b] < max_seq_len:
                mask[b, frame_counts[b]:] = True
        
        return mask


class TemporalLivenessLoss(nn.Module):
    """
    Combined loss for temporal liveness training.
    
    Components:
    1. Classification loss (BCE) - distinguish live vs spoof
    2. Temporal consistency regularization - adjacent frames should agree
    
    This encourages the model to learn temporal consistency rather than relying
    on single-frame sharpness, which helps with low-quality videos.
    """
    
    def __init__(self, consistency_weight: float = 0.1):
        """
        Args:
            consistency_weight: Weight for temporal consistency term
        """
        super().__init__()
        self.consistency_weight = consistency_weight
        self.bce_loss = nn.BCELoss()
    
    def forward(
        self,
        liveness_scores: torch.Tensor,
        frame_logits: torch.Tensor,
        targets: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            liveness_scores: (B,) - video-level predictions
            frame_logits: (B, T) - per-frame logits
            targets: (B,) - binary labels (0=spoof, 1=live)
            attention_weights: (B, T, 1) - attention weights
        
        Returns:
            tuple of (total_loss, classification_loss, consistency_loss)
        """
        # Classification loss
        targets = targets.float().unsqueeze(1)  # (B, 1)
        classification_loss = self.bce_loss(liveness_scores, targets)
        
        # Temporal consistency regularization
        # Penalize high variance between consecutive frame logits
        # This teaches the transformer to look for stable, consistent cues
        # rather than momentary artifacts
        frame_diff = torch.abs(frame_logits[:, 1:] - frame_logits[:, :-1])
        consistency_loss = torch.mean(frame_diff)
        
        # Total loss
        total_loss = classification_loss + self.consistency_weight * consistency_loss
        
        return total_loss, classification_loss, consistency_loss


def extract_frame_features(
    image_batch: torch.Tensor,
    efficientnet_model: nn.Module,
    preprocessor,  # LivenessPreprocessor
    device: torch.device = 'cpu'
) -> torch.Tensor:
    """
    Extract combined features from a batch of face images.
    
    Args:
        image_batch: (B, 3, H, W) - normalized face images
        efficientnet_model: Loaded EfficientNet model
        preprocessor: LivenessPreprocessor instance
        device: Device to run on
    
    Returns:
        (B, D) - concatenated features for each image
    """
    with torch.no_grad():
        # Get CNN embedding (global average pooled)
        # For EfficientNet, we need to extract intermediate layer
        # This is model-specific; adjust based on your architecture
        
        # Placeholder: extract features before final classifier
        cnn_features = efficientnet_model.model.avgpool(
            efficientnet_model.model.features(image_batch)
        ).view(image_batch.size(0), -1)
        
        # Note: Handcrafted features (LBP, freq, etc.) need to be extracted
        # during preprocessing. This function assumes they're computed separately.
        
        return cnn_features
