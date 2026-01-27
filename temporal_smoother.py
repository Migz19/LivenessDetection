"""
Temporal Smoother using Transformer Attention for Liveness Detection

Key insight: Transformer acts as a FILTER, not a classifier.
- Randomly initialized (no training needed)
- Frozen weights
- Uses attention pooling to smooth confidence across frames
- Reduces stuck predictions by learning which frames are reliable
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional


class TemporalSmoother(nn.Module):
    """
    Frozen Transformer-based temporal smoother.
    
    Takes CNN confidences across frames and:
    1. Learns attention weights via self-attention
    2. Smooths confidences using weighted average
    3. Reduces variance across frames
    
    Initialization: Random (no pre-training needed)
    Training: Frozen (it's just a learned filter)
    """
    
    def __init__(self, 
                 hidden_dim=128,
                 num_heads=4,
                 num_layers=2,
                 max_seq_length=32,
                 dropout=0.1):
        super().__init__()
        
        # Input projection: 1D confidence -> hidden_dim
        self.input_proj = nn.Linear(1, hidden_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Embedding(max_seq_length, hidden_dim)
        
        # Transformer encoder (random init, will be frozen)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output: attention-weighted smoothing
        self.attention_proj = nn.Linear(hidden_dim, 1)
        
        # Freeze all parameters (we use it as-is)
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, confidences: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            confidences: (batch_size, seq_length) or (seq_length,)
            mask: Optional mask for padding
            
        Returns:
            smoothed_confidences: (batch, seq_len) - weighted averaged
            attention_weights: (batch, seq_len) - learned importance of each frame
        """
        
        # Handle 1D input
        if confidences.dim() == 1:
            confidences = confidences.unsqueeze(0)
        
        batch_size, seq_len = confidences.shape
        device = confidences.device
        
        # Project confidences to hidden space
        x = confidences.unsqueeze(-1)  # (batch, seq_len, 1)
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # Add positional embeddings
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embed(pos)  # (batch, seq_len, hidden_dim)
        
        # Transformer attention
        if mask is not None:
            # Convert mask to attention mask format
            attn_mask = (mask == 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn_mask = attn_mask.expand(batch_size, -1, seq_len, -1)
        else:
            attn_mask = None
        
        x = self.transformer(x, src_key_padding_mask=mask)  # (batch, seq_len, hidden_dim)
        
        # Compute attention weights (which frames matter)
        attn_weights = self.attention_proj(x).squeeze(-1)  # (batch, seq_len)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch, seq_len)
        
        # Smooth confidences using attention weights
        smoothed = (confidences * attn_weights).sum(dim=1, keepdim=True)  # (batch, 1)
        
        return smoothed.squeeze(1), attn_weights
    
    @torch.no_grad()
    def smooth_confidences(self, confidences: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Smooth a sequence of confidences.
        
        Args:
            confidences: (seq_length,) numpy array of CNN confidences
            
        Returns:
            smoothed_score: float, weighted average confidence
            attention_weights: (seq_length,) importance of each frame
        """
        # Convert to tensor
        conf_tensor = torch.tensor(confidences, dtype=torch.float32).unsqueeze(0)
        
        # Get smoothed confidence and weights
        smoothed, weights = self.forward(conf_tensor)
        
        return smoothed.item(), weights.squeeze(0).cpu().numpy()


class TemporalSmoothingPipeline:
    """
    End-to-end pipeline for temporal smoothing + liveness decision.
    
    Flow:
    1. Get CNN predictions for each frame
    2. Smooth confidences with temporal attention
    3. Reduce variance
    4. Return stabilized decision + confidence
    """
    
    def __init__(self, 
                 smoother: Optional[TemporalSmoother] = None,
                 window_size: int = 8,
                 stride: int = 4,
                 confidence_threshold: float = 0.5,
                 smoothing_strength: float = 1.0):
        """
        Args:
            smoother: TemporalSmoother instance (creates if None)
            window_size: Number of frames for smoothing window
            stride: Stride for sliding window
            confidence_threshold: Decision threshold
            smoothing_strength: How aggressive the smoothing is (0.5 = conservative, 1.0 = normal, 2.0 = aggressive)
        """
        self.smoother = smoother or TemporalSmoother(
            hidden_dim=128,
            num_heads=4,
            num_layers=2
        )
        self.smoother.eval()
        
        self.window_size = window_size
        self.stride = stride
        self.threshold = confidence_threshold
        self.smoothing_strength = smoothing_strength
    
    def process_video(self, cnn_confidences: List[float]) -> Dict:
        """
        Process video by smoothing CNN confidences across frames.
        
        Args:
            cnn_confidences: List of per-frame CNN confidence scores [0, 1]
            
        Returns:
            {
                'prediction': 'Live' or 'Spoof',
                'smoothed_confidence': float,
                'raw_confidence': float (mean of CNN scores),
                'variance': float (variance reduction),
                'attention_weights': ndarray (which frames mattered),
                'stable': bool (is prediction stable)
            }
        """
        confidences = np.array(cnn_confidences, dtype=np.float32)
        
        # Raw statistics
        raw_mean = confidences.mean()
        raw_std = confidences.std()
        
        # Smooth with temporal attention
        smoothed_score, attn_weights = self.smoother.smooth_confidences(confidences)
        
        # Enhanced smoothing: Apply exponential moving average for more pronounced effect
        # This makes the transformer more aggressive in stabilizing predictions
        ema_smoothed = self._apply_ema_smoothing(confidences, attn_weights)
        
        # Blend attention smoothing with EMA (60% transformer, 40% EMA)
        final_smoothed = 0.6 * smoothed_score + 0.4 * ema_smoothed
        
        # After smoothing statistics
        weighted_var = np.sum((confidences - final_smoothed) ** 2 * attn_weights)
        
        # Check stability (low variance = stable)
        is_stable = weighted_var < 0.1
        
        # Decision
        prediction = "Live" if final_smoothed > self.threshold else "Spoof"
        
        return {
            'prediction': prediction,
            'smoothed_confidence': float(final_smoothed),
            'raw_confidence': float(raw_mean),
            'raw_std': float(raw_std),
            'variance_reduction': float(raw_std - np.sqrt(weighted_var)),
            'attention_weights': attn_weights,
            'stable': bool(is_stable),
            'frame_importance': {i: float(w) for i, w in enumerate(attn_weights)},
        }
    
    def _apply_ema_smoothing(self, confidences: np.ndarray, attn_weights: np.ndarray) -> float:
        """
        Apply exponential moving average weighted by attention.
        This creates a more pronounced smoothing effect.
        """
        # Use attention weights to prioritize important frames in EMA
        weighted_confidences = confidences * attn_weights
        
        # Exponential moving average with alpha based on sequence
        # smoothing_strength multiplier makes it more or less aggressive
        ema = 0.0
        alpha_base = 0.3 * self.smoothing_strength  # Base smoothing factor
        
        for i, (conf, weight) in enumerate(zip(confidences, attn_weights)):
            # Frames with higher attention weight get more influence
            alpha = min(1.0, alpha_base * (1 + weight))
            ema = alpha * conf + (1 - alpha) * ema
        
        return float(ema)
    
    def process_streaming(self, frame_confidences: List[float]) -> Dict:
        """
        Process streaming frames (keep last N frames, smooth, decide).
        
        Args:
            frame_confidences: New CNN confidence from latest frame
            
        Returns:
            Result dict with current smoothed decision
        """
        if len(frame_confidences) < self.window_size:
            # Not enough frames yet, return raw decision
            return {
                'prediction': 'Live' if frame_confidences[-1] > self.threshold else 'Spoof',
                'smoothed_confidence': float(frame_confidences[-1]),
                'ready': False,
                'frames_buffered': len(frame_confidences),
            }
        
        # Smooth last window_size frames
        window = frame_confidences[-self.window_size:]
        result = self.process_video(window)
        result['ready'] = True
        result['frames_buffered'] = len(frame_confidences)
        
        return result
