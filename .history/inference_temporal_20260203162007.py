# """
# Inference module for Temporal Liveness Transformer

# Implements:
# 1. Sliding window frame extraction from video
# 2. Per-window transformer inference
# 3. Confidence calibration based on temporal variance
# 4. Real-time streaming compatible
# """

# import torch
# import torch.nn as nn
# import cv2
# import numpy as np
# from typing import Tuple, Optional, List, Dict
# from collections import deque
# from pathlib import Path

# from models.temporal_transformer import TemporalLivenessTransformer
# from models.efficientnet_model import EfficientNetLiveness, load_efficientnet_model
# from utils.liveness_features import LivenessPreprocessor


# class TemporalLivenessInference:
#     """
#     Inference pipeline for temporal liveness detection.
    
#     Processes video frames using sliding windows and combines predictions
#     with confidence calibration based on temporal consistency.
#     """
    
#     def __init__(
#         self,
#         transformer_model: TemporalLivenessTransformer,
#         efficientnet_model: EfficientNetLiveness,
#         device: torch.device = 'cpu',
#         window_size: int = 12,
#         stride: int = 4,
#     ):
#         """
#         Args:
#             transformer_model: Trained TemporalLivenessTransformer
#             efficientnet_model: EfficientNet backbone
#             device: Device for inference
#             window_size: Number of frames per window (8-16 recommended)
#             stride: Frame stride for sliding window
#         """
#         self.transformer = transformer_model.to(device).eval()
#         self.efficientnet = efficientnet_model.to(device).eval()
#         self.device = device
#         self.window_size = window_size
#         self.stride = stride
        
#         self.preprocessor = LivenessPreprocessor(model_type='cnn')
    
#     def process_video(
#         self,
#         video_path: str,
#         max_frames: int = None,
#         return_details: bool = False,
#     ) -> Tuple[float, float]:
#         """
#         Process entire video and return liveness prediction.
        
#         Args:
#             video_path: Path to video file
#             max_frames: Maximum frames to process (None = all)
#             return_details: If True, also return confidence details
        
#         Returns:
#             (liveness_score, confidence)
#             - liveness_score: [0, 1] - probability of liveness
#             - confidence: [0, 1] - confidence (high if temporal variance is low)
#         """
        
#         # Load video frames
#         frames = self._load_video_frames(video_path, max_frames)
        
#         if frames is None or len(frames) < self.window_size:
#             raise ValueError(f"Video has fewer than {self.window_size} frames")
        
#         # Extract features for all frames
#         frame_features = []
#         with torch.no_grad():
#             for frame in frames:
#                 features = self._extract_frame_features(frame)
#                 frame_features.append(features)
        
#         # Process sliding windows
#         window_scores = []
#         window_variances = []
        
#         for start_idx in range(0, len(frame_features) - self.window_size + 1, self.stride):
#             end_idx = start_idx + self.window_size
#             window_features = frame_features[start_idx:end_idx]
            
#             # Stack into (1, T, D) batch
#             window_tensor = torch.stack(window_features, dim=0).unsqueeze(0)
#             window_tensor = window_tensor.to(self.device)
            
#             # Run transformer
#             with torch.no_grad():
#                 liveness_score, frame_logits, attn_weights = self.transformer(window_tensor)
            
#             score = liveness_score.squeeze().item()
#             variance = torch.var(attn_weights.squeeze()).item()
            
#             window_scores.append(score)
#             window_variances.append(variance)
        
#         # Aggregate predictions
#         video_score = np.mean(window_scores)
#         avg_variance = np.mean(window_variances)
        
#         # Confidence calibration
#         # High variance → low confidence (inconsistent predictions = noisy input)
#         # Low variance → high confidence (consistent predictions = stable input)
#         confidence = 1.0 - np.clip(avg_variance, 0, 1)
        
#         if return_details:
#             return {
#                 'liveness_score': float(video_score),
#                 'confidence': float(confidence),
#                 'window_scores': window_scores,
#                 'window_variances': window_variances,
#                 'avg_variance': float(avg_variance),
#             }
        
#         return video_score, confidence
    
#     def process_frame_stream(
#         self,
#         frame: np.ndarray,
#         buffer_size: int = 12,
#     ) -> Optional[Tuple[float, float, float]]:
#         """
#         Process frame stream for real-time inference.
        
#         Maintains a sliding buffer of frames and runs inference when buffer is full.
        
#         Args:
#             frame: Input frame (BGR, numpy array)
#             buffer_size: Number of frames to buffer before inference
        
#         Returns:
#             (liveness_score, confidence, temporal_variance) if buffer full, else None
#         """
        
#         # Initialize buffer on first call
#         if not hasattr(self, '_frame_buffer'):
#             self._frame_buffer = deque(maxlen=buffer_size)
        
#         # Extract and add frame features
#         frame_features = self._extract_frame_features(frame)
#         self._frame_buffer.append(frame_features)
        
#         # Run inference when buffer is full
#         if len(self._frame_buffer) == buffer_size:
#             window_tensor = torch.stack(list(self._frame_buffer), dim=0).unsqueeze(0)
#             window_tensor = window_tensor.to(self.device)
            
#             with torch.no_grad():
#                 liveness_score, frame_logits, attn_weights = self.transformer(window_tensor)
            
#             score = liveness_score.squeeze().item()
#             variance = torch.var(attn_weights.squeeze()).item()
#             confidence = 1.0 - np.clip(variance, 0, 1)
            
#             return score, confidence, variance
        
#         return None
    
#     def reset_stream(self):
#         """Reset frame buffer for new video stream."""
#         if hasattr(self, '_frame_buffer'):
#             self._frame_buffer.clear()
    
#     def _load_video_frames(
#         self,
#         video_path: str,
#         max_frames: int = None,
#     ) -> Optional[List[np.ndarray]]:
#         """Load frames from video file."""
#         try:
#             cap = cv2.VideoCapture(video_path)
#             frames = []
            
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 frames.append(frame)
                
#                 if max_frames and len(frames) >= max_frames:
#                     break
            
#             cap.release()
#             return frames if frames else None
        
#         except Exception as e:
#             print(f"Error loading video: {e}")
#             return None
    
#     def _extract_frame_features(self, frame: np.ndarray) -> torch.Tensor:
#         """
#         Extract combined features from single frame.
        
#         Returns: (1, D) tensor - CNN + handcrafted features
#         """
#         with torch.no_grad():
#             # Get liveness features
#             face_img, lbp, freq, moire, depth = \
#                 self.preprocessor.preprocess_with_liveness_features(frame)
            
#             # Extract CNN embedding from EfficientNet
#             # Assuming face_img is already (1, 3, 300, 300)
#             face_tensor = face_img.to(self.device).unsqueeze(0)
            
#             # Get features before final classifier
#             with torch.no_grad():
#                 # Global average pooling output
#                 cnn_embedding = self.efficientnet.model.avgpool(
#                     self.efficientnet.model.features(face_tensor)
#                 ).view(1, -1)
            
#             # Move all features to same device
#             cnn_embedding = cnn_embedding.to(self.device)
#             lbp = lbp.to(self.device) if not lbp.is_cuda else lbp
#             freq = freq.to(self.device) if not freq.is_cuda else freq
#             moire = moire.to(self.device) if not moire.is_cuda else moire
#             depth = depth.to(self.device) if not depth.is_cuda else depth
            
#             # Concatenate all features
#             combined = torch.cat([cnn_embedding, lbp, freq, moire, depth], dim=1)
        
#         return combined.cpu()


# def run_inference_example(
#     video_path: str,
#     transformer_weights_path: str,
#     efficientnet_weights_path: str = None,
# ):
#     """
#     Example: Run inference on video using trained transformer.
    
#     Args:
#         video_path: Path to input video
#         transformer_weights_path: Path to saved transformer weights
#         efficientnet_weights_path: Path to saved EfficientNet weights (optional)
#     """
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Load models
#     print("Loading models...")
#     transformer = TemporalLivenessTransformer(
#         cnn_embedding_dim=1280,
#         lbp_dim=768,
#         freq_dim=785,
#         moire_dim=29,
#         depth_dim=16,
#         embedding_dim=256,
#         num_transformer_layers=2,
#         num_heads=4,
#     )
    
#     # Load transformer weights
#     state_dict = torch.load(transformer_weights_path, map_location=device)
#     transformer.load_state_dict(state_dict)
    
#     # Load EfficientNet
#     efficientnet = load_efficientnet_model(
#         weights_path=efficientnet_weights_path,
#         device=device,
#         pretrained=True
#     )
    
#     # Initialize inference
#     inference = TemporalLivenessInference(
#         transformer,
#         efficientnet,
#         device=device,
#         window_size=12,
#         stride=4,
#     )
    
#     # Run inference
#     print(f"Processing: {video_path}")
#     details = inference.process_video(video_path, return_details=True)
    
#     print("\n" + "=" * 60)
#     print("INFERENCE RESULTS")
#     print("=" * 60)
#     print(f"Liveness Score:  {details['liveness_score']:.4f}")
#     print(f"Confidence:      {details['confidence']:.4f}")
#     print(f"Avg Variance:    {details['avg_variance']:.4f}")
#     print(f"Prediction:      {'LIVE' if details['liveness_score'] > 0.5 else 'SPOOF'}")
#     print("=" * 60)
    
#     # Interpretation
#     score = details['liveness_score']
#     confidence = details['confidence']
    
#     if score > 0.75 and confidence > 0.7:
#         print("✓ Confident LIVE detection")
#     elif score < 0.25 and confidence > 0.7:
#         print("✓ Confident SPOOF detection")
#     elif 0.4 < score < 0.6:
#         print("⚠ Uncertain prediction (temporal inconsistency)")
#     else:
#         print(f"► Prediction: {'Live' if score > 0.5 else 'Spoof'} (confidence: {confidence:.2f})")
    
#     return details


# def stream_inference_example(video_path: str, transformer_weights_path: str):
#     """
#     Example: Stream-based inference for real-time processing.
    
#     Args:
#         video_path: Path to video
#         transformer_weights_path: Path to saved transformer weights
#     """
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Load models
#     transformer = TemporalLivenessTransformer()
#     state_dict = torch.load(transformer_weights_path, map_location=device)
#     transformer.load_state_dict(state_dict)
    
#     efficientnet = load_efficientnet_model(device=device, pretrained=True)
    
#     # Initialize streaming inference
#     inference = TemporalLivenessInference(transformer, efficientnet, device=device)
    
#     # Process video frame-by-frame
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     predictions = []
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_count += 1
        
#         # Process frame (returns result every buffer_size frames)
#         result = inference.process_frame_stream(frame, buffer_size=12)
        
#         if result is not None:
#             score, confidence, variance = result
#             predictions.append((frame_count, score, confidence, variance))
#             print(f"Frame {frame_count}: Score={score:.3f}, Conf={confidence:.3f}")
    
#     cap.release()
    
#     # Final aggregation
#     if predictions:
#         final_scores = [p[1] for p in predictions]
#         final_score = np.mean(final_scores)
#         print(f"\nFinal Score: {final_score:.4f} ({'LIVE' if final_score > 0.5 else 'SPOOF'})")
    
#     return predictions


# if __name__ == '__main__':
#     print("Temporal Liveness Transformer - Inference Module")
#     print("=" * 60)
#     print("This module provides:")
#     print("1. Batch video processing with sliding windows")
#     print("2. Stream-based frame-by-frame inference")
#     print("3. Confidence calibration via temporal variance")
#     print("4. Ready for integration with real-time systems")
#     print("=" * 60)
    
#     # Example usage (requires actual model and video):
#     # details = run_inference_example(
#     #     video_path='test_video.mp4',
#     #     transformer_weights_path='temporal_transformer_best.pt',
#     # )
