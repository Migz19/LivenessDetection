"""
Enhanced Preprocessing for Liveness Detection
Adds liveness-specific features beyond simple normalization
"""

import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional


class LivenessPreprocessor:
    """
    Enhanced preprocessing for liveness detection
    Includes: face alignment, LBP features, frequency analysis, motion detection,
    screen moire detection, depth estimation, and reflection analysis
    """
    
    def __init__(self, model_type='cnn'):
        self.model_type = model_type
        self.target_size = 300 if model_type == 'cnn' else 224
    
    def preprocess_with_liveness_features(self, image: np.ndarray, 
                                         face_bbox: Optional[Tuple] = None) -> torch.Tensor:
        """
        Preprocess with liveness detection features
        Returns: (face_tensor, lbp_tensor, freq_tensor, moire_tensor, depth_tensor)
        """
        # Crop and enhance face
        face_img = self._crop_face(image, face_bbox)
        
        # Main preprocessing
        main_tensor = self._preprocess_face(face_img)
        
        # Extract LBP features (texture analysis)
        lbp_features = self._extract_enhanced_lbp_features(face_img)
        
        # Extract frequency domain features
        freq_features = self._extract_frequency_features(face_img)
        
        # Extract moire pattern features (screen detection)
        moire_features = self._extract_moire_features(face_img)
        
        # Extract pseudo-depth features
        depth_features = self._extract_depth_cues(face_img)
        
        return main_tensor, lbp_features, freq_features, moire_features, depth_features
    
    def _crop_face(self, image: np.ndarray, bbox: Optional[Tuple]) -> np.ndarray:
        """Crop face region from image with extended context"""
        if bbox is None or bbox == (0, 0, 0, 0):
            return image
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = image.shape[:2]
        
        # Add larger padding to capture surrounding context (helps detect photo edges)
        margin = int(0.3 * (x2 - x1))
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        return image[y1:y2, x1:x2]
    
    def _preprocess_face(self, face_img: np.ndarray) -> torch.Tensor:
        """Normalize and resize face with multi-scale processing"""
        # Resize
        face_img = cv2.resize(face_img, (self.target_size, self.target_size))
        
        # Enhance contrast with adaptive CLAHE
        if len(face_img.shape) == 3:
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            face_img = cv2.merge([l, a, b])
            face_img = cv2.cvtColor(face_img, cv2.COLOR_LAB2BGR)
        
        # Normalize
        face_img = face_img.astype(np.float32) / 255.0
        face_img = (face_img - 0.5) / 0.5  # Normalize to [-1, 1]
        
        # Convert to tensor
        tensor = torch.from_numpy(face_img).permute(2, 0, 1)  # HWC -> CHW
        return tensor.unsqueeze(0)  # Add batch dim
    
    def _extract_enhanced_lbp_features(self, face_img: np.ndarray) -> torch.Tensor:
        """
        Extract multi-scale Local Binary Pattern features
        Photos have different texture patterns than real skin
        """
        # Convert to grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Resize for LBP
        gray = cv2.resize(gray, (128, 128))
        
        # Multi-scale LBP
        histograms = []
        
        # Scale 1: radius=1, 8 points
        lbp1 = self._compute_lbp(gray, radius=1, n_points=8)
        hist1 = cv2.calcHist([lbp1], [0], None, [256], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        histograms.append(hist1)
        
        # Scale 2: radius=2, 16 points (captures finer details)
        lbp2 = self._compute_lbp(gray, radius=2, n_points=16)
        hist2 = cv2.calcHist([lbp2], [0], None, [256], [0, 256])
        hist2 = cv2.normalize(hist2, hist2).flatten()
        histograms.append(hist2)
        
        # Scale 3: radius=3, 24 points (captures photo print patterns)
        lbp3 = self._compute_lbp(gray, radius=3, n_points=24)
        hist3 = cv2.calcHist([lbp3], [0], None, [256], [0, 256])
        hist3 = cv2.normalize(hist3, hist3).flatten()
        histograms.append(hist3)
        
        # Combine all scales
        combined = np.concatenate(histograms)
        
        # Convert to tensor
        tensor = torch.from_numpy(combined).float()
        return tensor.unsqueeze(0)
    
    def _compute_lbp(self, img: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Compute Local Binary Pattern with configurable radius and points"""
        rows, cols = img.shape
        lbp = np.zeros_like(img)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = img[i, j]
                pattern = 0
                
                # Sample points on circle
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j - radius * np.sin(angle))
                    
                    # Bilinear interpolation for sub-pixel accuracy
                    x = np.clip(x, 0, rows - 1)
                    y = np.clip(y, 0, cols - 1)
                    
                    neighbor = img[x, y]
                    
                    if neighbor >= center:
                        pattern |= (1 << k)
                
                lbp[i, j] = pattern
        
        return lbp.astype(np.uint8)
    
    def _extract_frequency_features(self, face_img: np.ndarray) -> torch.Tensor:
        """
        Extract frequency domain features using both DCT and FFT
        Printed photos and screens have distinct frequency signatures
        """
        # Convert to grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Resize for frequency analysis
        gray = cv2.resize(gray, (128, 128))
        normalized = np.float32(gray) / 255.0
        
        # === DCT Features ===
        dct = cv2.dct(normalized)
        
        # Extract DCT features from different frequency bands
        low_freq = dct[:32, :32].flatten()
        mid_freq = dct[32:64, 32:64].flatten()
        high_freq = dct[64:, 64:].flatten()
        
        # === FFT Features ===
        fft = np.fft.fft2(normalized)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Extract radial frequency distribution (helps detect print patterns)
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Create radial bins
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # Radial profile
        radial_bins = 16
        radial_profile = []
        for i in range(radial_bins):
            mask = (r >= i * h / (2 * radial_bins)) & (r < (i + 1) * h / (2 * radial_bins))
            if mask.any():
                radial_profile.append(np.mean(magnitude[mask]))
            else:
                radial_profile.append(0)
        
        radial_profile = np.array(radial_profile)
        
        # === Energy Ratios ===
        # High frequency energy (photos have less high-freq content)
        mask_high = r > h / 4
        mask_low = r <= h / 4
        high_energy = np.sum(magnitude[mask_high])
        low_energy = np.sum(magnitude[mask_low])
        energy_ratio = high_energy / (low_energy + 1e-7)
        
        # Combine all frequency features
        features = np.concatenate([
            low_freq[:256],  # Limit size
            mid_freq[:256],
            high_freq[:256],
            radial_profile,
            [energy_ratio]
        ])
        
        # Convert to tensor
        tensor = torch.from_numpy(features).float()
        return tensor.unsqueeze(0)
    
    def _extract_moire_features(self, face_img: np.ndarray) -> torch.Tensor:
        """
        Detect moiré patterns from phone/tablet screens
        Screens display characteristic interference patterns
        """
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        gray = cv2.resize(gray, (128, 128))
        
        # Apply bandpass filters to isolate moiré frequencies
        # Moiré patterns typically appear in mid-frequency ranges
        
        # Gaussian blur (low-pass)
        low_pass = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # High-pass filter (original - blurred)
        high_pass = cv2.subtract(gray, low_pass)
        
        # Compute FFT on high-pass
        fft = np.fft.fft2(high_pass)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Look for periodic patterns (moiré signature)
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Create annular masks to detect periodic peaks
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # Moiré patterns create peaks at specific frequencies
        features = []
        for ring in range(4, 20, 2):  # Check different frequency rings
            mask = (r >= ring) & (r < ring + 2)
            if mask.any():
                ring_energy = np.sum(magnitude[mask])
                ring_std = np.std(magnitude[mask])
                features.extend([ring_energy, ring_std])
        
        # Also check for directional patterns (screen grids)
        # Horizontal and vertical frequency responses
        h_mid = magnitude[center_h-2:center_h+2, :]
        v_mid = magnitude[:, center_w-2:center_w+2]
        
        h_energy = np.sum(h_mid)
        v_energy = np.sum(v_mid)
        
        features.extend([h_energy, v_energy, h_energy / (v_energy + 1e-7)])
        
        features = np.array(features)
        
        # Convert to tensor
        tensor = torch.from_numpy(features).float()
        return tensor.unsqueeze(0)
    
    def _extract_depth_cues(self, face_img: np.ndarray) -> torch.Tensor:
        """
        Extract pseudo-depth features
        Photos are flat; real faces have 3D structure with shading gradients
        """
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        gray = cv2.resize(gray, (128, 128))
        gray = gray.astype(np.float32) / 255.0
        
        # === Gradient Analysis ===
        # Real faces have smooth, consistent gradients
        # Photos often have abrupt changes or flat regions
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_direction = np.arctan2(sobely, sobelx)
        
        # Gradient statistics
        grad_mean = np.mean(gradient_magnitude)
        grad_std = np.std(gradient_magnitude)
        grad_max = np.max(gradient_magnitude)
        
        # === Laplacian (2nd derivative - detects edges) ===
        # Photos have sharper edges, real faces smoother
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = np.var(laplacian)
        
        # === Illumination Analysis ===
        # Divide face into regions and analyze lighting consistency
        h, w = gray.shape
        regions = []
        
        # 3x3 grid
        for i in range(3):
            for j in range(3):
                region = gray[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
                regions.append(np.mean(region))
        
        regions = np.array(regions)
        
        # Lighting variance (real faces have gradual changes)
        lighting_variance = np.var(regions)
        lighting_range = np.max(regions) - np.min(regions)
        
        # === Specular Highlights ===
        # Real skin has subtle specular reflections
        # Photos often have uniform or excessive glare
        
        # Threshold for bright spots
        threshold = np.percentile(gray, 95)
        bright_mask = gray > threshold
        specular_ratio = np.sum(bright_mask) / gray.size
        
        # Brightness gradient in specular regions
        if specular_ratio > 0:
            specular_gradient = np.mean(gradient_magnitude[bright_mask])
        else:
            specular_gradient = 0
        
        # === Shadow Analysis ===
        # Real faces cast soft shadows; photos may have printed shadows
        dark_threshold = np.percentile(gray, 5)
        shadow_mask = gray < dark_threshold
        shadow_ratio = np.sum(shadow_mask) / gray.size
        
        # === Combine Features ===
        features = np.array([
            grad_mean, grad_std, grad_max,
            lap_var,
            lighting_variance, lighting_range,
            specular_ratio, specular_gradient,
            shadow_ratio,
            np.std(gradient_direction)  # Directional consistency
        ])
        
        # Convert to tensor
        tensor = torch.from_numpy(features).float()
        return tensor.unsqueeze(0)
    
    def extract_batch_liveness_features(self, image_arrays: List[np.ndarray],
                                       face_bboxes: Optional[List[Tuple]] = None) -> Tuple[torch.Tensor, ...]:
        """Extract liveness features for batch of images"""
        main_tensors = []
        lbp_tensors = []
        freq_tensors = []
        moire_tensors = []
        depth_tensors = []
        
        for idx, image in enumerate(image_arrays):
            bbox = None
            if face_bboxes:
                if len(face_bboxes) == 1:
                    bbox = face_bboxes[0]
                elif idx < len(face_bboxes):
                    bbox = face_bboxes[idx]
            
            main, lbp, freq, moire, depth = self.preprocess_with_liveness_features(image, bbox)
            main_tensors.append(main.squeeze(0))
            lbp_tensors.append(lbp.squeeze(0))
            freq_tensors.append(freq.squeeze(0))
            moire_tensors.append(moire.squeeze(0))
            depth_tensors.append(depth.squeeze(0))
        
        return (torch.stack(main_tensors),
                torch.stack(lbp_tensors),
                torch.stack(freq_tensors),
                torch.stack(moire_tensors),
                torch.stack(depth_tensors))


class MotionBasedLivenessDetector:
    """
    Detect liveness based on motion patterns in video
    Real faces show natural head movements and micro-expressions
    Photos/screens are static or have artificial motion
    """
    
    def __init__(self, threshold=0.05):
        self.threshold = threshold
    
    def detect_from_frames(self, frames: List[np.ndarray], 
                          face_bboxes: List[Tuple]) -> Tuple[str, float, dict]:
        """
        Detect liveness from video frames using multiple motion cues
        Returns: (label, confidence, detailed_features)
        """
        if len(frames) < 2:
            return "Unknown", 0.5, {}
        
        # Extract face regions
        face_regions = []
        for frame in frames:
            if len(frame.shape) != 3:
                face_regions.append(cv2.resize(frame, (64, 64)))
                continue
                
            x1, y1, x2, y2 = [int(v) for v in face_bboxes[0]]
            h, w = frame.shape[:2]
            
            # Bounds checking
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                face = frame
            else:
                face = frame[y1:y2, x1:x2]
            
            face_regions.append(cv2.resize(face, (128, 128)))
        
        # === Optical Flow Analysis ===
        flows = []
        flow_directions = []
        
        try:
            gray_prev = cv2.cvtColor(face_regions[0], cv2.COLOR_BGR2GRAY)
        except:
            gray_prev = face_regions[0] if len(face_regions[0].shape) == 2 else cv2.cvtColor(face_regions[0], cv2.COLOR_BGR2GRAY)
        
        for i in range(1, len(face_regions)):
            try:
                gray_curr = cv2.cvtColor(face_regions[i], cv2.COLOR_BGR2GRAY)
            except:
                gray_curr = face_regions[i] if len(face_regions[i].shape) == 2 else cv2.cvtColor(face_regions[i], cv2.COLOR_BGR2GRAY)
            
            # Dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Compute flow magnitude and direction
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            angle = np.arctan2(flow[..., 1], flow[..., 0])
            
            flows.append(np.mean(mag))
            flow_directions.append(angle)
            
            gray_prev = gray_curr
        
        if not flows:
            return "Unknown", 0.5, {}
        
        # === Motion Pattern Analysis ===
        flows = np.array(flows)
        mean_motion = np.mean(flows)
        motion_variance = np.var(flows)
        motion_std = np.std(flows)
        
        # === Temporal Consistency ===
        # Real faces: smooth, continuous motion
        # Photos held by hand: shaky, discontinuous
        # Videos on screen: too smooth/looping
        
        # Check for motion smoothness
        motion_changes = np.abs(np.diff(flows))
        motion_smoothness = np.mean(motion_changes)
        
        # === Direction Consistency ===
        # Real faces: varied but consistent directions (head movement)
        # Photos: random jitter or overly uniform
        
        direction_variance = 0
        if len(flow_directions) > 1:
            # Calculate variance in flow directions
            for flow_dir in flow_directions:
                direction_variance += np.var(flow_dir)
            direction_variance /= len(flow_directions)
        
        # === Micro-Movement Detection ===
        # Real faces always have subtle movements (breathing, micro-expressions)
        # Static photos: zero or near-zero motion
        micro_motion_count = np.sum(flows > 0.01)
        micro_motion_ratio = micro_motion_count / len(flows)
        
        # === Feature Dictionary ===
        features = {
            'mean_motion': float(mean_motion),
            'motion_variance': float(motion_variance),
            'motion_smoothness': float(motion_smoothness),
            'direction_variance': float(direction_variance),
            'micro_motion_ratio': float(micro_motion_ratio)
        }
        
        # === Decision Logic ===
        score = 0.5  # Start neutral
        
        # Motion analysis (softer thresholds for real videos with subtle movement)
        # Real faces may have subtle movement - we penalize EXTREME static, not all low motion
        if mean_motion < 0.005:  # VERY static = photo
            score -= 0.3
        elif mean_motion < 0.01:  # Extremely low motion
            score -= 0.1
        
        # Moderate to high motion = likely real
        if mean_motion > 0.05:
            score += 0.15
        if mean_motion > 0.1:
            score += 0.2
        
        # High motion variance = likely real (natural movement variation)
        if motion_variance > 0.001:
            score += 0.1
        if motion_variance > 0.01:
            score += 0.2
        
        # Too smooth motion with moderate-high speed = likely screen video
        if mean_motion > 0.08 and motion_variance < 0.0001:
            score -= 0.2
        
        # Good micro-motion = likely real (breathing, micro-expressions)
        if micro_motion_ratio > 0.5:
            score += 0.2
        elif micro_motion_ratio < 0.1:
            score -= 0.15
        
        # Direction variance check (real movement has varied directions)
        if direction_variance > 0.1:
            score += 0.1
        elif direction_variance < 0.01:
            score -= 0.1
        
        # Motion smoothness (natural movement should have some variation)
        if motion_smoothness > 0.01:  # Some variation = good
            score += 0.1
        
        # Clamp score
        score = np.clip(score, 0, 1)
        
        # Determine label and confidence
        if score > 0.55:
            label = "Live"
            confidence = min(0.98, score)
        elif score < 0.45:
            label = "Fake"
            confidence = min(0.98, 1 - score)
        else:
            label = "Uncertain"
            confidence = 0.5
        
        return label, confidence, features


class ColorAnalysisLiveness:
    """
    Analyze color characteristics to detect printed photos and screens
    """
    
    @staticmethod
    def analyze_color_saturation(face_img: np.ndarray) -> dict:
        """
        Real skin has specific saturation characteristics
        Photos often have altered saturation
        """
        if len(face_img.shape) != 3:
            return {'saturation_mean': 0, 'saturation_std': 0}
        
        # Convert to HSV
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        
        # Saturation statistics
        saturation_mean = np.mean(s_channel)
        saturation_std = np.std(s_channel)
        saturation_range = np.max(s_channel) - np.min(s_channel)
        
        return {
            'saturation_mean': float(saturation_mean),
            'saturation_std': float(saturation_std),
            'saturation_range': float(saturation_range)
        }
    
    @staticmethod
    def detect_color_cast(face_img: np.ndarray) -> dict:
        """
        Detect unnatural color casts (common in photos/screens)
        """
        if len(face_img.shape) != 3:
            return {'r_mean': 0, 'g_mean': 0, 'b_mean': 0, 'color_balance': 0}
        
        b, g, r = cv2.split(face_img)
        
        r_mean = np.mean(r)
        g_mean = np.mean(g)
        b_mean = np.mean(b)
        
        # Color balance (should be relatively even for real faces)
        color_balance = np.std([r_mean, g_mean, b_mean])
        
        return {
            'r_mean': float(r_mean),
            'g_mean': float(g_mean),
            'b_mean': float(b_mean),
            'color_balance': float(color_balance)
        }


def get_liveness_features_summary(image: np.ndarray, face_bbox: Optional[Tuple] = None) -> dict:
    """
    Get a comprehensive summary of liveness features for analysis
    """
    preprocessor = LivenessPreprocessor()
    
    # Get face region
    x1, y1, x2, y2 = face_bbox if face_bbox else (0, 0, image.shape[1], image.shape[0])
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Bounds checking
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Ensure valid bbox
    if x2 <= x1 or y2 <= y1:
        face = image
    else:
        face = image[y1:y2, x1:x2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
    
    # Basic features
    brightness = np.mean(gray)
    contrast = np.std(gray)
    blurriness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Color analysis (only for color images)
    if len(face.shape) == 3:
        color_features = ColorAnalysisLiveness.analyze_color_saturation(face)
        color_cast = ColorAnalysisLiveness.detect_color_cast(face)
    else:
        color_features = {
            'saturation_mean': 0.0,
            'saturation_std': 0.0,
            'saturation_range': 0.0
        }
        color_cast = {
            'r_mean': 0.0,
            'g_mean': 0.0,
            'b_mean': 0.0,
            'color_balance': 0.0
        }
    
    # Edge sharpness (photos tend to have sharper edges)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Combine all features safely
    summary = {
        'brightness': float(brightness),
        'contrast': float(contrast),
        'blurriness': float(blurriness),
        'face_size': int((x2 - x1) * (y2 - y1)),
        'edge_density': float(edge_density),
        'saturation_mean': float(color_features['saturation_mean']),
        'saturation_std': float(color_features['saturation_std']),
        'saturation_range': float(color_features['saturation_range']),
        'r_mean': float(color_cast['r_mean']),
        'g_mean': float(color_cast['g_mean']),
        'b_mean': float(color_cast['b_mean']),
        'color_balance': float(color_cast['color_balance'])
    }
    
    return summary


def compute_photo_artifacts_score(image: np.ndarray, face_bbox: Optional[Tuple] = None) -> float:
    """
    Compute a score indicating likelihood of photo artifacts
    Higher score = more likely to be a photo
    Returns: score between 0 and 1
    """
    features = get_liveness_features_summary(image, face_bbox)
    
    score = 0.5  # Start neutral
    
    # Very sharp edges suggest printed photo
    if features['edge_density'] > 0.15:
        score += 0.2
    
    # Low blur suggests photo (real faces have motion blur)
    if features['blurriness'] > 500:
        score -= 0.15
    elif features['blurriness'] < 100:
        score += 0.15
    
    # Extreme saturation suggests processed photo
    if features['saturation_std'] < 20 or features['saturation_std'] > 80:
        score += 0.1
    
    # Color imbalance suggests photo/screen
    if features['color_balance'] > 30:
        score += 0.1
    
    return np.clip(score, 0, 1)