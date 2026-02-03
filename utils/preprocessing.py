import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, List, Dict


class ImagePreprocessor:
    """
    Preprocessing for images
    """
    def __init__(self, model_type='cnn'):
        self.model_type = model_type
        
        if model_type == 'cnn':
            self.input_size = (300, 300)
            self.transforms = self._get_cnn_transforms()
        else:  # efficientnet-b3
            self.input_size = (300, 300)
            self.transforms = self._get_efficientnet_transforms()
    
    def _get_cnn_transforms(self):
        """Transforms for CNN model (inference)"""
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_efficientnet_transforms(self):
        """Transforms for EfficientNet model (inference)"""
        return transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_cnn_augmented_transforms(self):
        """Augmented transforms for CNN training (if needed)"""
        return transforms.Compose([
            transforms.RandomResizedCrop(300, scale=(0.75, 1.0)),
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
    
    def preprocess_image(self, image_path: str, face_bbox=None) -> torch.Tensor:
        """
        Preprocess image from file
        Args:
            image_path: Path to image file
            face_bbox: Optional bounding box (x1, y1, x2, y2) for face region
        Returns:
            Preprocessed tensor of shape (1, C, H, W)
        """
        image = Image.open(image_path).convert('RGB')
        
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            image = image.crop((x1, y1, x2, y2))
        
        tensor = self.transforms(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def preprocess_array(self, image_array: np.ndarray, face_bbox=None) -> torch.Tensor:
        """
        Preprocess image from numpy array (BGR from OpenCV)
        Args:
            image_array: Numpy array in BGR format
            face_bbox: Optional bounding box (x1, y1, x2, y2) for face region
        Returns:
            Preprocessed tensor of shape (1, C, H, W)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            image = image.crop((x1, y1, x2, y2))
        
        tensor = self.transforms(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def preprocess_batch(self, image_arrays: list, face_bboxes=None) -> torch.Tensor:
        """
        Preprocess multiple images (for batch processing)
        Args:
            image_arrays: List of numpy arrays in BGR format
            face_bboxes: Optional list of bounding boxes (can be None, per-frame, or single)
        Returns:
            Preprocessed tensor of shape (B, C, H, W)
        """
        tensors = []
        for idx, image_array in enumerate(image_arrays):
            # Handle different bbox formats:
            # - None: no bboxes
            # - Single bbox: apply to all frames (most common in video)
            # - Multiple bboxes matching frames: use idx
            # - Multiple bboxes not matching: use first bbox
            bbox = None
            if face_bboxes:
                if len(face_bboxes) == 1:
                    # Single bbox for all frames
                    bbox = face_bboxes[0]
                elif len(face_bboxes) == len(image_arrays):
                    # One bbox per frame
                    bbox = face_bboxes[idx]
                else:
                    # Mismatch: use first bbox or skip
                    bbox = face_bboxes[0] if idx < len(face_bboxes) else None
            
            tensor = self.preprocess_array(image_array, face_bbox=bbox)
            tensors.append(tensor.squeeze(0))  # Remove batch dim
        
        return torch.stack(tensors)  # Stack into batch


class VideoPreprocessor:
    """
    Enhanced preprocessing for video frames with liveness detection features
    """
    def __init__(self, model_type='cnn'):
        self.image_preprocessor = ImagePreprocessor(model_type)
        self.model_type = model_type
    
    def extract_frames(self, video_path: str, num_frames: int = 10, 
                      start_time: float = 0, duration: float = None) -> list:
        """
        Extract frames from video with temporal sampling
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            start_time: Start time in seconds
            duration: Duration to sample from (seconds)
        Returns:
            List of numpy arrays (BGR format)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = int(start_time * fps)
        if duration is None:
            end_frame = total_frames
        else:
            end_frame = min(int((start_time + duration) * fps), total_frames)
        
        frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def extract_frames_from_webcam(self, num_frames: int = 10) -> list:
        """
        Capture frames from webcam
        Args:
            num_frames: Number of frames to capture
        Returns:
            List of numpy arrays (BGR format)
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frames = []
        count = 0
        
        while count < num_frames:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                count += 1
            else:
                break
        
        cap.release()
        return frames
    
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame for better face detection
        Args:
            frame: Input frame in BGR format
        Returns:
            Enhanced frame
        """
        # Increase brightness and contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def preprocess_video_frames(self, frames: list, face_bboxes=None) -> torch.Tensor:
        """
        Preprocess video frames
        Args:
            frames: List of frames in BGR format
            face_bboxes: Optional list of bounding boxes for each frame
        Returns:
            Preprocessed tensor of shape (B, C, H, W)
        """
        enhanced_frames = [self.enhance_frame(f) for f in frames]
        return self.image_preprocessor.preprocess_batch(enhanced_frames, face_bboxes)
    
    # === LIVENESS DETECTION ENHANCEMENTS ===
    
    def extract_optical_flow(self, frames: List[np.ndarray], face_bbox=None) -> np.ndarray:
        """
        Extract optical flow between consecutive frames to detect micro-movements
        Args:
            frames: List of frames in BGR format
            face_bbox: Optional bounding box to focus on face region
        Returns:
            Optical flow magnitude array of shape (num_frames-1, H, W)
        """
        flow_magnitudes = []
        
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            # Crop to face region if bbox provided
            if face_bbox:
                x1, y1, x2, y2 = face_bbox
                frame1 = frame1[y1:y2, x1:x2]
                frame2 = frame2[y1:y2, x1:x2]
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_magnitudes.append(magnitude)
        
        return np.array(flow_magnitudes)
    
    def extract_texture_features(self, frame: np.ndarray, face_bbox=None) -> Dict[str, float]:
        """
        Extract texture features using LBP (Local Binary Patterns) for spoofing detection
        Args:
            frame: Frame in BGR format
            face_bbox: Optional bounding box for face region
        Returns:
            Dictionary of texture features
        """
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            frame = frame[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate LBP
        lbp = self._calculate_lbp(gray)
        
        # Calculate histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        # Extract statistical features
        features = {
            'lbp_entropy': -np.sum(hist * np.log2(hist + 1e-7)),
            'lbp_energy': np.sum(hist ** 2),
            'lbp_variance': np.var(lbp),
            'texture_contrast': self._calculate_contrast(gray)
        }
        
        return features
    
    def _calculate_lbp(self, gray_image: np.ndarray, radius: int = 1, 
                       n_points: int = 8) -> np.ndarray:
        """
        Calculate Local Binary Pattern
        Args:
            gray_image: Grayscale image
            radius: Radius of circular pattern
            n_points: Number of sampling points
        Returns:
            LBP image
        """
        height, width = gray_image.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = gray_image[i, j]
                binary_string = ""
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j - radius * np.sin(angle))
                    
                    neighbor = gray_image[x, y]
                    binary_string += '1' if neighbor >= center else '0'
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp
    
    def _calculate_contrast(self, gray_image: np.ndarray) -> float:
        """
        Calculate image contrast using standard deviation
        Args:
            gray_image: Grayscale image
        Returns:
            Contrast value
        """
        return np.std(gray_image)
    
    def extract_frequency_features(self, frame: np.ndarray, face_bbox=None) -> Dict[str, float]:
        """
        Extract frequency domain features using FFT to detect print/screen artifacts
        Args:
            frame: Frame in BGR format
            face_bbox: Optional bounding box for face region
        Returns:
            Dictionary of frequency features
        """
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            frame = frame[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        # Calculate features
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # High frequency energy (outer regions)
        mask_outer = np.ones((h, w))
        mask_outer[center_h - h//4:center_h + h//4, 
                   center_w - w//4:center_w + w//4] = 0
        high_freq_energy = np.sum(magnitude_spectrum * mask_outer)
        
        # Low frequency energy (center regions)
        mask_inner = np.zeros((h, w))
        mask_inner[center_h - h//4:center_h + h//4, 
                   center_w - w//4:center_w + w//4] = 1
        low_freq_energy = np.sum(magnitude_spectrum * mask_inner)
        
        total_energy = np.sum(magnitude_spectrum)
        
        features = {
            'high_freq_ratio': high_freq_energy / (total_energy + 1e-7),
            'low_freq_ratio': low_freq_energy / (total_energy + 1e-7),
            'freq_energy_ratio': high_freq_energy / (low_freq_energy + 1e-7)
        }
        
        return features
    
    def detect_blink_pattern(self, frames: List[np.ndarray], 
                            face_landmarks: List = None) -> Dict[str, any]:
        """
        Detect eye blink patterns for liveness verification
        Args:
            frames: List of frames in BGR format
            face_landmarks: Optional list of facial landmarks for each frame
        Returns:
            Dictionary with blink detection results
        """
        if not face_landmarks or len(face_landmarks) != len(frames):
            return {'blink_detected': False, 'blink_count': 0, 'avg_closure_ratio': 0.0}
        
        ear_values = []  # Eye Aspect Ratio
        
        for landmarks in face_landmarks:
            if landmarks is not None:
                ear = self._calculate_eye_aspect_ratio(landmarks)
                ear_values.append(ear)
        
        if len(ear_values) == 0:
            return {'blink_detected': False, 'blink_count': 0, 'avg_closure_ratio': 0.0}
        
        # Detect blinks (when EAR drops below threshold)
        ear_threshold = 0.21
        blink_count = 0
        is_blinking = False
        
        for ear in ear_values:
            if ear < ear_threshold:
                if not is_blinking:
                    blink_count += 1
                    is_blinking = True
            else:
                is_blinking = False
        
        return {
            'blink_detected': blink_count > 0,
            'blink_count': blink_count,
            'avg_closure_ratio': np.mean(ear_values),
            'ear_variance': np.var(ear_values)
        }
    


# MediaPipe FaceMesh canonical eye landmark indices (6-point EAR style)
MP_LEFT_EYE  = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# dlib-68 eye indices
DLIB_LEFT_EYE  = [36, 37, 38, 39, 40, 41]
DLIB_RIGHT_EYE = [42, 43, 44, 45, 46, 47]

def _to_xy(landmarks, image_shape=None):
    """
    landmarks: array-like (N,2) or (N,3), possibly normalized if MediaPipe.
    image_shape: (H, W) if you want to convert normalized->pixel coords.
    """
    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 6:
        return None
    pts = pts[:, :2]

    if image_shape is not None:
        H, W = image_shape[:2]
        # If values look normalized, scale them
        if np.max(pts) <= 1.5:
            pts[:, 0] *= W
            pts[:, 1] *= H
    return pts

def _ear_from_6pts(eye_pts_6):
    # eye_pts_6 in order: [p1,p2,p3,p4,p5,p6]
    p1, p2, p3, p4, p5, p6 = eye_pts_6
    def norm(a, b): return float(np.linalg.norm(a - b))
    vertical1 = norm(p2, p6)
    vertical2 = norm(p3, p5)
    horizontal = norm(p1, p4)

    if horizontal < 1e-6:
        return np.nan
    return (vertical1 + vertical2) / (2.0 * horizontal)

def calculate_eye_aspect_ratio(landmarks, image_shape=None, fmt="mediapipe"):
    """
    Returns: float EAR (avg of both eyes) or np.nan if not computable.
    fmt: 'mediapipe' or 'dlib68'
    """
    pts = _to_xy(landmarks, image_shape=image_shape)
    if pts is None:
        return np.nan

    if fmt == "mediapipe":
        left_idx, right_idx = MP_LEFT_EYE, MP_RIGHT_EYE
    elif fmt == "dlib68":
        left_idx, right_idx = DLIB_LEFT_EYE, DLIB_RIGHT_EYE
    else:
        raise ValueError("fmt must be 'mediapipe' or 'dlib68'")

    try:
        left_eye = pts[left_idx]
        right_eye = pts[right_idx]
    except Exception:
        return np.nan

    left_ear = _ear_from_6pts(left_eye)
    right_ear = _ear_from_6pts(right_eye)

    if np.isnan(left_ear) and np.isnan(right_ear):
        return np.nan
    if np.isnan(left_ear):
        return right_ear
    if np.isnan(right_ear):
        return left_ear

    return 0.5 * (left_ear + right_ear)

def calculate_temporal_consistency(frames: List[np.ndarray], face_bboxes: List=None,
                                  resize=(128, 128)) -> Dict[str, float]:
    if len(frames) < 2:
        return {'diff_mean': 0.0, 'diff_std': 0.0, 'flow_mean': 0.0, 'flow_std': 0.0, 'motion_smoothness': 0.0}

    diffs = []
    flow_mags = []

    prev_gray = None

    for i in range(len(frames)):
        frame = frames[i]

        # crop with bbox (clip to bounds)
        if face_bboxes is not None and len(face_bboxes) > 0:
            bbox = face_bboxes[i] if i < len(face_bboxes) else face_bboxes[-1]
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            frame = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, resize, interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is not None:
            # illumination-robust-ish difference
            diff = cv2.absdiff(prev_gray, gray)
            diffs.append(float(np.mean(diff)))

            # optical flow magnitude
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_mags.append(float(np.mean(mag)))

        prev_gray = gray

    if len(diffs) == 0:
        return {'diff_mean': 0.0, 'diff_std': 0.0, 'flow_mean': 0.0, 'flow_std': 0.0, 'motion_smoothness': 0.0}

    diffs = np.asarray(diffs, dtype=np.float32)
    flow_mags = np.asarray(flow_mags, dtype=np.float32) if len(flow_mags) else np.zeros_like(diffs)

    # Smoothness: low "jerk" (2nd derivative) -> smoother, natural motion
    if len(flow_mags) >= 3:
        accel = np.diff(flow_mags, n=2)
        jerk_var = float(np.var(accel))
        motion_smoothness = 1.0 / (jerk_var + 1e-6)
    else:
        motion_smoothness = 1.0 / (float(np.var(flow_mags)) + 1e-6)

    return {
        'diff_mean': float(np.mean(diffs)),
        'diff_std': float(np.std(diffs)),
        'flow_mean': float(np.mean(flow_mags)),
        'flow_std': float(np.std(flow_mags)),
        'motion_smoothness': float(motion_smoothness),
        'max_diff': float(np.max(diffs))
    }

def extract_color_distribution(self, frame: np.ndarray, 
                                   face_bbox=None) -> Dict[str, np.ndarray]:
        """
        Extract color distribution features to detect color artifacts in spoofed images
        Args:
            frame: Frame in BGR format
            face_bbox: Optional bounding box for face region
        Returns:
            Dictionary of color histograms and statistics
        """
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            frame = frame[y1:y2, x1:x2]
        
        # Calculate histograms for each channel
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
        
        # Normalize
        hist_b = hist_b.flatten() / (hist_b.sum() + 1e-7)
        hist_g = hist_g.flatten() / (hist_g.sum() + 1e-7)
        hist_r = hist_r.flatten() / (hist_r.sum() + 1e-7)
        
        # Calculate color moments
        b, g, r = cv2.split(frame)
        
        features = {
            'hist_b': hist_b,
            'hist_g': hist_g,
            'hist_r': hist_r,
            'mean_b': np.mean(b),
            'mean_g': np.mean(g),
            'mean_r': np.mean(r),
            'std_b': np.std(b),
            'std_g': np.std(g),
            'std_r': np.std(r)
        }
        
        return features


def blend_frames(frames: list, alpha: float = 0.7) -> np.ndarray:
    """
    Blend consecutive frames to reduce noise
    Args:
        frames: List of frames
        alpha: Blending factor
    Returns:
        Blended frames
    """
    blended = []
    for i in range(len(frames)):
        if i == 0:
            blended.append(frames[i])
        else:
            blended_frame = cv2.addWeighted(
                frames[i], alpha, 
                blended[-1], 1 - alpha, 0
            )
            blended.append(blended_frame)
    return blended