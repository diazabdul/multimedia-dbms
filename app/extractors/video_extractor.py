"""
Video Feature Extractor - PyTorch Version
Extracts keyframe features, motion vectors, and scene statistics from videos
Supports GPU acceleration with CPU fallback
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import os
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class VideoFeatureExtractor:
    """Extract features from video files for similarity search using PyTorch"""
    
    def __init__(self, use_deep_features: bool = True, n_keyframes: int = 5, force_cpu: bool = False):
        """
        Initialize video feature extractor
        
        Args:
            use_deep_features: Whether to use MobileNetV2 for keyframe features
            n_keyframes: Number of keyframes to extract
            force_cpu: If True, force CPU even if GPU is available
        """
        self.use_deep_features = use_deep_features
        self.n_keyframes = n_keyframes
        self.model = None
        self.force_cpu = force_cpu
        self.device = None
        self.transform = None
        
        if use_deep_features:
            self._load_model()
    
    def _load_model(self):
        """Load MobileNetV2 for deep feature extraction using PyTorch"""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            # Initialize GPU if available
            from app.utils.gpu_utils import init_gpu, get_device, get_device_summary
            init_gpu(force_cpu=self.force_cpu)
            self.device = get_device()
            logger.info(f"Video extractor using: {get_device_summary()}")
            
            # Load MobileNetV2 pretrained model
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            
            # Remove classifier to get features only
            self.model.classifier = torch.nn.Identity()
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Define preprocessing transform
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info(f"MobileNetV2 loaded successfully for video extraction on {self.device}")
            
        except ImportError as e:
            logger.warning(f"PyTorch/torchvision not available: {e}")
            self.use_deep_features = False
        except Exception as e:
            logger.error(f"Could not load MobileNetV2: {e}")
            self.use_deep_features = False
    
    def extract_keyframes(self, video_path: str) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract keyframes from video using scene change detection
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (list of keyframe images, list of timestamps)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        keyframes = []
        timestamps = []
        
        if total_frames <= self.n_keyframes:
            # If video is short, sample all frames
            frame_indices = list(range(total_frames))
        else:
            # Sample frames evenly across the video
            frame_indices = np.linspace(0, total_frames - 1, self.n_keyframes, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                keyframes.append(frame)
                timestamps.append(idx / fps if fps > 0 else 0)
        
        cap.release()
        
        return keyframes, timestamps
    
    def extract_keyframe_features(self, keyframes: List[np.ndarray]) -> np.ndarray:
        """
        Extract deep features from keyframes and aggregate using PyTorch
        
        Args:
            keyframes: List of keyframe images
            
        Returns:
            Aggregated feature vector (1280 dimensions)
        """
        if not self.use_deep_features or self.model is None or len(keyframes) == 0:
            return np.zeros(1280, dtype=np.float32)
        
        try:
            import torch
            
            features_list = []
            
            for frame in keyframes:
                # Convert BGR to RGB and to PIL Image
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                
                # Apply transforms
                img_tensor = self.transform(pil_image).unsqueeze(0)
                img_tensor = img_tensor.to(self.device)
                
                # Extract features
                with torch.no_grad():
                    features = self.model(img_tensor)
                
                features_list.append(features.cpu().numpy().flatten())
            
            # Aggregate features (mean pooling)
            aggregated = np.mean(features_list, axis=0)
            
            return aggregated.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting keyframe features: {e}")
            return np.zeros(1280, dtype=np.float32)
    
    def extract_motion_features(self, video_path: str, sample_interval: int = 5) -> np.ndarray:
        """
        Extract motion features using optical flow
        
        Args:
            video_path: Path to video file
            sample_interval: Sample every N frames
            
        Returns:
            Motion feature vector (64 dimensions - histogram of optical flow)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return np.zeros(64, dtype=np.float32)
        
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return np.zeros(64, dtype=np.float32)
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        flow_magnitudes = []
        flow_angles = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % sample_interval != 0:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate magnitude and angle
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            flow_magnitudes.extend(mag.flatten())
            flow_angles.extend(ang.flatten())
            
            prev_gray = gray
        
        cap.release()
        
        if len(flow_magnitudes) == 0:
            return np.zeros(64, dtype=np.float32)
        
        # Create histograms
        mag_hist, _ = np.histogram(flow_magnitudes, bins=32, range=(0, 20), density=True)
        ang_hist, _ = np.histogram(flow_angles, bins=32, range=(0, 2 * np.pi), density=True)
        
        # Concatenate
        motion_features = np.concatenate([mag_hist, ang_hist])
        
        return motion_features.astype(np.float32)
    
    def extract_scene_stats(self, video_path: str) -> np.ndarray:
        """
        Extract scene statistics
        
        Args:
            video_path: Path to video file
            
        Returns:
            Scene statistics vector (10 dimensions)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return np.zeros(10, dtype=np.float32)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Sample frames for brightness and color analysis
        brightness_values = []
        color_variances = []
        
        sample_indices = np.linspace(0, total_frames - 1, min(20, total_frames), dtype=int)
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness_values.append(np.mean(gray))
                
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                color_variances.append(np.var(hsv[:, :, 0]))  # Hue variance
        
        cap.release()
        
        # Calculate statistics
        stats = np.array([
            duration,  # Video duration
            fps,  # Frame rate
            width,  # Width
            height,  # Height
            width * height,  # Resolution (pixels)
            len(brightness_values),  # Sampled frames
            np.mean(brightness_values) if brightness_values else 0,  # Mean brightness
            np.std(brightness_values) if brightness_values else 0,  # Brightness variation
            np.mean(color_variances) if color_variances else 0,  # Mean color variance
            np.std(color_variances) if color_variances else 0  # Color variance variation
        ])
        
        return stats.astype(np.float32)
    
    def extract_all_features(self, video_path: str) -> Dict[str, np.ndarray]:
        """
        Extract all features from a video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing all feature vectors
        """
        # Extract keyframes
        keyframes, timestamps = self.extract_keyframes(video_path)
        
        # Extract features
        keyframe_features = self.extract_keyframe_features(keyframes)
        motion_features = self.extract_motion_features(video_path)
        scene_stats = self.extract_scene_stats(video_path)
        
        # Create combined feature vector
        combined = np.concatenate([
            self._normalize(keyframe_features),
            self._normalize(motion_features),
            self._normalize(scene_stats)
        ])
        
        return {
            'keyframe_features': keyframe_features,
            'motion_features': motion_features,
            'scene_stats': scene_stats,
            'combined_features': combined,
            'keyframe_timestamps': timestamps
        }
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """L2 normalize a vector"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def get_video_metadata(self, video_path: str) -> Dict:
        """Get video duration and other metadata"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'duration': duration,
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames
        }
    
    def generate_thumbnails(self, video_path: str, output_dir: str, 
                           size: Tuple[int, int] = (256, 256)) -> List[str]:
        """
        Generate thumbnail images from video keyframes
        
        Args:
            video_path: Source video path
            output_dir: Output directory for thumbnails
            size: Thumbnail size (width, height)
            
        Returns:
            List of paths to generated thumbnails
        """
        keyframes, timestamps = self.extract_keyframes(video_path)
        thumbnail_paths = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, frame in enumerate(keyframes):
            # Resize
            thumbnail = cv2.resize(frame, size)
            
            # Save
            filename = f"frame_{i}.jpg"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
            thumbnail_paths.append(output_path)
        
        return thumbnail_paths
