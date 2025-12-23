"""
Image Feature Extractor - PyTorch Version
Extracts color histogram, texture (LBP), and deep features from images
Supports GPU acceleration with CPU fallback
"""
import numpy as np
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern
from typing import Dict, Tuple, Optional
import os
import logging

logger = logging.getLogger(__name__)


class ImageFeatureExtractor:
    """Extract features from images for similarity search using PyTorch"""
    
    def __init__(self, use_deep_features: bool = True, force_cpu: bool = False):
        """
        Initialize image feature extractor
        
        Args:
            use_deep_features: Whether to use MobileNetV2 for deep features
            force_cpu: If True, force CPU even if GPU is available
        """
        self.use_deep_features = use_deep_features
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
            logger.info(f"Image extractor using: {get_device_summary()}")
            
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
            
            logger.info(f"MobileNetV2 loaded successfully on {self.device}")
            
        except ImportError as e:
            logger.warning(f"PyTorch/torchvision not available: {e}")
            self.use_deep_features = False
        except Exception as e:
            logger.error(f"Could not load MobileNetV2: {e}")
            self.use_deep_features = False
    
    def extract_color_histogram(self, image: np.ndarray, bins: int = 64) -> np.ndarray:
        """
        Extract color histogram from HSV color space
        
        Args:
            image: BGR image (OpenCV format)
            bins: Number of bins per channel
            
        Returns:
            Flattened histogram vector (192 dimensions for 64 bins × 3 channels)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        
        # Normalize histograms
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Concatenate
        histogram = np.concatenate([hist_h, hist_s, hist_v])
        
        return histogram.astype(np.float32)
    
    def extract_texture_lbp(self, image: np.ndarray, radius: int = 3, n_points: int = 24) -> np.ndarray:
        """
        Extract texture features using Local Binary Patterns
        
        Args:
            image: BGR image
            radius: Radius of the LBP
            n_points: Number of points in the LBP
            
        Returns:
            LBP histogram (256 dimensions)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate LBP
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Calculate histogram
        n_bins = 256
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        return hist.astype(np.float32)
    
    def extract_deep_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract deep features using MobileNetV2 (PyTorch)
        
        Args:
            image: BGR image
            
        Returns:
            Feature vector (1280 dimensions) or None if model not loaded
        """
        if not self.use_deep_features or self.model is None:
            return None
        
        try:
            import torch
            
            # Convert BGR to RGB and to PIL Image
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            
            # Apply transforms
            img_tensor = self.transform(pil_image).unsqueeze(0)  # Add batch dimension
            img_tensor = img_tensor.to(self.device)
            
            # Extract features (no gradient computation needed)
            with torch.no_grad():
                features = self.model(img_tensor)
            
            # Move to CPU and convert to numpy
            features_np = features.cpu().numpy().flatten()
            
            return features_np.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting deep features: {e}")
            return None
    
    def extract_all_features(self, image_path: str) -> Dict[str, np.ndarray]:
        """
        Extract all features from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing all feature vectors
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Extract features
        color_hist = self.extract_color_histogram(image)
        texture_lbp = self.extract_texture_lbp(image)
        deep_features = self.extract_deep_features(image)
        
        # Create combined feature vector
        if deep_features is not None:
            combined = np.concatenate([
                self._normalize(color_hist),
                self._normalize(texture_lbp),
                self._normalize(deep_features)
            ])
        else:
            # If no deep features, pad with zeros
            combined = np.concatenate([
                self._normalize(color_hist),
                self._normalize(texture_lbp),
                np.zeros(1280, dtype=np.float32)
            ])
        
        return {
            'color_histogram': color_hist,
            'texture_lbp': texture_lbp,
            'deep_features': deep_features,
            'combined_features': combined
        }
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """L2 normalize a vector"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def get_image_metadata(self, image_path: str) -> Dict:
        """Get image dimensions and other metadata"""
        with Image.open(image_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode
            }
    
    def generate_thumbnail(self, image_path: str, output_path: str, size: Tuple[int, int] = (256, 256)) -> str:
        """
        Generate a thumbnail for the image
        
        Args:
            image_path: Source image path
            output_path: Output thumbnail path
            size: Thumbnail size (width, height)
            
        Returns:
            Path to the generated thumbnail
        """
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Create thumbnail maintaining aspect ratio
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save thumbnail
            img.save(output_path, 'JPEG', quality=85)
        
        return output_path
