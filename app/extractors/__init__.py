"""
Feature Extractors Package
"""
from app.extractors.image_extractor import ImageFeatureExtractor
from app.extractors.audio_extractor import AudioFeatureExtractor
from app.extractors.video_extractor import VideoFeatureExtractor

# Aliases for simpler imports
ImageExtractor = ImageFeatureExtractor
AudioExtractor = AudioFeatureExtractor
VideoExtractor = VideoFeatureExtractor

__all__ = [
    'ImageFeatureExtractor', 'AudioFeatureExtractor', 'VideoFeatureExtractor',
    'ImageExtractor', 'AudioExtractor', 'VideoExtractor'
]

