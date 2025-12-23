"""
Audio Feature Extractor
Extracts MFCC, spectral features, and waveform statistics from audio files
"""
import numpy as np
import librosa
from typing import Dict, Tuple, Optional
import os


class AudioFeatureExtractor:
    """Extract features from audio files for similarity search"""
    
    def __init__(self, sr: int = 22050, duration: Optional[float] = None):
        """
        Initialize audio feature extractor
        
        Args:
            sr: Sample rate for loading audio
            duration: Maximum duration in seconds (None for full audio)
        """
        self.sr = sr
        self.duration = duration
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio signal, sample rate)
        """
        y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
        return y, sr
    
    def extract_mfcc(self, y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
        """
        Extract MFCC features with delta and delta-delta
        
        Args:
            y: Audio signal
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            MFCC feature vector (39 dimensions: 13 + 13 delta + 13 delta-delta)
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Calculate deltas
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Take mean across time
        mfcc_mean = np.mean(mfccs, axis=1)
        delta_mean = np.mean(mfcc_delta, axis=1)
        delta2_mean = np.mean(mfcc_delta2, axis=1)
        
        # Concatenate
        features = np.concatenate([mfcc_mean, delta_mean, delta2_mean])
        
        return features.astype(np.float32)
    
    def extract_spectral_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract spectral features
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Spectral feature vector (6 dimensions)
        """
        # Spectral centroid - brightness of sound
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Spectral rolloff - frequency below which 85% of energy is contained
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # Spectral bandwidth - width of the band of frequencies
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        # Spectral contrast - difference between peaks and valleys
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        
        # Spectral flatness - how noise-like vs tonal
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        features = np.array([
            spectral_centroid,
            spectral_rolloff,
            spectral_bandwidth,
            spectral_contrast,
            spectral_flatness,
            zcr
        ])
        
        return features.astype(np.float32)
    
    def extract_waveform_stats(self, y: np.ndarray) -> np.ndarray:
        """
        Extract waveform statistics
        
        Args:
            y: Audio signal
            
        Returns:
            Waveform statistics vector (5 dimensions)
        """
        # RMS energy
        rms = np.sqrt(np.mean(y ** 2))
        
        # Peak amplitude
        peak = np.max(np.abs(y))
        
        # Crest factor (peak to RMS ratio)
        crest_factor = peak / (rms + 1e-10)
        
        # Dynamic range (in dB)
        if peak > 0:
            dynamic_range = 20 * np.log10(peak / (np.min(np.abs(y[y != 0])) + 1e-10))
        else:
            dynamic_range = 0
        
        # Silence ratio (percentage of near-silence samples)
        silence_threshold = 0.01 * peak
        silence_ratio = np.sum(np.abs(y) < silence_threshold) / len(y)
        
        features = np.array([
            rms,
            peak,
            crest_factor,
            dynamic_range,
            silence_ratio
        ])
        
        return features.astype(np.float32)
    
    def extract_all_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract all features from an audio file
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing all feature vectors
        """
        # Load audio
        y, sr = self.load_audio(audio_path)
        
        if len(y) == 0:
            raise ValueError(f"Could not load audio: {audio_path}")
        
        # Extract features
        mfcc = self.extract_mfcc(y, sr)
        spectral = self.extract_spectral_features(y, sr)
        waveform = self.extract_waveform_stats(y)
        
        # Create combined feature vector
        combined = np.concatenate([
            self._normalize(mfcc),
            self._normalize(spectral),
            self._normalize(waveform)
        ])
        
        return {
            'mfcc_features': mfcc,
            'spectral_features': spectral,
            'waveform_stats': waveform,
            'combined_features': combined
        }
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """L2 normalize a vector"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def get_audio_metadata(self, audio_path: str) -> Dict:
        """Get audio duration and other metadata"""
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'samples': len(y)
        }
    
    def generate_waveform_image(self, audio_path: str, output_path: str, 
                                 width: int = 800, height: int = 200) -> str:
        """
        Generate a waveform visualization for the audio
        
        Args:
            audio_path: Source audio path
            output_path: Output image path
            width: Image width
            height: Image height
            
        Returns:
            Path to the generated waveform image
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Load audio
        y, sr = self.load_audio(audio_path)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        # Plot waveform
        times = np.linspace(0, len(y) / sr, len(y))
        ax.plot(times, y, color='#4a90d9', linewidth=0.5)
        ax.fill_between(times, y, alpha=0.3, color='#4a90d9')
        
        # Style
        ax.set_xlim(0, len(y) / sr)
        ax.set_ylim(-1, 1)
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, 
                   facecolor='#1a1a2e', edgecolor='none')
        plt.close()
        
        return output_path
