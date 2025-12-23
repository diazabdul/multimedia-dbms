"""
Dataset Population Script for Multimedia DBMS
Populates the database with sample datasets:
- CIFAR-100: 500 images (auto-download via TensorFlow)
- ESC-50: 500 audio clips (auto-download from GitHub)
- Synthetic Videos: 500 video clips (generated programmatically)

Usage:
    python scripts/populate_datasets.py [--images] [--audio] [--video] [--all]
"""

import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import Media, Thumbnail
from app.extractors import ImageExtractor, AudioExtractor, VideoExtractor


def get_cifar100_labels():
    """CIFAR-100 fine labels"""
    return [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
        'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]


def populate_cifar100(app, count=500, batch_size=50):
    """
    Populate database with CIFAR-100 images
    
    Args:
        app: Flask application instance
        count: Number of images to add
        batch_size: Process in batches to avoid memory issues
    """
    print(f"\n{'='*60}")
    print(f"Populating CIFAR-100 Images ({count} samples)")
    print(f"{'='*60}")
    
    try:
        import torchvision.datasets as datasets
        from PIL import Image
        import numpy as np
    except ImportError:
        print("Error: torchvision and Pillow required. Install with:")
        print("  pip install torchvision pillow")
        return 0
    
    # Download CIFAR-100 dataset using torchvision
    print("Downloading CIFAR-100 dataset...")
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)
    
    cifar100 = datasets.CIFAR100(root=datasets_dir, train=True, download=True)
    labels = get_cifar100_labels()
    
    # Limit to requested count
    count = min(count, len(cifar100))
    
    upload_folder = app.config['UPLOAD_FOLDER']
    image_folder = os.path.join(upload_folder, 'image')
    thumb_folder = app.config['THUMBNAIL_FOLDER']
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(thumb_folder, exist_ok=True)
    
    extractor = ImageExtractor()
    added = 0
    
    with app.app_context():
        for i in range(0, count, batch_size):
            batch_end = min(i + batch_size, count)
            print(f"Processing batch {i//batch_size + 1}/{(count + batch_size - 1)//batch_size} ({i}-{batch_end})...")
            
            for j in range(i, batch_end):
                try:
                    # Get image and label from torchvision dataset
                    img, label_idx = cifar100[j]  # Returns (PIL Image, label)
                    label = labels[label_idx]
                    
                    # Upscale from 32x32 to 256x256 for better quality
                    img = img.resize((256, 256), Image.Resampling.LANCZOS)
                    
                    # Save image
                    filename = f"cifar100_{j:05d}_{label}.png"
                    filepath = os.path.join(image_folder, filename)
                    img.save(filepath, 'PNG')
                    
                    file_size = os.path.getsize(filepath)
                    
                    # Create media record
                    media = Media(
                        filename=filename,
                        original_filename=filename,
                        file_path=filepath,
                        media_type='image',
                        mime_type='image/png',
                        file_size=file_size,
                        title=f"CIFAR-100: {label.replace('_', ' ').title()}",
                        description=f"Sample image from CIFAR-100 dataset - {label}",
                        tags=['cifar100', label, 'sample', 'dataset'],
                        width=256,
                        height=256,
                        is_processed=False
                    )
                    db.session.add(media)
                    db.session.flush()  # Get the ID
                    
                    # Extract features
                    try:
                        features = extractor.extract_all_features(filepath)
                        
                        from app.models import ImageFeatures
                        image_features = ImageFeatures(
                            media_id=media.id,
                            color_histogram=features['color_histogram'],
                            texture_lbp=features['texture_lbp'],
                            deep_features=features.get('deep_features'),
                            combined_features=features['combined_features']
                        )
                        db.session.add(image_features)
                        
                        # Generate thumbnail
                        thumb_filename = f"thumb_cifar100_{j:05d}.jpg"
                        thumb_path = os.path.join(thumb_folder, thumb_filename)
                        extractor.generate_thumbnail(filepath, thumb_path, (256, 256))
                        
                        thumbnail = Thumbnail(
                            media_id=media.id,
                            thumbnail_path=thumb_path,
                            thumbnail_type='default',
                            width=256,
                            height=256
                        )
                        db.session.add(thumbnail)
                        
                        media.is_processed = True
                        added += 1
                        
                    except Exception as e:
                        print(f"  Warning: Feature extraction failed for {filename}: {e}")
                        media.processing_error = str(e)
                        added += 1
                    
                except Exception as e:
                    print(f"  Error processing image {j}: {e}")
                    continue
            
            # Commit batch
            db.session.commit()
            print(f"  Committed batch. Total added: {added}")
    
    print(f"\nCIFAR-100 population complete: {added} images added")
    return added


def download_esc50(target_dir):
    """
    Download ESC-50 dataset from GitHub
    
    ESC-50: Dataset for Environmental Sound Classification
    https://github.com/karolpiczak/ESC-50
    
    Returns: Path to extracted audio folder
    """
    import urllib.request
    import zipfile
    
    esc50_url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = os.path.join(target_dir, "ESC-50-master.zip")
    extract_path = os.path.join(target_dir, "ESC-50-master")
    audio_path = os.path.join(extract_path, "audio")
    
    # Check if already downloaded
    if os.path.exists(audio_path):
        print("ESC-50 already downloaded.")
        return audio_path
    
    os.makedirs(target_dir, exist_ok=True)
    
    print("Downloading ESC-50 dataset (~600MB)...")
    print("Source: https://github.com/karolpiczak/ESC-50")
    
    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            print(f"\r  Downloading: {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end='')
        
        urllib.request.urlretrieve(esc50_url, zip_path, report_progress)
        print("\n  Download complete!")
        
        # Extract
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        
        # Clean up zip
        os.remove(zip_path)
        print("  Extraction complete!")
        
        return audio_path
        
    except Exception as e:
        print(f"\nError downloading ESC-50: {e}")
        return None


def get_esc50_metadata():
    """ESC-50 category labels (50 categories, 5 major groups)"""
    return {
        # Animals (0-9)
        0: 'dog', 1: 'rooster', 2: 'pig', 3: 'cow', 4: 'frog',
        5: 'cat', 6: 'hen', 7: 'insects', 8: 'sheep', 9: 'crow',
        # Natural soundscapes (10-19)
        10: 'rain', 11: 'sea_waves', 12: 'crackling_fire', 13: 'crickets', 14: 'chirping_birds',
        15: 'water_drops', 16: 'wind', 17: 'pouring_water', 18: 'toilet_flush', 19: 'thunderstorm',
        # Human non-speech (20-29)
        20: 'crying_baby', 21: 'sneezing', 22: 'clapping', 23: 'breathing', 24: 'coughing',
        25: 'footsteps', 26: 'laughing', 27: 'brushing_teeth', 28: 'snoring', 29: 'drinking_sipping',
        # Interior/domestic (30-39)
        30: 'door_knock', 31: 'mouse_click', 32: 'keyboard_typing', 33: 'door_wood_creaks', 34: 'can_opening',
        35: 'washing_machine', 36: 'vacuum_cleaner', 37: 'clock_alarm', 38: 'clock_tick', 39: 'glass_breaking',
        # Exterior/urban (40-49)
        40: 'helicopter', 41: 'chainsaw', 42: 'siren', 43: 'car_horn', 44: 'engine',
        45: 'train', 46: 'church_bells', 47: 'airplane', 48: 'fireworks', 49: 'hand_saw'
    }


def populate_esc50(app, count=500, batch_size=50):
    """
    Populate database with ESC-50 environmental audio clips
    
    ESC-50 contains 2000 audio clips (5 seconds each) across 50 categories.
    Auto-downloads from GitHub if not present.
    
    Args:
        app: Flask application instance
        count: Number of audio clips to add (max 2000)
        batch_size: Process in batches
    """
    print(f"\n{'='*60}")
    print(f"Populating ESC-50 Audio Samples ({count} samples)")
    print(f"{'='*60}")
    
    try:
        import numpy as np
        import librosa
    except ImportError:
        print("Error: numpy and librosa required. Install with:")
        print("  pip install numpy librosa")
        return 0
    
    # Download ESC-50 if needed
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')
    audio_path = download_esc50(datasets_dir)
    
    if not audio_path or not os.path.exists(audio_path):
        print("Failed to download ESC-50. Aborting audio population.")
        return 0
    
    # Get all audio files
    import glob
    audio_files = sorted(glob.glob(os.path.join(audio_path, '*.wav')))
    
    if not audio_files:
        print(f"No audio files found in {audio_path}")
        return 0
    
    print(f"Found {len(audio_files)} audio files in ESC-50")
    
    # Limit to requested count
    audio_files = audio_files[:min(count, len(audio_files))]
    
    upload_folder = app.config['UPLOAD_FOLDER']
    audio_folder = os.path.join(upload_folder, 'audio')
    thumb_folder = app.config['THUMBNAIL_FOLDER']
    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(thumb_folder, exist_ok=True)
    
    extractor = AudioExtractor()
    categories = get_esc50_metadata()
    added = 0
    
    with app.app_context():
        for i, src_path in enumerate(audio_files):
            try:
                # Parse ESC-50 filename format: {fold}-{clip_id}-{take}-{target}.wav
                # Example: 1-100032-A-0.wav
                basename = os.path.basename(src_path)
                parts = basename.replace('.wav', '').split('-')
                
                if len(parts) >= 4:
                    fold = parts[0]
                    clip_id = parts[1]
                    take = parts[2]
                    category_id = int(parts[3])
                    category = categories.get(category_id, 'unknown')
                else:
                    category = 'unknown'
                    category_id = -1
                
                # Copy file to uploads folder
                filename = f"esc50_{i:05d}_{category}.wav"
                filepath = os.path.join(audio_folder, filename)
                shutil.copy(src_path, filepath)
                
                file_size = os.path.getsize(filepath)
                
                # Create media record
                media = Media(
                    filename=filename,
                    original_filename=basename,
                    file_path=filepath,
                    media_type='audio',
                    mime_type='audio/wav',
                    file_size=file_size,
                    title=f"ESC-50: {category.replace('_', ' ').title()}",
                    description=f"Environmental sound from ESC-50 dataset - {category}",
                    tags=['esc50', category, 'environmental', 'audio', 'sample'],
                    duration=5.0,  # ESC-50 clips are always 5 seconds
                    is_processed=False
                )
                db.session.add(media)
                db.session.flush()
                
                # Extract features
                try:
                    features = extractor.extract_all_features(filepath)
                    
                    from app.models import AudioFeatures
                    audio_features = AudioFeatures(
                        media_id=media.id,
                        mfcc_features=features['mfcc_features'],
                        spectral_features=features['spectral_features'],
                        waveform_stats=features['waveform_stats'],
                        combined_features=features['combined_features']
                    )
                    db.session.add(audio_features)
                    
                    # Generate waveform thumbnail
                    thumb_filename = f"waveform_esc50_{i:05d}.png"
                    thumb_path = os.path.join(thumb_folder, thumb_filename)
                    extractor.generate_waveform_image(filepath, thumb_path)
                    
                    thumbnail = Thumbnail(
                        media_id=media.id,
                        thumbnail_path=thumb_path,
                        thumbnail_type='default',
                        width=512,
                        height=256
                    )
                    db.session.add(thumbnail)
                    
                    media.is_processed = True
                    added += 1
                    
                except Exception as e:
                    print(f"  Warning: Feature extraction failed for {basename}: {e}")
                    media.processing_error = str(e)
                    added += 1
                
                if (i + 1) % batch_size == 0:
                    db.session.commit()
                    print(f"  Progress: {i+1}/{len(audio_files)} audio samples processed")
                    
            except Exception as e:
                print(f"  Error processing {src_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        db.session.commit()
    
    print(f"\nESC-50 population complete: {added} audio samples added")
    return added


def populate_from_urbansound(app, urbansound_path, count, batch_size):
    """Populate from actual UrbanSound8K dataset"""
    import glob
    
    # Find all audio files
    audio_files = glob.glob(os.path.join(urbansound_path, '**', '*.wav'), recursive=True)
    audio_files = audio_files[:count]
    
    extractor = AudioExtractor()
    added = 0
    
    upload_folder = app.config['UPLOAD_FOLDER']
    audio_folder = os.path.join(upload_folder, 'audio')
    thumb_folder = app.config['THUMBNAIL_FOLDER']
    
    urbansound_labels = [
        'air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
        'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
    ]
    
    with app.app_context():
        for i, src_path in enumerate(audio_files):
            try:
                # Copy file to uploads
                basename = os.path.basename(src_path)
                filename = f"urbansound_{i:05d}_{basename}"
                filepath = os.path.join(audio_folder, filename)
                shutil.copy(src_path, filepath)
                
                file_size = os.path.getsize(filepath)
                
                # Try to get label from filename (UrbanSound8K format)
                parts = basename.split('-')
                label_idx = int(parts[-1].split('.')[0]) if len(parts) > 1 else 0
                label = urbansound_labels[label_idx] if label_idx < len(urbansound_labels) else 'unknown'
                
                # Create media record
                media = Media(
                    filename=filename,
                    original_filename=basename,
                    file_path=filepath,
                    media_type='audio',
                    mime_type='audio/wav',
                    file_size=file_size,
                    title=f"UrbanSound8K: {label.replace('_', ' ').title()}",
                    description=f"Urban sound sample - {label}",
                    tags=['urbansound8k', label, 'sample', 'urban'],
                    is_processed=False
                )
                db.session.add(media)
                db.session.flush()
                
                # Extract features and create thumbnail
                try:
                    features = extractor.extract_all_features(filepath)
                    metadata = extractor.get_audio_metadata(filepath)
                    media.duration = metadata.get('duration')
                    
                    from app.models import AudioFeatures
                    audio_features = AudioFeatures(
                        media_id=media.id,
                        mfcc_features=features['mfcc_features'],
                        spectral_features=features['spectral_features'],
                        waveform_stats=features['waveform_stats'],
                        combined_features=features['combined_features']
                    )
                    db.session.add(audio_features)
                    
                    thumb_filename = f"waveform_urban_{i:05d}.png"
                    thumb_path = os.path.join(thumb_folder, thumb_filename)
                    extractor.generate_waveform_image(filepath, thumb_path)
                    
                    thumbnail = Thumbnail(
                        media_id=media.id,
                        thumbnail_path=thumb_path,
                        thumbnail_type='default',
                        width=512,
                        height=256
                    )
                    db.session.add(thumbnail)
                    
                    media.is_processed = True
                    added += 1
                    
                except Exception as e:
                    media.processing_error = str(e)
                    added += 1
                
                if (i + 1) % batch_size == 0:
                    db.session.commit()
                    print(f"  Progress: {i+1}/{count} audio samples processed")
                    
            except Exception as e:
                print(f"  Error processing {src_path}: {e}")
                continue
        
        db.session.commit()
    
    return added


def populate_videos(app, count=500, batch_size=10):
    """
    Populate database with sample videos
    
    Note: YouTube-8M contains features, not actual videos.
    This function generates synthetic test videos with various patterns.
    
    Args:
        app: Flask application instance
        count: Number of videos to add
        batch_size: Process in batches (smaller for videos due to size)
    """
    print(f"\n{'='*60}")
    print(f"Populating Video Samples ({count} samples)")
    print(f"{'='*60}")
    
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Error: opencv-python and numpy required. Install with:")
        print("  pip install opencv-python numpy")
        return 0
    
    upload_folder = app.config['UPLOAD_FOLDER']
    video_folder = os.path.join(upload_folder, 'video')
    thumb_folder = app.config['THUMBNAIL_FOLDER']
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(thumb_folder, exist_ok=True)
    
    video_types = [
        ('color_gradient', 'Color Gradient Animation'),
        ('moving_shapes', 'Moving Geometric Shapes'),
        ('noise_pattern', 'Dynamic Noise Pattern'),
        ('wave_effect', 'Wave Effect Animation'),
        ('rotating_square', 'Rotating Square'),
        ('bouncing_ball', 'Bouncing Ball'),
        ('color_pulse', 'Color Pulsing'),
        ('checkerboard', 'Animated Checkerboard'),
        ('spiral', 'Spiral Animation'),
        ('particles', 'Particle System')
    ]
    
    extractor = VideoExtractor()
    added = 0
    
    # Video settings
    width, height = 320, 240
    fps = 24
    duration = 3  # 3 seconds per video
    total_frames = fps * duration
    
    with app.app_context():
        for i in range(count):
            try:
                video_type, video_name = video_types[i % len(video_types)]
                
                filename = f"synth_video_{i:05d}_{video_type}.mp4"
                filepath = os.path.join(video_folder, filename)
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
                
                # Generate frames based on video type
                for frame_idx in range(total_frames):
                    t = frame_idx / total_frames
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    if video_type == 'color_gradient':
                        for y in range(height):
                            for x in range(width):
                                frame[y, x] = [
                                    int(255 * (x / width)),
                                    int(255 * (y / height)),
                                    int(128 + 127 * np.sin(2 * np.pi * t))
                                ]
                    
                    elif video_type == 'moving_shapes':
                        cx = int(width * (0.3 + 0.4 * np.sin(2 * np.pi * t)))
                        cy = int(height * (0.3 + 0.4 * np.cos(2 * np.pi * t)))
                        cv2.circle(frame, (cx, cy), 30, (255, 100, 50), -1)
                        cv2.rectangle(frame, (width-80, 20), (width-20, 80), (50, 255, 100), -1)
                    
                    elif video_type == 'noise_pattern':
                        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                        frame = cv2.GaussianBlur(frame, (5, 5), 0)
                    
                    elif video_type == 'wave_effect':
                        for y in range(height):
                            offset = int(20 * np.sin(2 * np.pi * (y / 30 + t * 2)))
                            hue = int(180 * t + y) % 180
                            frame[y, :] = [hue, 255, 200]
                        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
                    
                    elif video_type == 'rotating_square':
                        center = (width // 2, height // 2)
                        angle = 360 * t
                        size = 50
                        pts = np.array([
                            [center[0] + size * np.cos(np.radians(angle + a)), 
                             center[1] + size * np.sin(np.radians(angle + a))]
                            for a in [0, 90, 180, 270]
                        ], np.int32)
                        cv2.fillPoly(frame, [pts], (100, 150, 255))
                    
                    elif video_type == 'bouncing_ball':
                        x = int(width * abs(np.sin(np.pi * t * 2)))
                        y = int(height * 0.8 * abs(np.sin(np.pi * t * 3)))
                        cv2.circle(frame, (x, height - y - 20), 20, (0, 200, 255), -1)
                    
                    elif video_type == 'color_pulse':
                        intensity = int(127 + 127 * np.sin(2 * np.pi * t * 3))
                        frame[:, :] = [intensity, 255 - intensity, 128]
                    
                    elif video_type == 'checkerboard':
                        cell_size = 40
                        offset = int(cell_size * t * 2) % (cell_size * 2)
                        for y in range(0, height, cell_size):
                            for x in range(0, width, cell_size):
                                if ((x + offset) // cell_size + y // cell_size) % 2 == 0:
                                    cv2.rectangle(frame, (x, y), (x + cell_size, y + cell_size), (255, 255, 255), -1)
                    
                    elif video_type == 'spiral':
                        for r in range(10, min(width, height) // 2, 5):
                            angle = r / 10 + t * 10
                            x = int(width / 2 + r * np.cos(angle))
                            y = int(height / 2 + r * np.sin(angle))
                            color = ((r * 3) % 256, (r * 5) % 256, (r * 7) % 256)
                            cv2.circle(frame, (x, y), 3, color, -1)
                    
                    elif video_type == 'particles':
                        np.random.seed(i * 1000 + frame_idx)
                        for _ in range(50):
                            x = np.random.randint(0, width)
                            y = np.random.randint(0, height)
                            size = np.random.randint(2, 6)
                            color = tuple(np.random.randint(100, 256, 3).tolist())
                            cv2.circle(frame, (x, y), size, color, -1)
                    
                    out.write(frame)
                
                out.release()
                
                file_size = os.path.getsize(filepath)
                
                # Create media record
                media = Media(
                    filename=filename,
                    original_filename=filename,
                    file_path=filepath,
                    media_type='video',
                    mime_type='video/mp4',
                    file_size=file_size,
                    title=f"Synthetic Video: {video_name} #{i+1}",
                    description=f"Synthetic video sample - {video_name}",
                    tags=['synthetic', video_type, 'sample', 'video'],
                    width=width,
                    height=height,
                    duration=duration,
                    is_processed=False
                )
                db.session.add(media)
                db.session.flush()
                
                # Extract features  
                try:
                    features = extractor.extract_all_features(filepath)
                    
                    from app.models import VideoFeatures
                    video_features = VideoFeatures(
                        media_id=media.id,
                        keyframe_features=features['keyframe_features'],
                        motion_features=features.get('motion_features'),
                        scene_stats=features.get('scene_stats'),
                        combined_features=features['combined_features'],
                        keyframe_timestamps=features.get('keyframe_timestamps')
                    )
                    db.session.add(video_features)
                    
                    # Generate thumbnail (first frame)
                    thumb_filename = f"thumb_video_{i:05d}.jpg"
                    thumb_path = os.path.join(thumb_folder, thumb_filename)
                    
                    cap = cv2.VideoCapture(filepath)
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(thumb_path, frame)
                        thumbnail = Thumbnail(
                            media_id=media.id,
                            thumbnail_path=thumb_path,
                            thumbnail_type='default',
                            width=width,
                            height=height
                        )
                        db.session.add(thumbnail)
                    cap.release()
                    
                    media.is_processed = True
                    added += 1
                    
                except Exception as e:
                    print(f"  Warning: Feature extraction failed for {filename}: {e}")
                    media.processing_error = str(e)
                    added += 1
                
                if (i + 1) % batch_size == 0:
                    db.session.commit()
                    print(f"  Progress: {i+1}/{count} videos processed")
                    
            except Exception as e:
                print(f"  Error generating video {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        db.session.commit()
    
    print(f"\nVideo population complete: {added} videos added")
    return added


def main():
    parser = argparse.ArgumentParser(description='Populate Multimedia DBMS with sample datasets')
    parser.add_argument('--images', action='store_true', help='Populate CIFAR-100 images')
    parser.add_argument('--audio', action='store_true', help='Populate audio samples')
    parser.add_argument('--video', action='store_true', help='Populate video samples')
    parser.add_argument('--all', action='store_true', help='Populate all datasets')
    parser.add_argument('--count', type=int, default=500, help='Number of samples per type (default: 500)')
    
    args = parser.parse_args()
    
    # Default to all if nothing specified
    if not (args.images or args.audio or args.video or args.all):
        args.all = True
    
    if args.all:
        args.images = args.audio = args.video = True
    
    print("="*60)
    print("Multimedia DBMS - Dataset Population Script")
    print("="*60)
    print(f"Target samples per type: {args.count}")
    print(f"Populate images: {args.images}")
    print(f"Populate audio: {args.audio}")
    print(f"Populate video: {args.video}")
    
    # Create Flask app
    app = create_app()
    
    total_added = 0
    
    if args.images:
        total_added += populate_cifar100(app, args.count)
    
    if args.audio:
        total_added += populate_esc50(app, args.count)
    
    if args.video:
        total_added += populate_videos(app, args.count)
    
    print("\n" + "="*60)
    print(f"POPULATION COMPLETE: {total_added} total items added")
    print("="*60)


if __name__ == '__main__':
    main()
