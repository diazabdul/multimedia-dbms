"""
Unified Dataset Script
Downloads images and videos from Pexels + uploads all data (including CIFAR) to the DBMS
Run with: python setup_dataset.py
"""

import os
import sys
import time
import json
import requests
import subprocess
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================

# Pexels API Key - Get free key at https://www.pexels.com/api/
PEXELS_API_KEY = "wMAiZdBf2crypaR4X3wn2KGifCGWR8AynQtRwgf7vLJhfoOgpMnFT1Hs"

# Directories
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGE_DIR = DATASET_DIR / "images"
VIDEO_DIR = DATASET_DIR / "videos"
CIFAR_DIR = BASE_DIR / "synthetic_dataset"  # Existing CIFAR data
AUDIO_DIR = BASE_DIR / "datasets" / "ESC-50-master" / "audio"  # ESC-50 audio dataset

# Progress tracking
PROGRESS_FILE = BASE_DIR / "dataset_progress.json"

# API base URL
API_BASE = "http://localhost:5000/api"

# ============================================
# IMAGE CATEGORIES (from Pexels)
# ============================================
IMAGE_CATEGORIES = {
    "nature": ["forest landscape", "mountain view", "ocean sunset", "flower garden"],
    "animals": ["dog portrait", "cat face", "bird flying", "wildlife safari"],
    "city": ["city skyline", "street photography", "architecture building", "urban night"],
    "food": ["healthy food", "coffee cup", "dessert cake", "restaurant dish"],
    "people": ["portrait photography", "people working", "friends together", "family moment"],
}

# Images per search term
IMAGES_PER_SEARCH = 3

# ============================================
# VIDEO CATEGORIES (from Pexels)
# ============================================
VIDEO_CATEGORIES = {
    "ocean": ["ocean waves", "beach sunset"],
    "nature": ["forest walk", "waterfall nature"],
    "city": ["city traffic", "urban street"],
    "animals": ["dog playing", "birds flying"],
    "sky": ["clouds timelapse", "sunset sky"],
}

# Videos per search term
VIDEOS_PER_SEARCH = 2

# Max file sizes (MB)
MAX_IMAGE_SIZE_MB = 10
MAX_VIDEO_SIZE_MB = 50


# ============================================
# PROGRESS TRACKING
# ============================================

def load_progress():
    """Load progress from file"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"downloaded": [], "uploaded": []}


def save_progress(progress):
    """Save progress to file"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def clear_progress():
    """Clear progress file"""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


# ============================================
# PEXELS API FUNCTIONS
# ============================================

def pexels_search_images(query, per_page=5):
    """Search for images on Pexels"""
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": per_page, "orientation": "landscape"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("photos", [])
    except Exception as e:
        print(f"  ❌ Error searching images '{query}': {e}")
        return []


def pexels_search_videos(query, per_page=5):
    """Search for videos on Pexels"""
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": per_page, "orientation": "landscape"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get("videos", [])
    except Exception as e:
        print(f"  ❌ Error searching videos '{query}': {e}")
        return []


def get_sd_video_url(video_data):
    """Get SD quality video URL for smaller file size"""
    video_files = video_data.get("video_files", [])
    if not video_files:
        return None
    
    # Sort by height (prefer smaller)
    video_files.sort(key=lambda x: x.get("height", 0))
    
    # Find SD quality (360-540p)
    for vf in video_files:
        height = vf.get("height", 0)
        if vf.get("quality") == "sd" and 360 <= height <= 540:
            return vf.get("link")
    
    # Fallback to smallest
    return video_files[0].get("link")


def download_file(url, save_path, max_size_mb=None):
    """Download a file with optional size limit"""
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        # Check content length if available
        content_length = response.headers.get('content-length')
        if content_length and max_size_mb:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > max_size_mb:
                print(f"    ⚠️ File too large ({size_mb:.1f}MB), skipping...")
                return False
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify file size after download
        actual_size_mb = save_path.stat().st_size / (1024 * 1024)
        if max_size_mb and actual_size_mb > max_size_mb:
            save_path.unlink()
            print(f"    ⚠️ File too large after download ({actual_size_mb:.1f}MB), removed")
            return False
        
        return True
    except Exception as e:
        print(f"    ❌ Download error: {e}")
        return False


# ============================================
# DOWNLOAD FUNCTIONS
# ============================================

def download_pexels_images(progress):
    """Download images from Pexels"""
    print("\n" + "=" * 60)
    print("📷 Downloading Images from Pexels")
    print("=" * 60)
    
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    
    for category, search_terms in IMAGE_CATEGORIES.items():
        cat_dir = IMAGE_DIR / category
        cat_dir.mkdir(exist_ok=True)
        print(f"\n📂 Category: {category}")
        
        for term in search_terms:
            print(f"  🔍 Searching: '{term}'")
            photos = pexels_search_images(term, IMAGES_PER_SEARCH)
            
            for i, photo in enumerate(photos[:IMAGES_PER_SEARCH]):
                # Get medium size image
                src = photo.get("src", {}).get("medium") or photo.get("src", {}).get("original")
                if not src:
                    continue
                
                filename = f"{term.replace(' ', '_')}_{i+1}.jpg"
                save_path = cat_dir / filename
                
                if save_path.exists() or str(save_path) in progress["downloaded"]:
                    print(f"    ⏭️ Exists: {filename}")
                    continue
                
                print(f"    ⬇️ Downloading: {filename}")
                if download_file(src, save_path, MAX_IMAGE_SIZE_MB):
                    downloaded += 1
                    progress["downloaded"].append(str(save_path))
                    save_progress(progress)
                    print(f"    ✅ Downloaded: {filename}")
                
                time.sleep(0.3)
    
    print(f"\n✅ Downloaded {downloaded} new images")
    return downloaded


def download_pexels_videos(progress):
    """Download videos from Pexels"""
    print("\n" + "=" * 60)
    print("📹 Downloading Videos from Pexels")
    print("=" * 60)
    
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    
    for category, search_terms in VIDEO_CATEGORIES.items():
        cat_dir = VIDEO_DIR / category
        cat_dir.mkdir(exist_ok=True)
        print(f"\n📂 Category: {category}")
        
        for term in search_terms:
            print(f"  🔍 Searching: '{term}'")
            videos = pexels_search_videos(term, VIDEOS_PER_SEARCH)
            
            for i, video in enumerate(videos[:VIDEOS_PER_SEARCH]):
                video_url = get_sd_video_url(video)
                if not video_url:
                    continue
                
                filename = f"{term.replace(' ', '_')}_{i+1}.mp4"
                save_path = cat_dir / filename
                
                if save_path.exists() or str(save_path) in progress["downloaded"]:
                    print(f"    ⏭️ Exists: {filename}")
                    continue
                
                print(f"    ⬇️ Downloading: {filename}")
                if download_file(video_url, save_path, MAX_VIDEO_SIZE_MB):
                    size_mb = save_path.stat().st_size / (1024 * 1024)
                    downloaded += 1
                    progress["downloaded"].append(str(save_path))
                    save_progress(progress)
                    print(f"    ✅ Downloaded: {filename} ({size_mb:.1f}MB)")
                
                time.sleep(0.5)
    
    print(f"\n✅ Downloaded {downloaded} new videos")
    return downloaded


# ============================================
# UPLOAD FUNCTIONS
# ============================================

def upload_file(file_path, title, category, media_type, progress):
    """Upload a single file to the DBMS"""
    if str(file_path) in progress["uploaded"]:
        return "skipped"
    
    try:
        with open(file_path, 'rb') as f:
            # Set correct MIME type
            if media_type == "image":
                mime_type = "image/jpeg"
            elif media_type == "video":
                mime_type = "video/mp4"
            elif media_type == "audio":
                mime_type = "audio/wav"
            else:
                mime_type = "application/octet-stream"
            files = {'file': (file_path.name, f, mime_type)}
            data = {
                'title': title,
                'description': f'{media_type.title()} from {category}',
                'tags': f'{category},{media_type}'
            }
            
            response = requests.post(
                f"{API_BASE}/upload",
                files=files,
                data=data,
                timeout=600  # 10 minutes for videos
            )
            
            if response.status_code in [200, 201]:
                progress["uploaded"].append(str(file_path))
                save_progress(progress)
                return "success"
            else:
                try:
                    error = response.json().get('error', '')
                    if 'already exists' in error.lower():
                        progress["uploaded"].append(str(file_path))
                        save_progress(progress)
                        return "exists"
                except:
                    pass
                return f"error: {response.status_code}"
    except requests.exceptions.Timeout:
        return "timeout"
    except Exception as e:
        return f"error: {e}"


def upload_pexels_images(progress):
    """Upload Pexels images to DBMS"""
    print("\n" + "=" * 60)
    print("📤 Uploading Pexels Images")
    print("=" * 60)
    
    if not IMAGE_DIR.exists():
        print("❌ No images to upload. Run download first.")
        return 0, 0
    
    uploaded = 0
    skipped = 0
    
    for cat_dir in IMAGE_DIR.iterdir():
        if not cat_dir.is_dir():
            continue
        
        category = cat_dir.name
        print(f"\n📂 Category: {category}")
        
        for img_file in cat_dir.glob("*.jpg"):
            title = img_file.stem.replace("_", " ").title()
            print(f"  ⬆️ {img_file.name}...", end=" ")
            
            result = upload_file(img_file, title, category, "image", progress)
            
            if result == "success":
                print("✅")
                uploaded += 1
            elif result in ["skipped", "exists"]:
                print("⏭️")
                skipped += 1
            else:
                print(f"❌ {result}")
            
            time.sleep(0.5)
    
    print(f"\n✅ Uploaded: {uploaded}, Skipped: {skipped}")
    return uploaded, skipped


def upload_pexels_videos(progress):
    """Upload Pexels videos to DBMS"""
    print("\n" + "=" * 60)
    print("📤 Uploading Pexels Videos")
    print("=" * 60)
    
    if not VIDEO_DIR.exists():
        print("❌ No videos to upload. Run download first.")
        return 0, 0
    
    uploaded = 0
    skipped = 0
    
    for cat_dir in VIDEO_DIR.iterdir():
        if not cat_dir.is_dir():
            continue
        
        category = cat_dir.name
        print(f"\n📂 Category: {category}")
        
        for video_file in cat_dir.glob("*.mp4"):
            title = video_file.stem.replace("_", " ").title()
            size_mb = video_file.stat().st_size / (1024 * 1024)
            print(f"  ⬆️ {video_file.name} ({size_mb:.1f}MB)...", end=" ", flush=True)
            
            result = upload_file(video_file, title, category, "video", progress)
            
            if result == "success":
                print("✅")
                uploaded += 1
            elif result in ["skipped", "exists"]:
                print("⏭️")
                skipped += 1
            elif result == "timeout":
                print("⏰ (timeout, will retry)")
            else:
                print(f"❌ {result}")
            
            time.sleep(1)
    
    print(f"\n✅ Uploaded: {uploaded}, Skipped: {skipped}")
    return uploaded, skipped


def upload_cifar_images(progress):
    """Upload existing CIFAR images to DBMS"""
    print("\n" + "=" * 60)
    print("📤 Uploading CIFAR-10 Images")
    print("=" * 60)
    
    if not CIFAR_DIR.exists():
        print("❌ No CIFAR data found at:", CIFAR_DIR)
        return 0, 0
    
    uploaded = 0
    skipped = 0
    
    # Find all images in CIFAR directory
    for cat_dir in CIFAR_DIR.iterdir():
        if not cat_dir.is_dir():
            continue
        
        category = cat_dir.name
        print(f"\n📂 Category: {category}")
        
        images = list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg"))
        total = len(images)
        
        for i, img_file in enumerate(images):
            title = f"CIFAR {category.replace('_', ' ').title()} {i+1}"
            
            if str(img_file) in progress["uploaded"]:
                skipped += 1
                continue
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{total}] Uploading {category}...")
            
            result = upload_file(img_file, title, f"cifar-{category}", "image", progress)
            
            if result == "success":
                uploaded += 1
            elif result in ["skipped", "exists"]:
                skipped += 1
            
            time.sleep(0.2)
    
    print(f"\n✅ Uploaded: {uploaded}, Skipped: {skipped}")
    return uploaded, skipped


def upload_esc50_audio(progress, max_files=100):
    """Upload ESC-50 audio files to DBMS"""
    print("\n" + "=" * 60)
    print("🎵 Uploading ESC-50 Audio Files")
    print("=" * 60)
    
    if not AUDIO_DIR.exists():
        print(f"❌ No audio data found at: {AUDIO_DIR}")
        return 0, 0
    
    uploaded = 0
    skipped = 0
    
    # Get all .wav files
    audio_files = list(AUDIO_DIR.glob("*.wav"))
    total = min(len(audio_files), max_files)  # Limit number of files
    
    print(f"📁 Found {len(audio_files)} audio files, uploading first {total}")
    
    for i, audio_file in enumerate(audio_files[:max_files]):
        # Extract category from filename (ESC-50 format: fold-id-category.wav)
        # e.g., 1-100032-A-0.wav where last number is category
        parts = audio_file.stem.split('-')
        if len(parts) >= 4:
            category_id = parts[-1]
            # Map category IDs to names (simplified)
            category_names = {
                '0': 'dog', '1': 'rooster', '2': 'pig', '3': 'cow', '4': 'frog',
                '5': 'cat', '6': 'hen', '7': 'insects', '8': 'sheep', '9': 'crow',
                '10': 'rain', '11': 'sea_waves', '12': 'crackling_fire', '13': 'crickets', '14': 'chirping_birds',
                '15': 'water_drops', '16': 'wind', '17': 'pouring_water', '18': 'toilet_flush', '19': 'thunderstorm',
                '20': 'crying_baby', '21': 'sneezing', '22': 'clapping', '23': 'breathing', '24': 'coughing',
                '25': 'footsteps', '26': 'laughing', '27': 'brushing_teeth', '28': 'snoring', '29': 'drinking',
                '30': 'door_knock', '31': 'mouse_click', '32': 'keyboard', '33': 'door_bell', '34': 'can_opening',
                '35': 'engine', '36': 'train', '37': 'church_bells', '38': 'airplane', '39': 'fireworks',
                '40': 'hand_saw', '41': 'vacuum', '42': 'clock_alarm', '43': 'clock_tick', '44': 'glass_breaking',
                '45': 'helicopter', '46': 'chainsaw', '47': 'siren', '48': 'car_horn', '49': 'claps'
            }
            category = category_names.get(category_id, f'sound_{category_id}')
        else:
            category = 'audio'
        
        title = f"ESC-50 {category.replace('_', ' ').title()} {i+1}"
        
        if str(audio_file) in progress["uploaded"]:
            skipped += 1
            continue
        
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{total}] Uploading audio files...")
        
        result = upload_file(audio_file, title, f"esc50-{category}", "audio", progress)
        
        if result == "success":
            uploaded += 1
        elif result in ["skipped", "exists"]:
            skipped += 1
        
        time.sleep(0.3)
    
    print(f"\n✅ Uploaded: {uploaded}, Skipped: {skipped}")
    return uploaded, skipped


# ============================================
# MAIN FUNCTIONS
# ============================================

def download_all():
    """Download all datasets"""
    print("\n" + "=" * 60)
    print("📥 DOWNLOADING ALL DATASETS")
    print("=" * 60)
    
    progress = load_progress()
    
    # Download images
    img_count = download_pexels_images(progress)
    
    # Download videos
    vid_count = download_pexels_videos(progress)
    
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  📷 Pexels Images: {img_count}")
    print(f"  📹 Pexels Videos: {vid_count}")
    print(f"  📁 CIFAR-10: (existing)")


def upload_all():
    """Upload all datasets to DBMS"""
    print("\n" + "=" * 60)
    print("📤 UPLOADING ALL DATASETS TO DBMS")
    print("=" * 60)
    print("⚠️ Make sure Flask server is running (python run.py)")
    
    progress = load_progress()
    
    # Test server connection
    try:
        response = requests.get(f"{API_BASE.replace('/api', '')}/", timeout=5)
        print("✅ Server is running")
    except:
        print("❌ Cannot connect to server. Start it with: python run.py")
        return
    
    # Upload CIFAR images
    cifar_up, cifar_skip = upload_cifar_images(progress)
    
    # Upload Pexels images
    img_up, img_skip = upload_pexels_images(progress)
    
    # Upload ESC-50 audio (limit to 100 files for reasonable time)
    audio_up, audio_skip = upload_esc50_audio(progress, max_files=100)
    
    # Upload Pexels videos
    vid_up, vid_skip = upload_pexels_videos(progress)
    
    print("\n" + "=" * 60)
    print("📊 UPLOAD SUMMARY")
    print("=" * 60)
    print(f"  📷 CIFAR Images: {cifar_up} uploaded, {cifar_skip} skipped")
    print(f"  📷 Pexels Images: {img_up} uploaded, {img_skip} skipped")
    print(f"  🎵 ESC-50 Audio: {audio_up} uploaded, {audio_skip} skipped")
    print(f"  📹 Pexels Videos: {vid_up} uploaded, {vid_skip} skipped")
    print(f"  📁 Total uploaded: {cifar_up + img_up + audio_up + vid_up}")


def reset_database():
    """Reset the database (delete all media)"""
    print("\n" + "=" * 60)
    print("🗑️ RESETTING DATABASE")
    print("=" * 60)
    
    confirm = input("⚠️ This will DELETE all media from the database. Continue? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Cancelled.")
        return
    
    try:
        # Run database reset
        result = subprocess.run([
            sys.executable, "-c",
            "from app import create_app, db; "
            "from app.models import Media, Thumbnail, ImageFeatures, VideoFeatures, AudioFeatures; "
            "app = create_app(); "
            "app.app_context().push(); "
            "Thumbnail.query.delete(); "
            "ImageFeatures.query.delete(); "
            "VideoFeatures.query.delete(); "
            "AudioFeatures.query.delete(); "
            "count = Media.query.count(); "
            "Media.query.delete(); "
            "db.session.commit(); "
            "print(f'Deleted {count} media records')"
        ], capture_output=True, text=True, cwd=str(BASE_DIR))
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        # Clear progress
        clear_progress()
        print("✅ Progress cleared")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🗄️  MULTIMEDIA DATASET MANAGER")
    print("=" * 60)
    print("\nChoose an option:")
    print("  1. Download datasets (Pexels images + videos)")
    print("  2. Upload all to DBMS (CIFAR + Pexels)")
    print("  3. Full setup (Download + Upload)")
    print("  4. Reset database (delete all media)")
    print("  5. Clear download/upload progress")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        download_all()
    elif choice == "2":
        upload_all()
    elif choice == "3":
        download_all()
        upload_all()
    elif choice == "4":
        reset_database()
    elif choice == "5":
        clear_progress()
        print("✅ Progress cleared!")
    else:
        print("Invalid choice.")
