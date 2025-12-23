"""
Video Dataset Download Script
Downloads videos from Pexels with similar variations for testing similarity search.
"""

import os
import sys
import time
import requests
import subprocess
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================

# Pexels API Key - Get free key at https://www.pexels.com/api/
# Replace with your own API key
PEXELS_API_KEY = "wMAiZdBf2crypaR4X3wn2KGifCGWR8AynQtRwgf7vLJhfoOgpMnFT1Hs"

# Download directory
DOWNLOAD_DIR = Path(__file__).parent / "dataset_videos"

# Categories with search terms (each will download multiple similar videos)
CATEGORIES = {
    "ocean": ["ocean waves", "beach waves", "sea water", "ocean shore"],
    "forest": ["forest walk", "green forest", "trees nature", "forest path"],
    "city": ["city traffic", "urban street", "city skyline", "downtown"],
    "animals": ["dog playing", "cat video", "birds flying", "wildlife"],
    "food": ["cooking food", "restaurant kitchen", "food preparation"],
    "sports": ["basketball game", "running track", "soccer football", "fitness workout"],
    "sky": ["clouds sky", "sunset timelapse", "sunrise sky", "blue sky"],
    "water": ["waterfall nature", "river stream", "lake peaceful", "rain drops"],
}

# Number of videos per search term
VIDEOS_PER_SEARCH = 2

# Video quality preference - use SD for smaller file sizes
VIDEO_QUALITY = "sd"

# Maximum file size in MB (skip larger files)
MAX_FILE_SIZE_MB = 50


# ============================================
# PEXELS API FUNCTIONS
# ============================================

def search_pexels_videos(query, per_page=5):
    """Search for videos on Pexels"""
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": query,
        "per_page": per_page,
        "orientation": "landscape"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json().get("videos", [])
    except Exception as e:
        print(f"  ❌ Error searching '{query}': {e}")
        return []


def get_best_video_file(video_data):
    """Get a small/medium quality video file URL (SD preferred for smaller sizes)"""
    video_files = video_data.get("video_files", [])
    
    if not video_files:
        return None, None, 0
    
    # Sort by height (prefer smaller resolution for smaller file size)
    video_files.sort(key=lambda x: x.get("height", 0))
    
    # Find SD quality videos (360p - 540p range)
    for vf in video_files:
        height = vf.get("height", 0)
        quality = vf.get("quality", "")
        file_size = vf.get("size", 0)  # Size in bytes if available
        
        # Prefer SD quality, 360p-540p for small files
        if quality == "sd" and 360 <= height <= 540:
            return vf.get("link"), vf.get("file_type", "video/mp4"), file_size
    
    # Fallback: any SD quality
    for vf in video_files:
        if vf.get("quality") == "sd":
            return vf.get("link"), vf.get("file_type", "video/mp4"), vf.get("size", 0)
    
    # Last resort: smallest available
    smallest = video_files[0]
    return smallest.get("link"), smallest.get("file_type", "video/mp4"), smallest.get("size", 0)


def download_video(url, save_path):
    """Download a video file"""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"  ❌ Download error: {e}")
        return False


# ============================================
# VIDEO VARIATION FUNCTIONS (using ffmpeg)
# ============================================

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def create_variation_bright(input_path, output_path):
    """Create a brighter version of the video"""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", "eq=brightness=0.1:saturation=1.2",
        "-c:a", "copy",
        str(output_path)
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except:
        return False


def create_variation_contrast(input_path, output_path):
    """Create a high contrast version"""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", "eq=contrast=1.3",
        "-c:a", "copy",
        str(output_path)
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except:
        return False


def create_variation_speed(input_path, output_path, speed=1.5):
    """Create a speed variation"""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"setpts={1/speed}*PTS",
        "-an",  # Remove audio for speed changes
        str(output_path)
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except:
        return False


def create_variations(original_path, category_dir, base_name):
    """Create multiple variations of a video"""
    variations_created = []
    
    if not check_ffmpeg():
        print("  ⚠️ ffmpeg not found, skipping variations")
        return variations_created
    
    # Variation 1: Bright version
    bright_path = category_dir / f"{base_name}_bright.mp4"
    if create_variation_bright(original_path, bright_path):
        variations_created.append(bright_path)
        print(f"    ✓ Created bright variation")
    
    # Variation 2: High contrast
    contrast_path = category_dir / f"{base_name}_contrast.mp4"
    if create_variation_contrast(original_path, contrast_path):
        variations_created.append(contrast_path)
        print(f"    ✓ Created contrast variation")
    
    return variations_created


# ============================================
# MAIN DOWNLOAD FUNCTION
# ============================================

def download_dataset():
    """Download the complete video dataset"""
    
    print("=" * 60)
    print("📹 Video Dataset Downloader for Multimedia DBMS")
    print("=" * 60)
    
    # Check API key
    if PEXELS_API_KEY == "YOUR_PEXELS_API_KEY_HERE":
        print("\n❌ ERROR: Please set your Pexels API key!")
        print("   1. Go to https://www.pexels.com/api/")
        print("   2. Create free account and get API key")
        print("   3. Replace 'YOUR_PEXELS_API_KEY_HERE' in this script")
        return
    
    # Create download directory
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Download directory: {DOWNLOAD_DIR}")
    
    # Check ffmpeg
    has_ffmpeg = check_ffmpeg()
    if has_ffmpeg:
        print("✓ ffmpeg found - will create video variations")
    else:
        print("⚠️ ffmpeg not found - variations disabled")
        print("   Install ffmpeg for additional video variations")
    
    total_downloaded = 0
    total_variations = 0
    
    # Process each category
    for category, search_terms in CATEGORIES.items():
        print(f"\n{'='*50}")
        print(f"📂 Category: {category.upper()}")
        print(f"{'='*50}")
        
        category_dir = DOWNLOAD_DIR / category
        category_dir.mkdir(exist_ok=True)
        
        for search_term in search_terms:
            print(f"\n  🔍 Searching: '{search_term}'")
            
            videos = search_pexels_videos(search_term, VIDEOS_PER_SEARCH)
            
            if not videos:
                print(f"    No videos found")
                continue
            
            for i, video in enumerate(videos[:VIDEOS_PER_SEARCH]):
                video_id = video.get("id")
                video_url, file_type, expected_size = get_best_video_file(video)
                
                if not video_url:
                    continue
                
                # Create filename
                safe_term = search_term.replace(" ", "_").lower()
                filename = f"{safe_term}_{i+1}.mp4"
                save_path = category_dir / filename
                
                # Skip if already exists
                if save_path.exists():
                    print(f"    ⏭️ Skipping (exists): {filename}")
                    continue
                
                print(f"    ⬇️ Downloading: {filename}")
                
                if download_video(video_url, save_path):
                    # Check file size after download
                    actual_size_mb = save_path.stat().st_size / (1024 * 1024)
                    
                    if actual_size_mb > MAX_FILE_SIZE_MB:
                        print(f"    ⚠️ File too large ({actual_size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB), removing...")
                        save_path.unlink()
                        continue
                    
                    total_downloaded += 1
                    print(f"    ✅ Downloaded: {filename} ({actual_size_mb:.1f}MB)")
                    
                    # Create variations (only if not too large)
                    if has_ffmpeg and actual_size_mb < 30:  # Only create variations for smaller files
                        base_name = safe_term + f"_{i+1}"
                        variations = create_variations(save_path, category_dir, base_name)
                        total_variations += len(variations)
                
                # Be nice to API
                time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  ✅ Videos downloaded: {total_downloaded}")
    print(f"  🔄 Variations created: {total_variations}")
    print(f"  📁 Total files: {total_downloaded + total_variations}")
    print(f"  📂 Location: {DOWNLOAD_DIR}")
    print("\n🎉 Dataset ready for upload to Multimedia DBMS!")


# ============================================
# UPLOAD TO DBMS FUNCTION
# ============================================

PROGRESS_FILE = Path(__file__).parent / "upload_progress.json"


def load_progress():
    """Load upload progress from file"""
    if PROGRESS_FILE.exists():
        try:
            import json
            with open(PROGRESS_FILE, 'r') as f:
                return set(json.load(f))
        except:
            return set()
    return set()


def save_progress(uploaded_files):
    """Save upload progress to file"""
    import json
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(list(uploaded_files), f)


def clear_progress():
    """Clear upload progress file"""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("✅ Progress cleared")


def upload_to_dbms(api_base="http://localhost:5000/api"):
    """Upload downloaded videos to the Multimedia DBMS with resume support"""
    
    print("\n" + "=" * 60)
    print("📤 Uploading Videos to Multimedia DBMS")
    print("=" * 60)
    print("⚠️ Note: Video processing may take several minutes per file")
    print("💾 Progress is saved - you can resume if interrupted!")
    
    if not DOWNLOAD_DIR.exists():
        print("❌ No dataset found. Run download first.")
        return
    
    # Load previous progress
    uploaded_files = load_progress()
    if uploaded_files:
        print(f"📂 Found {len(uploaded_files)} previously uploaded files")
    
    # Collect all video files
    all_files = []
    for category_dir in DOWNLOAD_DIR.iterdir():
        if category_dir.is_dir():
            for video_file in category_dir.glob("*.mp4"):
                all_files.append((category_dir.name, video_file))
    
    total_files = len(all_files)
    remaining_files = [(cat, f) for cat, f in all_files if str(f) not in uploaded_files]
    
    print(f"📁 Total: {total_files} files, Remaining: {len(remaining_files)} files\n")
    
    if not remaining_files:
        print("✅ All files already uploaded!")
        return
    
    uploaded = 0
    failed = 0
    skipped = len(uploaded_files)
    current = skipped
    
    try:
        for category, video_file in remaining_files:
            current += 1
            
            # Prepare metadata
            title = video_file.stem.replace("_", " ").title()
            tags = [category]
            
            # Add variation tag if applicable
            if "bright" in video_file.stem:
                tags.append("bright")
            elif "contrast" in video_file.stem:
                tags.append("high-contrast")
            
            print(f"\n  [{current}/{total_files}] ⬆️ {video_file.name}")
            print(f"      Category: {category}, Tags: {tags}")
            
            # Retry logic
            max_retries = 2
            success = False
            
            for attempt in range(max_retries + 1):
                try:
                    with open(video_file, 'rb') as f:
                        files = {'file': (video_file.name, f, 'video/mp4')}
                        data = {
                            'title': title,
                            'description': f'Dataset video - {category}',
                            'tags': ','.join(tags)
                        }
                        
                        # Long timeout for video processing (10 minutes)
                        response = requests.post(
                            f"{api_base}/upload",
                            files=files,
                            data=data,
                            timeout=600  # 10 minutes
                        )
                        
                        if response.status_code in [200, 201]:
                            uploaded += 1
                            success = True
                            uploaded_files.add(str(video_file))
                            save_progress(uploaded_files)  # Save progress after each success
                            print(f"    ✅ Uploaded successfully")
                            break
                        else:
                            try:
                                error_msg = response.json().get('error', f'Status {response.status_code}')
                            except:
                                error_msg = f'Status {response.status_code}'
                            
                            if "already exists" in error_msg.lower():
                                skipped += 1
                                success = True
                                uploaded_files.add(str(video_file))
                                save_progress(uploaded_files)
                                print(f"    ⏭️ Skipped (already exists)")
                                break
                            else:
                                print(f"    ⚠️ Attempt {attempt+1}: {error_msg}")
                
                except requests.exceptions.Timeout:
                    print(f"    ⚠️ Attempt {attempt+1}: Timeout (processing took too long)")
                    if attempt < max_retries:
                        print(f"    🔄 Retrying in 5 seconds...")
                        time.sleep(5)
                
                except requests.exceptions.ConnectionError:
                    print(f"    ⚠️ Attempt {attempt+1}: Connection error (server may be down)")
                    if attempt < max_retries:
                        print(f"    🔄 Retrying in 10 seconds...")
                        time.sleep(10)
                
                except Exception as e:
                    print(f"    ⚠️ Attempt {attempt+1}: {e}")
                    if attempt < max_retries:
                        print(f"    🔄 Retrying in 5 seconds...")
                        time.sleep(5)
            
            if not success:
                failed += 1
                print(f"    ❌ Failed after {max_retries + 1} attempts")
            
            # Wait between uploads to avoid overwhelming server
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Upload interrupted by user!")
        print(f"💾 Progress saved - {len(uploaded_files)} files recorded")
        print("   Run the script again to resume from where you left off.")
    
    finally:
        print("\n" + "=" * 60)
        print("📊 UPLOAD SUMMARY")
        print("=" * 60)
        print(f"  ✅ Successfully uploaded: {uploaded}")
        print(f"  ⏭️ Skipped (already done): {skipped}")
        print(f"  ❌ Failed: {failed}")
        print(f"  📁 Total progress: {len(uploaded_files)}/{total_files}")
        
        if len(uploaded_files) == total_files:
            print("\n🎉 All files uploaded! Clearing progress file...")
            clear_progress()


# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    print("\n📹 Video Dataset Tool")
    print("Choose an option:")
    print("  1. Download videos from Pexels")
    print("  2. Upload videos to DBMS")
    print("  3. Both (download + upload)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        download_dataset()
    elif choice == "2":
        upload_to_dbms()
    elif choice == "3":
        download_dataset()
        upload_to_dbms()
    else:
        print("Invalid choice. Running download only.")
        download_dataset()
