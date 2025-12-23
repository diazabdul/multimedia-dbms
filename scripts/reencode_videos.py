"""
Re-encode videos to H.264 for browser compatibility
Converts mp4v encoded videos to H.264 which all browsers support
"""

import os
import subprocess
import shutil
from pathlib import Path

# Full path to ffmpeg
FFMPEG_PATH = r"C:\Dependency\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"


def reencode_video(input_path, output_path):
    """Re-encode a video to H.264 using ffmpeg"""
    cmd = [
        FFMPEG_PATH,
        '-i', input_path,
        '-c:v', 'libx264',      # H.264 video codec
        '-preset', 'fast',       # Encoding speed
        '-crf', '23',            # Quality (lower = better, 23 is default)
        '-c:a', 'aac',           # AAC audio codec  
        '-movflags', '+faststart',  # Allow streaming
        '-y',                    # Overwrite output
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True
        else:
            print(f"  Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print("  Timeout - skipping")
        return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def main():
    video_folder = Path("uploads/video")
    
    if not video_folder.exists():
        print("Video folder not found!")
        return
    
    videos = list(video_folder.glob("*.mp4"))
    print(f"Found {len(videos)} videos to re-encode")
    
    temp_folder = Path("uploads/video_temp")
    temp_folder.mkdir(exist_ok=True)
    
    success_count = 0
    
    for i, video_path in enumerate(videos):
        print(f"[{i+1}/{len(videos)}] Re-encoding: {video_path.name}")
        
        temp_output = temp_folder / video_path.name
        
        if reencode_video(str(video_path), str(temp_output)):
            # Replace original with re-encoded version
            os.replace(str(temp_output), str(video_path))
            success_count += 1
            print(f"  ✓ Done")
        else:
            print(f"  ✗ Failed")
    
    # Cleanup temp folder
    if temp_folder.exists():
        shutil.rmtree(temp_folder)
    
    print(f"\n{'='*50}")
    print(f"Re-encoding complete: {success_count}/{len(videos)} videos converted")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
