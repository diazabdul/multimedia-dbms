"""
Upload Routes - Handle media file uploads
"""
import os
import uuid
import cv2
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from app import db
from app.models import Media, Thumbnail
from app.extractors import ImageExtractor, AudioExtractor, VideoExtractor

upload_bp = Blueprint('upload', __name__)


def allowed_file(filename, media_type):
    """Check if file extension is allowed for the given media type"""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    
    if media_type == 'image':
        return ext in current_app.config['ALLOWED_IMAGE_EXTENSIONS']
    elif media_type == 'audio':
        return ext in current_app.config['ALLOWED_AUDIO_EXTENSIONS']
    elif media_type == 'video':
        return ext in current_app.config['ALLOWED_VIDEO_EXTENSIONS']
    return False


def detect_media_type(filename):
    """Detect media type based on file extension"""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    
    if ext in current_app.config['ALLOWED_IMAGE_EXTENSIONS']:
        return 'image'
    elif ext in current_app.config['ALLOWED_AUDIO_EXTENSIONS']:
        return 'audio'
    elif ext in current_app.config['ALLOWED_VIDEO_EXTENSIONS']:
        return 'video'
    return None


def get_mime_type(filename):
    """Get MIME type for file"""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or 'application/octet-stream'


@upload_bp.route('/upload', methods=['POST'])
def upload_media():
    """
    Upload a media file (image, audio, or video)
    
    Request: multipart/form-data with 'file' field
    Optional fields: title, description, tags (comma-separated)
    
    Returns: JSON with media ID and details
    """
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Detect media type
    media_type = detect_media_type(file.filename)
    if not media_type:
        return jsonify({'error': 'Unsupported file type'}), 400
    
    # Generate unique filename
    original_filename = secure_filename(file.filename)
    ext = original_filename.rsplit('.', 1)[-1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    
    # Create upload subdirectory based on media type
    upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], media_type)
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, unique_filename)
    
    try:
        # Save file
        file.save(file_path)
        file_size = os.path.getsize(file_path)
        
        # Get optional metadata
        title = request.form.get('title', original_filename)
        description = request.form.get('description', '')
        tags_str = request.form.get('tags', '')
        tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
        
        # Create media record
        media = Media(
            filename=unique_filename,
            original_filename=original_filename,
            file_path=file_path,
            media_type=media_type,
            mime_type=get_mime_type(original_filename),
            file_size=file_size,
            title=title,
            description=description,
            tags=tags if tags else None,
            is_processed=False
        )
        
        db.session.add(media)
        db.session.commit()
        
        # Process features and thumbnails based on media type
        try:
            process_media(media)
        except Exception as e:
            media.processing_error = str(e)
            db.session.commit()
            return jsonify({
                'id': media.id,
                'message': 'File uploaded but feature extraction failed',
                'error': str(e),
                'media': media.to_dict()
            }), 201
        
        return jsonify({
            'id': media.id,
            'message': 'File uploaded and processed successfully',
            'media': media.to_dict()
        }), 201
        
    except Exception as e:
        # Clean up file if database operation fails
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


def process_media(media):
    """Process media file: extract features and generate thumbnails"""
    try:
        if media.media_type == 'image':
            process_image(media)
        elif media.media_type == 'audio':
            process_audio(media)
        elif media.media_type == 'video':
            process_video(media)
        
        media.is_processed = True
        db.session.commit()
        
    except Exception as e:
        media.processing_error = str(e)
        db.session.commit()
        raise


def process_image(media):
    """Process image: extract features and generate thumbnail"""
    extractor = ImageExtractor()
    
    # Extract features
    features = extractor.extract_all_features(media.file_path)
    
    # Get image metadata for dimensions
    try:
        metadata = extractor.get_image_metadata(media.file_path)
        media.width = metadata.get('width')
        media.height = metadata.get('height')
    except Exception:
        pass
    
    # Save features to database
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
    thumb_filename = f"thumb_{media.filename.rsplit('.', 1)[0]}.jpg"
    thumb_output_path = os.path.join(current_app.config['THUMBNAIL_FOLDER'], thumb_filename)
    size = current_app.config['THUMBNAIL_SIZE']
    
    try:
        thumbnail_path = extractor.generate_thumbnail(
            media.file_path,
            thumb_output_path,
            (size, size)
        )
        
        if thumbnail_path:
            thumbnail = Thumbnail(
                media_id=media.id,
                thumbnail_path=thumbnail_path,
                thumbnail_type='default',
                width=size,
                height=size
            )
            db.session.add(thumbnail)
    except Exception as e:
        print(f"Warning: Could not generate thumbnail: {e}")


def process_audio(media):
    """Process audio: extract features and generate waveform thumbnail"""
    extractor = AudioExtractor()
    
    # Extract features
    features = extractor.extract_all_features(media.file_path)
    
    # Get audio metadata for duration
    try:
        metadata = extractor.get_audio_metadata(media.file_path)
        media.duration = metadata.get('duration')
    except Exception:
        pass
    
    # Save features to database
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
    thumb_filename = f"waveform_{media.filename.rsplit('.', 1)[0]}.png"
    thumb_output_path = os.path.join(current_app.config['THUMBNAIL_FOLDER'], thumb_filename)
    size = current_app.config['THUMBNAIL_SIZE']
    
    try:
        thumbnail_path = extractor.generate_waveform_image(
            media.file_path,
            thumb_output_path,
            width=size * 2,
            height=size
        )
        
        if thumbnail_path:
            thumbnail = Thumbnail(
                media_id=media.id,
                thumbnail_path=thumbnail_path,
                thumbnail_type='default',
                width=size * 2,
                height=size
            )
            db.session.add(thumbnail)
    except Exception as e:
        print(f"Warning: Could not generate waveform thumbnail: {e}")


def process_video(media):
    """Process video: extract features and generate keyframe thumbnails"""
    extractor = VideoExtractor()
    
    # Extract features
    features = extractor.extract_all_features(media.file_path)
    
    # Get video metadata
    try:
        metadata = extractor.get_video_metadata(media.file_path)
        media.duration = metadata.get('duration')
        media.width = metadata.get('width')
        media.height = metadata.get('height')
    except Exception:
        pass
    
    # Save features to database
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
    
    # Generate keyframe thumbnails
    thumb_output_dir = current_app.config['THUMBNAIL_FOLDER']
    size = current_app.config['THUMBNAIL_SIZE']
    
    try:
        # Get keyframes
        keyframes, timestamps = extractor.extract_keyframes(media.file_path)
        
        if keyframes:
            os.makedirs(thumb_output_dir, exist_ok=True)
            
            # Use middle frame as default thumbnail (more representative than first frame)
            middle_idx = len(keyframes) // 2
            
            for i, frame in enumerate(keyframes):
                # Resize
                thumbnail = cv2.resize(frame, (size, size))
                
                # Unique filename per media
                thumb_filename = f"video_{media.id}_frame_{i}.jpg"
                thumb_path = os.path.join(thumb_output_dir, thumb_filename)
                cv2.imwrite(thumb_path, thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                # Set middle frame as default, others as frame_N
                if i == middle_idx:
                    thumb_type = 'default'
                else:
                    thumb_type = f'frame_{i}'
                
                thumbnail_record = Thumbnail(
                    media_id=media.id,
                    thumbnail_path=thumb_path,
                    thumbnail_type=thumb_type,
                    width=size,
                    height=size
                )
                db.session.add(thumbnail_record)
    except Exception as e:
        print(f"Warning: Could not generate video thumbnails: {e}")


@upload_bp.route('/upload/batch', methods=['POST'])
def upload_batch():
    """
    Upload multiple media files at once
    
    Request: multipart/form-data with multiple 'files' fields
    
    Returns: JSON with list of uploaded media IDs
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    errors = []
    
    for file in files:
        if file.filename == '':
            continue
            
        media_type = detect_media_type(file.filename)
        if not media_type:
            errors.append({
                'filename': file.filename,
                'error': 'Unsupported file type'
            })
            continue
        
        try:
            # Generate unique filename
            original_filename = secure_filename(file.filename)
            ext = original_filename.rsplit('.', 1)[-1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            
            # Create upload subdirectory
            upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], media_type)
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, unique_filename)
            file.save(file_path)
            file_size = os.path.getsize(file_path)
            
            # Create media record
            media = Media(
                filename=unique_filename,
                original_filename=original_filename,
                file_path=file_path,
                media_type=media_type,
                mime_type=get_mime_type(original_filename),
                file_size=file_size,
                title=original_filename,
                is_processed=False
            )
            
            db.session.add(media)
            db.session.commit()
            
            # Process in background (for batch, we just mark them as pending)
            try:
                process_media(media)
                results.append({
                    'id': media.id,
                    'filename': original_filename,
                    'status': 'processed'
                })
            except Exception as e:
                results.append({
                    'id': media.id,
                    'filename': original_filename,
                    'status': 'pending',
                    'error': str(e)
                })
                
        except Exception as e:
            errors.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return jsonify({
        'message': f'Processed {len(results)} files',
        'results': results,
        'errors': errors
    }), 201 if results else 400
