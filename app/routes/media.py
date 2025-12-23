"""
Media Routes - CRUD operations and media serving
"""
import os
from flask import Blueprint, request, jsonify, current_app, send_file, Response
from app import db
from app.models import Media, Thumbnail

media_bp = Blueprint('media', __name__)


@media_bp.route('/media', methods=['GET'])
def list_media():
    """
    List all media with optional filters
    
    Query params:
        - type: Filter by media type (image, audio, video)
        - page: Page number (default: 1)
        - per_page: Items per page (default: 20, max: 100)
        - sort: Sort field (created_at, title, file_size)
        - order: Sort order (asc, desc)
    
    Returns: JSON with paginated media list
    """
    # Get query parameters
    media_type = request.args.get('type')
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    sort_field = request.args.get('sort', 'created_at')
    sort_order = request.args.get('order', 'desc')
    
    # Build query
    query = Media.query
    
    if media_type:
        query = query.filter(Media.media_type == media_type)
    
    # Apply sorting
    sort_column = getattr(Media, sort_field, Media.created_at)
    if sort_order == 'asc':
        query = query.order_by(sort_column.asc())
    else:
        query = query.order_by(sort_column.desc())
    
    # Paginate
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'page': page,
        'per_page': per_page,
        'total': pagination.total,
        'total_pages': pagination.pages,
        'has_next': pagination.has_next,
        'has_prev': pagination.has_prev,
        'items': [m.to_dict() for m in pagination.items]
    })


@media_bp.route('/media/<int:media_id>', methods=['GET'])
def get_media(media_id):
    """
    Get details of a specific media item
    
    Returns: JSON with full media details including features status
    """
    media = Media.query.get(media_id)
    
    if not media:
        return jsonify({'error': 'Media not found'}), 404
    
    result = media.to_dict()
    
    # Add features status
    result['features'] = {
        'has_image_features': media.image_features is not None,
        'has_audio_features': media.audio_features is not None,
        'has_video_features': media.video_features is not None
    }
    
    # Add thumbnails
    result['thumbnails'] = [
        {
            'id': t.id,
            'type': t.thumbnail_type,
            'url': f'/api/media/{media_id}/thumbnail/{t.id}'
        }
        for t in media.thumbnails
    ]
    
    return jsonify(result)


@media_bp.route('/media/<int:media_id>/file_available', methods=['GET'])
def check_file_available(media_id):
    """
    Check if the media file exists on the server
    
    Returns: JSON with file_exists boolean
    """
    media = Media.query.get(media_id)
    
    if not media:
        return jsonify({'error': 'Media not found'}), 404
    
    # Convert relative path to absolute if needed
    file_path = media.file_path
    if not os.path.isabs(file_path):
        file_path = os.path.join(current_app.root_path, '..', file_path)
        file_path = os.path.abspath(file_path)
    
    # Check if file exists
    file_exists = os.path.exists(file_path)
    
    return jsonify({
        'file_exists': file_exists,
        'file_path': media.filename if file_exists else None,
        'file_size': media.file_size if file_exists else None
    })



@media_bp.route('/media/<int:media_id>', methods=['PUT'])
def update_media(media_id):
    """
    Update media metadata
    
    Request JSON:
        - title: New title
        - description: New description
        - tags: New tags list
    
    Returns: JSON with updated media details
    """
    media = Media.query.get(media_id)
    
    if not media:
        return jsonify({'error': 'Media not found'}), 404
    
    data = request.get_json() or {}
    
    if 'title' in data:
        media.title = data['title']
    
    if 'description' in data:
        media.description = data['description']
    
    if 'tags' in data:
        media.tags = data['tags'] if isinstance(data['tags'], list) else None
    
    try:
        db.session.commit()
        return jsonify({
            'message': 'Media updated successfully',
            'media': media.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Update failed: {str(e)}'}), 500


@media_bp.route('/media/<int:media_id>', methods=['DELETE'])
def delete_media(media_id):
    """
    Delete a media item and its associated files
    
    Returns: JSON confirmation
    """
    media = Media.query.get(media_id)
    
    if not media:
        return jsonify({'error': 'Media not found'}), 404
    
    try:
        # Delete physical files
        if os.path.exists(media.file_path):
            os.remove(media.file_path)
        
        # Delete thumbnails
        for thumbnail in media.thumbnails:
            if os.path.exists(thumbnail.thumbnail_path):
                os.remove(thumbnail.thumbnail_path)
        
        # Delete from database (cascades to features and thumbnails)
        db.session.delete(media)
        db.session.commit()
        
        return jsonify({
            'message': 'Media deleted successfully',
            'id': media_id
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Delete failed: {str(e)}'}), 500


@media_bp.route('/media/<int:media_id>/file', methods=['GET'])
def serve_media_file(media_id):
    """
    Serve the original media file
    
    Returns: Media file with appropriate content type
    """
    media = Media.query.get(media_id)
    
    if not media:
        return jsonify({'error': 'Media not found'}), 404
    
    # Convert relative path to absolute if needed
    file_path = media.file_path
    if not os.path.isabs(file_path):
        file_path = os.path.join(current_app.root_path, '..', file_path)
        file_path = os.path.abspath(file_path)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(
        file_path,
        mimetype=media.mime_type,
        as_attachment=False,
        download_name=media.original_filename
    )


@media_bp.route('/media/<int:media_id>/download', methods=['GET'])
def download_media_file(media_id):
    """
    Download the original media file
    
    Returns: Media file as attachment
    """
    media = Media.query.get(media_id)
    
    if not media:
        return jsonify({'error': 'Media not found'}), 404
    
    # Convert relative path to absolute if needed
    file_path = media.file_path
    if not os.path.isabs(file_path):
        file_path = os.path.join(current_app.root_path, '..', file_path)
        file_path = os.path.abspath(file_path)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(
        file_path,
        mimetype=media.mime_type,
        as_attachment=True,
        download_name=media.original_filename
    )


@media_bp.route('/media/<int:media_id>/thumbnail', methods=['GET'])
def serve_thumbnail(media_id):
    """
    Serve the default thumbnail for a media item
    
    Returns: Thumbnail image
    """
    try:
        media = Media.query.get(media_id)
        
        if not media:
            return jsonify({'error': 'Media not found'}), 404
        
        # Get default thumbnail
        thumbnail = Thumbnail.query.filter_by(
            media_id=media_id,
            thumbnail_type='default'
        ).first()
        
        # Fallback to first available thumbnail
        if not thumbnail and media.thumbnails:
            thumbnail = media.thumbnails[0]
        
        if not thumbnail:
            return jsonify({'error': 'No thumbnail record found'}), 404
        
        # Convert relative path to absolute if needed
        thumb_path = thumbnail.thumbnail_path
        if not os.path.isabs(thumb_path):
            # Path is relative to app root (parent of 'app' folder)
            thumb_path = os.path.join(current_app.root_path, '..', thumb_path)
            thumb_path = os.path.abspath(thumb_path)
            
        if not os.path.exists(thumb_path):
            return jsonify({'error': f'Thumbnail file not found: {thumb_path}'}), 404
        
        return send_file(
            thumb_path,
            mimetype='image/jpeg'
        )
    except Exception as e:
        import traceback
        return jsonify({'error': f'Thumbnail error: {str(e)}', 'traceback': traceback.format_exc()}), 500


@media_bp.route('/media/<int:media_id>/thumbnail/<int:thumbnail_id>', methods=['GET'])
def serve_specific_thumbnail(media_id, thumbnail_id):
    """
    Serve a specific thumbnail by ID
    
    Returns: Thumbnail image
    """
    thumbnail = Thumbnail.query.filter_by(
        id=thumbnail_id,
        media_id=media_id
    ).first()
    
    if not thumbnail or not os.path.exists(thumbnail.thumbnail_path):
        return jsonify({'error': 'Thumbnail not found'}), 404
    
    return send_file(
        thumbnail.thumbnail_path,
        mimetype='image/jpeg'
    )


@media_bp.route('/media/<int:media_id>/stream', methods=['GET'])
def stream_media(media_id):
    """
    Stream audio/video media with range request support
    
    Returns: Streaming response with range support
    """
    media = Media.query.get(media_id)
    
    if not media:
        return jsonify({'error': 'Media not found'}), 404
    
    if media.media_type not in ['audio', 'video']:
        return jsonify({'error': 'Streaming only supported for audio/video'}), 400
    
    # Convert relative path to absolute if needed
    file_path = media.file_path
    if not os.path.isabs(file_path):
        file_path = os.path.join(current_app.root_path, '..', file_path)
        file_path = os.path.abspath(file_path)
    
    if not os.path.exists(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 404
    
    file_size = os.path.getsize(file_path)
    
    # Handle range requests for seeking
    range_header = request.headers.get('Range', None)
    
    if range_header:
        # Parse range header
        byte_start = 0
        byte_end = file_size - 1
        
        match = range_header.replace('bytes=', '').split('-')
        if match[0]:
            byte_start = int(match[0])
        if match[1]:
            byte_end = int(match[1])
        
        chunk_size = byte_end - byte_start + 1
        
        def generate():
            with open(file_path, 'rb') as f:
                f.seek(byte_start)
                remaining = chunk_size
                while remaining > 0:
                    chunk = f.read(min(8192, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        
        response = Response(
            generate(),
            status=206,
            mimetype=media.mime_type,
            direct_passthrough=True
        )
        response.headers['Content-Range'] = f'bytes {byte_start}-{byte_end}/{file_size}'
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Length'] = chunk_size
        
        return response
    
    else:
        # Full file request
        return send_file(
            file_path,
            mimetype=media.mime_type
        )


@media_bp.route('/media/batch', methods=['DELETE'])
def batch_delete():
    """
    Delete multiple media items at once
    
    Request JSON:
        - ids: List of media IDs to delete
    
    Returns: JSON with delete results
    """
    data = request.get_json() or {}
    ids = data.get('ids', [])
    
    if not ids:
        return jsonify({'error': 'No media IDs provided'}), 400
    
    deleted = []
    errors = []
    
    for media_id in ids:
        media = Media.query.get(media_id)
        
        if not media:
            errors.append({'id': media_id, 'error': 'Not found'})
            continue
        
        try:
            # Delete physical files
            if os.path.exists(media.file_path):
                os.remove(media.file_path)
            
            for thumbnail in media.thumbnails:
                if os.path.exists(thumbnail.thumbnail_path):
                    os.remove(thumbnail.thumbnail_path)
            
            db.session.delete(media)
            deleted.append(media_id)
            
        except Exception as e:
            errors.append({'id': media_id, 'error': str(e)})
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Batch delete failed: {str(e)}'}), 500
    
    return jsonify({
        'message': f'Deleted {len(deleted)} media items',
        'deleted': deleted,
        'errors': errors
    })
