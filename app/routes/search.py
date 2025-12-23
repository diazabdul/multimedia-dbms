"""
Search Routes - Query By Example, Metadata, and Hybrid Search
"""
import os
import uuid
import tempfile
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from app import db
from app.models import Media
from app.extractors import ImageExtractor, AudioExtractor, VideoExtractor
from app.search.knn import KNNSearch
from app.search.distance import DistanceMetric

search_bp = Blueprint('search', __name__)


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


def extract_query_features(file_path, media_type):
    """Extract features from query file - optimized for fast QBE search"""
    if media_type == 'image':
        extractor = ImageExtractor()
        features = extractor.extract_all_features(file_path)
        return features['combined_features']
    elif media_type == 'audio':
        extractor = AudioExtractor()
        features = extractor.extract_all_features(file_path)
        return features['combined_features']
    elif media_type == 'video':
        # For video QBE, use ONLY scene stats to avoid timeout
        # Motion features (optical flow) take 120+ seconds
        # Deep features (MobileNetV2 on CPU) also too slow
        extractor = VideoExtractor(use_deep_features=False, n_keyframes=1)
        
        # Extract only the fast scene statistics
        scene_stats = extractor.extract_scene_stats(file_path)
        
        # CRITICAL: Match the stored feature dimensions (1354 for videos)
        # Stored videos have: keyframe_features + motion_features + scene_stats
        # We only use scene stats (10 dims) for speed, so pad to 1354
        import numpy as np
        features = np.zeros(1354, dtype=np.float32)
        features[:len(scene_stats)] = scene_stats  # Fill first 10 positions
        return features
    return None


@search_bp.route('/search/qbe', methods=['POST'])
def query_by_example():
    """
    Query By Example - Find similar media based on uploaded file
    
    Request: multipart/form-data with:
        - file: Query media file
        - k: Number of results (default: 10)
        - metric: Distance metric (euclidean, manhattan, cosine)
    
    Returns: JSON with ranked results and similarity scores
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No query file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Detect media type
    media_type = detect_media_type(file.filename)
    if not media_type:
        return jsonify({'error': 'Unsupported file type'}), 400
    
    # Get search parameters
    k = request.form.get('k', 10, type=int)
    metric = request.form.get('metric', 'euclidean')
    
    # Create temp file for query
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, secure_filename(file.filename))
    
    try:
        # Save query file temporarily
        file.save(temp_path)
        
        # Extract features from query
        query_features = extract_query_features(temp_path, media_type)
        
        if query_features is None:
            return jsonify({'error': 'Failed to extract features from query file'}), 500
        
        # Perform similarity search
        results = KNNSearch.search_by_type(query_features, media_type, k, metric)
        
        return jsonify({
            'query_type': 'qbe',
            'media_type': media_type,
            'metric': metric,
            'k': k,
            'results_count': len(results),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


@search_bp.route('/search/qbe/<int:media_id>', methods=['GET'])
def query_by_existing_media(media_id):
    """
    Query By Example using existing media in database
    
    URL params:
        - k: Number of results (default: 10)
        - metric: Distance metric (euclidean, manhattan, cosine)
    
    Returns: JSON with ranked results (includes the query media itself)
    """
    # Get the source media
    media = Media.query.get(media_id)
    if not media:
        return jsonify({'error': 'Media not found'}), 404
    
    if not media.is_processed:
        return jsonify({'error': 'Media features not yet extracted'}), 400
    
    # Get search parameters
    k = request.args.get('k', 10, type=int)
    metric = request.args.get('metric', 'euclidean')
    
    try:
        # Get features from the existing media
        if media.media_type == 'image' and media.image_features:
            query_features = media.image_features.combined_features
        elif media.media_type == 'audio' and media.audio_features:
            query_features = media.audio_features.combined_features
        elif media.media_type == 'video' and media.video_features:
            query_features = media.video_features.combined_features
        else:
            return jsonify({'error': 'No features found for this media'}), 400
        
        # Perform similarity search (include the query file in results)
        results = KNNSearch.search_by_type(query_features, media.media_type, k, metric)
        
        return jsonify({
            'query_type': 'qbe',
            'query_media_id': media_id,
            'media_type': media.media_type,
            'metric': metric,
            'k': k,
            'results_count': len(results),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500


@search_bp.route('/search/metadata', methods=['POST'])
def query_by_metadata():
    """
    Query By Metadata - Find media based on metadata filters
    
    Request JSON:
        - title: Title search (partial match)
        - tags: List of tags to match
        - media_type: Filter by type (image, audio, video)
        - date_from: Filter by creation date (ISO format)
        - date_to: Filter by creation date (ISO format)
        - limit: Number of results (default: 20)
    
    Returns: JSON with matching media
    """
    try:
        data = request.get_json() or {}
        
        limit = data.get('limit', 20)
        
        # Build query
        query = Media.query.filter(Media.is_processed == True)
        
        # Filter by media type
        if 'media_type' in data and data['media_type']:
            query = query.filter(Media.media_type == data['media_type'])
        
        # Filter by title (partial match)
        if 'title' in data and data['title']:
            query = query.filter(Media.title.ilike(f"%{data['title']}%"))
        
        # Filter by description
        if 'description' in data and data['description']:
            query = query.filter(Media.description.ilike(f"%{data['description']}%"))
        
        # Filter by tags (array overlap) - check if any of the search tags match any media tags
        if 'tags' in data and data['tags']:
            tags = data['tags'] if isinstance(data['tags'], list) else [data['tags']]
            # Use ANY to check if any of the user's tags are in the media's tags array
            from sqlalchemy import any_
            tag_filters = [Media.tags.any(tag) for tag in tags]
            if tag_filters:
                # OR condition - match if ANY of the tags match
                from sqlalchemy import or_
                query = query.filter(or_(*tag_filters))
        
        # Filter by date range
        if 'date_from' in data and data['date_from']:
            from datetime import datetime
            date_from = datetime.fromisoformat(data['date_from'].replace('Z', '+00:00'))
            query = query.filter(Media.created_at >= date_from)
        
        if 'date_to' in data and data['date_to']:
            from datetime import datetime
            date_to = datetime.fromisoformat(data['date_to'].replace('Z', '+00:00'))
            query = query.filter(Media.created_at <= date_to)
        
        # Order by creation date
        query = query.order_by(Media.created_at.desc())
        
        # Execute query
        results = query.limit(limit).all()
        
        return jsonify({
            'query_type': 'metadata',
            'filters': {k: v for k, v in data.items() if k != 'limit'},
            'results_count': len(results),
            'results': [m.to_dict() for m in results]
        })
    
    except Exception as e:
        current_app.logger.error(f"Metadata search error: {str(e)}")
        return jsonify({'error': f'Search failed: {str(e)}'}), 500


@search_bp.route('/search/hybrid', methods=['POST'])
def hybrid_search():
    """
    Hybrid Search - Combine QBE and metadata filters
    
    Request: multipart/form-data with:
        - file: Query media file (optional if using media_id)
        - media_id: ID of existing media to use as query (optional)
        - title: Title filter for metadata matching
        - tags: Comma-separated tags for metadata matching
        - k: Number of results (default: 10)
        - metric: Distance metric (default: euclidean)
        - weight_feature: Weight for feature similarity (default: 0.7)
        - weight_metadata: Weight for metadata score (default: 0.3)
    
    Returns: JSON with ranked results combining feature similarity and metadata
    """
    # Get query features
    query_features = None
    media_type = None
    
    # Option 1: Use uploaded file
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        media_type = detect_media_type(file.filename)
        
        if not media_type:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, secure_filename(file.filename))
        
        try:
            file.save(temp_path)
            query_features = extract_query_features(temp_path, media_type)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
    
    # Option 2: Use existing media ID
    elif 'media_id' in request.form:
        media_id = int(request.form['media_id'])
        media = Media.query.get(media_id)
        
        if not media:
            return jsonify({'error': 'Media not found'}), 404
        
        if not media.is_processed:
            return jsonify({'error': 'Media features not yet extracted'}), 400
        
        media_type = media.media_type
        
        if media.media_type == 'image' and media.image_features:
            query_features = media.image_features.combined_features
        elif media.media_type == 'audio' and media.audio_features:
            query_features = media.audio_features.combined_features
        elif media.media_type == 'video' and media.video_features:
            query_features = media.video_features.combined_features
    
    if query_features is None:
        return jsonify({'error': 'No query provided. Use file or media_id.'}), 400
    
    # Get parameters
    k = request.form.get('k', 10, type=int)
    metric = request.form.get('metric', 'euclidean')
    weight_feature = request.form.get('weight_feature', 0.7, type=float)
    weight_metadata = request.form.get('weight_metadata', 0.3, type=float)
    
    # Get metadata filters
    metadata_filters = {}
    
    if 'title' in request.form and request.form['title']:
        metadata_filters['title'] = request.form['title']
    
    if 'tags' in request.form and request.form['tags']:
        tags_str = request.form['tags']
        metadata_filters['tags'] = [t.strip() for t in tags_str.split(',') if t.strip()]
    
    try:
        # Perform hybrid search
        results = KNNSearch.hybrid_search(
            query_features=query_features,
            media_type=media_type,
            metadata_filters=metadata_filters,
            k=k,
            metric=metric,
            weight_feature=weight_feature,
            weight_metadata=weight_metadata
        )
        
        return jsonify({
            'query_type': 'hybrid',
            'media_type': media_type,
            'metric': metric,
            'k': k,
            'weights': {
                'feature': weight_feature,
                'metadata': weight_metadata
            },
            'metadata_filters': metadata_filters,
            'results_count': len(results),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Hybrid search failed: {str(e)}'}), 500


@search_bp.route('/search/stats', methods=['GET'])
def search_stats():
    """
    Get statistics about the media database
    
    Returns: JSON with counts and metadata about stored media
    """
    from sqlalchemy import func
    
    # Count by media type
    type_counts = db.session.query(
        Media.media_type,
        func.count(Media.id)
    ).filter(
        Media.is_processed == True
    ).group_by(
        Media.media_type
    ).all()
    
    type_stats = {t: c for t, c in type_counts}
    
    # Total count
    total_count = sum(type_stats.values())
    
    # Get all tags
    all_tags = db.session.query(
        func.unnest(Media.tags).label('tag')
    ).filter(
        Media.tags.isnot(None)
    ).distinct().all()
    
    tags = [t[0] for t in all_tags if t[0]]
    
    return jsonify({
        'total_media': total_count,
        'by_type': type_stats,
        'available_tags': sorted(tags)
    })
