"""
Feature Routes - API endpoints for feature vector retrieval and comparison
"""
import numpy as np
from flask import Blueprint, jsonify, current_app
from app import db
from app.models import Media, ImageFeatures, AudioFeatures, VideoFeatures
from app.search.distance import euclidean_distance, cosine_distance, manhattan_distance

feature_bp = Blueprint('features', __name__)


def vector_to_list(vector):
    """Convert pgvector to Python list"""
    if vector is None:
        return None
    if hasattr(vector, 'tolist'):
        return vector.tolist()
    if isinstance(vector, (list, tuple)):
        return list(vector)
    # pgvector returns as numpy array or list-like
    try:
        return list(vector)
    except:
        return None


def calculate_similarity(vec1, vec2, metric='cosine'):
    """Calculate similarity between two vectors"""
    if vec1 is None or vec2 is None:
        return 0.0
    
    arr1 = np.array(vec1, dtype=np.float32)
    arr2 = np.array(vec2, dtype=np.float32)
    
    if len(arr1) != len(arr2):
        return 0.0
    
    if metric == 'cosine':
        # Cosine similarity = 1 - cosine_distance
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(arr1, arr2) / (norm1 * norm2))
    elif metric == 'euclidean':
        # Convert euclidean distance to similarity
        dist = np.linalg.norm(arr1 - arr2)
        return float(1 / (1 + dist))
    elif metric == 'manhattan':
        dist = np.sum(np.abs(arr1 - arr2))
        return float(1 / (1 + dist))
    return 0.0


def downsample_histogram(histogram, target_bins=32):
    """Downsample histogram for visualization (reduce 192D to manageable size)"""
    if histogram is None:
        return None
    
    arr = np.array(histogram)
    # Color histogram is 64 bins × 3 channels = 192
    # Reshape and downsample each channel
    if len(arr) == 192:
        # Split into H, S, V channels (64 each)
        h_channel = arr[0:64]
        s_channel = arr[64:128]
        v_channel = arr[128:192]
        
        # Downsample each channel to target_bins/3
        bins_per_channel = target_bins // 3
        h_down = np.array([np.mean(h_channel[i:i+64//bins_per_channel]) 
                          for i in range(0, 64, 64//bins_per_channel)])[:bins_per_channel]
        s_down = np.array([np.mean(s_channel[i:i+64//bins_per_channel]) 
                          for i in range(0, 64, 64//bins_per_channel)])[:bins_per_channel]
        v_down = np.array([np.mean(v_channel[i:i+64//bins_per_channel]) 
                          for i in range(0, 64, 64//bins_per_channel)])[:bins_per_channel]
        
        return {
            'hue': h_down.tolist(),
            'saturation': s_down.tolist(),
            'value': v_down.tolist()
        }
    return arr.tolist()


def downsample_lbp(lbp, target_bins=32):
    """Downsample LBP histogram for visualization"""
    if lbp is None:
        return None
    
    arr = np.array(lbp)
    if len(arr) == 256:
        # Downsample from 256 to target_bins
        step = 256 // target_bins
        downsampled = [float(np.mean(arr[i:i+step])) for i in range(0, 256, step)]
        return downsampled[:target_bins]
    return arr.tolist()


def summarize_deep_features(features, n_groups=16):
    """Summarize deep features into groups for visualization"""
    if features is None:
        return None
    
    arr = np.array(features)
    if len(arr) == 1280:
        # Group 1280 features into n_groups
        group_size = 1280 // n_groups
        groups = []
        for i in range(n_groups):
            start = i * group_size
            end = start + group_size
            # Calculate mean absolute value for each group
            groups.append(float(np.mean(np.abs(arr[start:end]))))
        return groups
    return arr.tolist()


@feature_bp.route('/media/<int:media_id>/features', methods=['GET'])
def get_media_features(media_id):
    """
    Get feature vectors for a specific media item
    Returns features formatted for visualization
    """
    media = Media.query.get(media_id)
    if not media:
        return jsonify({'error': 'Media not found'}), 404
    
    if not media.is_processed:
        return jsonify({'error': 'Media features not yet extracted'}), 400
    
    result = {
        'id': media.id,
        'title': media.title or media.original_filename,
        'media_type': media.media_type,
        'thumbnail_url': f'/api/media/{media.id}/thumbnail'
    }
    
    if media.media_type == 'image' and media.image_features:
        features = media.image_features
        result['features'] = {
            'color_histogram': downsample_histogram(vector_to_list(features.color_histogram)),
            'texture_lbp': downsample_lbp(vector_to_list(features.texture_lbp)),
            'deep_features': summarize_deep_features(vector_to_list(features.deep_features)),
            'raw_dimensions': {
                'color_histogram': 192,
                'texture_lbp': 256,
                'deep_features': 1280,
                'combined': 1728
            }
        }
    elif media.media_type == 'audio' and media.audio_features:
        features = media.audio_features
        result['features'] = {
            'mfcc': vector_to_list(features.mfcc_features),
            'spectral': vector_to_list(features.spectral_features),
            'waveform': vector_to_list(features.waveform_stats),
            'raw_dimensions': {
                'mfcc': 39,
                'spectral': 6,
                'waveform': 5,
                'combined': 50
            }
        }
    elif media.media_type == 'video' and media.video_features:
        features = media.video_features
        result['features'] = {
            'keyframe': summarize_deep_features(vector_to_list(features.keyframe_features)),
            'motion': vector_to_list(features.motion_features),
            'scene_stats': vector_to_list(features.scene_stats),
            'raw_dimensions': {
                'keyframe': 1280,
                'motion': 64,
                'scene_stats': 10,
                'combined': 1354
            }
        }
    else:
        return jsonify({'error': 'No features found for this media'}), 404
    
    return jsonify(result)


@feature_bp.route('/compare/<int:query_id>/<int:result_id>', methods=['GET'])
def compare_features(query_id, result_id):
    """
    Compare features between two media items
    Returns visualization data and per-feature similarity scores
    """
    # Get both media items
    query_media = Media.query.get(query_id)
    result_media = Media.query.get(result_id)
    
    if not query_media:
        return jsonify({'error': 'Query media not found'}), 404
    if not result_media:
        return jsonify({'error': 'Result media not found'}), 404
    
    if query_media.media_type != result_media.media_type:
        return jsonify({'error': 'Cannot compare different media types'}), 400
    
    media_type = query_media.media_type
    
    comparison = {
        'query': {
            'id': query_media.id,
            'title': query_media.title or query_media.original_filename,
            'thumbnail_url': f'/api/media/{query_media.id}/thumbnail'
        },
        'result': {
            'id': result_media.id,
            'title': result_media.title or result_media.original_filename,
            'thumbnail_url': f'/api/media/{result_media.id}/thumbnail'
        },
        'media_type': media_type,
        'similarities': {},
        'features': {}
    }
    
    if media_type == 'image':
        q_feat = query_media.image_features
        r_feat = result_media.image_features
        
        if not q_feat or not r_feat:
            return jsonify({'error': 'Features not found for one or both images'}), 404
        
        # Get raw vectors
        q_color = vector_to_list(q_feat.color_histogram)
        r_color = vector_to_list(r_feat.color_histogram)
        q_lbp = vector_to_list(q_feat.texture_lbp)
        r_lbp = vector_to_list(r_feat.texture_lbp)
        q_deep = vector_to_list(q_feat.deep_features)
        r_deep = vector_to_list(r_feat.deep_features)
        q_combined = vector_to_list(q_feat.combined_features)
        r_combined = vector_to_list(r_feat.combined_features)
        
        # Calculate per-feature similarities
        comparison['similarities'] = {
            'color_histogram': round(calculate_similarity(q_color, r_color, 'cosine') * 100, 1),
            'texture_lbp': round(calculate_similarity(q_lbp, r_lbp, 'cosine') * 100, 1),
            'deep_features': round(calculate_similarity(q_deep, r_deep, 'cosine') * 100, 1),
            'overall': round(calculate_similarity(q_combined, r_combined, 'cosine') * 100, 1)
        }
        
        # Prepare visualization data
        comparison['features'] = {
            'color_histogram': {
                'query': downsample_histogram(q_color),
                'result': downsample_histogram(r_color),
                'labels': {
                    'hue': [f'H{i+1}' for i in range(10)],
                    'saturation': [f'S{i+1}' for i in range(10)],
                    'value': [f'V{i+1}' for i in range(10)]
                }
            },
            'texture_lbp': {
                'query': downsample_lbp(q_lbp),
                'result': downsample_lbp(r_lbp),
                'labels': [f'{i+1}' for i in range(32)]
            },
            'deep_features': {
                'query': summarize_deep_features(q_deep),
                'result': summarize_deep_features(r_deep),
                'labels': [f'G{i+1}' for i in range(16)]
            }
        }
        
    elif media_type == 'audio':
        q_feat = query_media.audio_features
        r_feat = result_media.audio_features
        
        if not q_feat or not r_feat:
            return jsonify({'error': 'Features not found for one or both audio files'}), 404
        
        q_mfcc = vector_to_list(q_feat.mfcc_features)
        r_mfcc = vector_to_list(r_feat.mfcc_features)
        q_spectral = vector_to_list(q_feat.spectral_features)
        r_spectral = vector_to_list(r_feat.spectral_features)
        q_waveform = vector_to_list(q_feat.waveform_stats)
        r_waveform = vector_to_list(r_feat.waveform_stats)
        q_combined = vector_to_list(q_feat.combined_features)
        r_combined = vector_to_list(r_feat.combined_features)
        
        comparison['similarities'] = {
            'mfcc': round(calculate_similarity(q_mfcc, r_mfcc, 'cosine') * 100, 1),
            'spectral': round(calculate_similarity(q_spectral, r_spectral, 'cosine') * 100, 1),
            'waveform': round(calculate_similarity(q_waveform, r_waveform, 'cosine') * 100, 1),
            'overall': round(calculate_similarity(q_combined, r_combined, 'cosine') * 100, 1)
        }
        
        comparison['features'] = {
            'mfcc': {
                'query': q_mfcc,
                'result': r_mfcc,
                'labels': [f'MFCC{i+1}' for i in range(13)] + 
                         [f'Δ{i+1}' for i in range(13)] + 
                         [f'ΔΔ{i+1}' for i in range(13)]
            },
            'spectral': {
                'query': q_spectral,
                'result': r_spectral,
                'labels': ['Centroid', 'Rolloff', 'Bandwidth', 'Contrast', 'Flatness', 'ZCR']
            },
            'waveform': {
                'query': q_waveform,
                'result': r_waveform,
                'labels': ['RMS', 'Peak', 'Crest', 'DynRange', 'Silence']
            }
        }
        
    elif media_type == 'video':
        q_feat = query_media.video_features
        r_feat = result_media.video_features
        
        if not q_feat or not r_feat:
            return jsonify({'error': 'Features not found for one or both videos'}), 404
        
        q_keyframe = vector_to_list(q_feat.keyframe_features)
        r_keyframe = vector_to_list(r_feat.keyframe_features)
        q_motion = vector_to_list(q_feat.motion_features)
        r_motion = vector_to_list(r_feat.motion_features)
        q_scene = vector_to_list(q_feat.scene_stats)
        r_scene = vector_to_list(r_feat.scene_stats)
        q_combined = vector_to_list(q_feat.combined_features)
        r_combined = vector_to_list(r_feat.combined_features)
        
        comparison['similarities'] = {
            'keyframe': round(calculate_similarity(q_keyframe, r_keyframe, 'cosine') * 100, 1),
            'motion': round(calculate_similarity(q_motion, r_motion, 'cosine') * 100, 1),
            'scene_stats': round(calculate_similarity(q_scene, r_scene, 'cosine') * 100, 1),
            'overall': round(calculate_similarity(q_combined, r_combined, 'cosine') * 100, 1)
        }
        
        comparison['features'] = {
            'keyframe': {
                'query': summarize_deep_features(q_keyframe),
                'result': summarize_deep_features(r_keyframe),
                'labels': [f'K{i+1}' for i in range(16)]
            },
            'motion': {
                'query': q_motion,
                'result': r_motion,
                'labels': [f'M{i+1}' for i in range(64)]
            },
            'scene_stats': {
                'query': q_scene,
                'result': r_scene,
                'labels': ['Duration', 'FPS', 'Scenes', 'AvgBright', 'StdBright',
                          'AvgContrast', 'AvgSat', 'DomColor', 'Motion', 'Complexity']
            }
        }
    
    return jsonify(comparison)
