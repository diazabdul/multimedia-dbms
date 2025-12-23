"""
KNN Search Implementation
Uses pgvector for efficient similarity search in PostgreSQL
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from sqlalchemy import text
from app import db
from app.models import Media, ImageFeatures, AudioFeatures, VideoFeatures
from app.search.distance import DistanceMetric, distance_to_similarity


class KNNSearch:
    """K-Nearest Neighbors search using pgvector"""
    
    # pgvector distance operators
    DISTANCE_OPERATORS = {
        DistanceMetric.EUCLIDEAN: '<->',   # L2 distance
        DistanceMetric.MANHATTAN: '<+>',   # L1 distance (requires pgvector 0.5+)
        DistanceMetric.COSINE: '<=>',      # Cosine distance
    }
    
    @staticmethod
    def search_images(query_features: np.ndarray, k: int = 10, 
                      metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
                      use_combined: bool = True) -> List[Dict]:
        """
        Search for similar images using feature vectors
        
        Args:
            query_features: Query feature vector
            k: Number of results to return
            metric: Distance metric to use
            use_combined: Whether to use combined features or individual
            
        Returns:
            List of search results with similarity scores
        """
        if isinstance(metric, str):
            metric = DistanceMetric(metric.lower())
        
        operator = KNNSearch.DISTANCE_OPERATORS.get(metric, '<->')
        feature_column = 'combined_features' if use_combined else 'deep_features'
        
        # Convert numpy array to PostgreSQL vector string
        vector_str = '[' + ','.join(map(str, query_features.tolist())) + ']'
        
        query = text(f"""
            SELECT 
                m.id, m.filename, m.original_filename, m.media_type,
                m.title, m.description, m.tags, m.width, m.height,
                m.created_at, m.is_processed,
                if.{feature_column} {operator} :query_vector AS distance
            FROM media m
            JOIN image_features if ON m.id = if.media_id
            WHERE m.is_processed = true
            ORDER BY if.{feature_column} {operator} :query_vector
            LIMIT :k
        """)
        
        result = db.session.execute(query, {'query_vector': vector_str, 'k': k})
        
        return KNNSearch._format_results(result, metric)
    
    @staticmethod
    def search_audio(query_features: np.ndarray, k: int = 10,
                    metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN) -> List[Dict]:
        """
        Search for similar audio files using feature vectors
        
        Args:
            query_features: Query feature vector (50 dimensions)
            k: Number of results to return
            metric: Distance metric to use
            
        Returns:
            List of search results with similarity scores
        """
        if isinstance(metric, str):
            metric = DistanceMetric(metric.lower())
        
        operator = KNNSearch.DISTANCE_OPERATORS.get(metric, '<->')
        
        vector_str = '[' + ','.join(map(str, query_features.tolist())) + ']'
        
        query = text(f"""
            SELECT 
                m.id, m.filename, m.original_filename, m.media_type,
                m.title, m.description, m.tags, m.duration,
                m.created_at, m.is_processed,
                af.combined_features {operator} :query_vector AS distance
            FROM media m
            JOIN audio_features af ON m.id = af.media_id
            WHERE m.is_processed = true
            ORDER BY af.combined_features {operator} :query_vector
            LIMIT :k
        """)
        
        result = db.session.execute(query, {'query_vector': vector_str, 'k': k})
        
        return KNNSearch._format_results(result, metric)
    
    @staticmethod
    def search_video(query_features: np.ndarray, k: int = 10,
                    metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN) -> List[Dict]:
        """
        Search for similar videos using feature vectors
        
        Args:
            query_features: Query feature vector (1354 dimensions)
            k: Number of results to return
            metric: Distance metric to use
            
        Returns:
            List of search results with similarity scores
        """
        if isinstance(metric, str):
            metric = DistanceMetric(metric.lower())
        
        operator = KNNSearch.DISTANCE_OPERATORS.get(metric, '<->')
        
        vector_str = '[' + ','.join(map(str, query_features.tolist())) + ']'
        
        query = text(f"""
            SELECT 
                m.id, m.filename, m.original_filename, m.media_type,
                m.title, m.description, m.tags, m.duration, m.width, m.height,
                m.created_at, m.is_processed,
                vf.combined_features {operator} :query_vector AS distance
            FROM media m
            JOIN video_features vf ON m.id = vf.media_id
            WHERE m.is_processed = true
            ORDER BY vf.combined_features {operator} :query_vector
            LIMIT :k
        """)
        
        result = db.session.execute(query, {'query_vector': vector_str, 'k': k})
        
        return KNNSearch._format_results(result, metric)
    
    @staticmethod
    def search_by_type(query_features: np.ndarray, media_type: str, k: int = 10,
                       metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN) -> List[Dict]:
        """
        Search for similar media of a specific type
        
        Args:
            query_features: Query feature vector
            media_type: Type of media ('image', 'audio', 'video')
            k: Number of results to return
            metric: Distance metric to use
            
        Returns:
            List of search results with similarity scores
        """
        if media_type == 'image':
            return KNNSearch.search_images(query_features, k, metric)
        elif media_type == 'audio':
            return KNNSearch.search_audio(query_features, k, metric)
        elif media_type == 'video':
            return KNNSearch.search_video(query_features, k, metric)
        else:
            raise ValueError(f"Unknown media type: {media_type}")
    
    @staticmethod
    def hybrid_search(query_features: np.ndarray, media_type: str,
                      metadata_filters: Dict, k: int = 10,
                      metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
                      weight_feature: float = 0.7,
                      weight_metadata: float = 0.3) -> List[Dict]:
        """
        Hybrid search combining feature similarity and metadata matching
        
        Args:
            query_features: Query feature vector
            media_type: Type of media
            metadata_filters: Dictionary of metadata filters
            k: Number of results
            metric: Distance metric
            weight_feature: Weight for feature similarity (0-1)
            weight_metadata: Weight for metadata match (0-1)
            
        Returns:
            List of search results with combined scores
        """
        # First, get feature-based results
        feature_results = KNNSearch.search_by_type(query_features, media_type, k * 3, metric)
        
        # Apply metadata filters and scoring
        filtered_results = []
        
        for result in feature_results:
            metadata_score = KNNSearch._calculate_metadata_score(result, metadata_filters)
            
            # Combine scores
            combined_score = (
                weight_feature * result['similarity'] +
                weight_metadata * metadata_score
            )
            
            result['metadata_score'] = metadata_score
            result['combined_score'] = combined_score
            filtered_results.append(result)
        
        # Sort by combined score and return top k
        filtered_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return filtered_results[:k]
    
    @staticmethod
    def _calculate_metadata_score(result: Dict, filters: Dict) -> float:
        """Calculate metadata matching score"""
        if not filters:
            return 1.0
        
        score = 0.0
        total_filters = 0
        
        # Title match
        if 'title' in filters and filters['title']:
            total_filters += 1
            if result.get('title'):
                query_title = filters['title'].lower()
                result_title = result['title'].lower()
                if query_title in result_title:
                    score += 1.0
                elif any(word in result_title for word in query_title.split()):
                    score += 0.5
        
        # Tag match
        if 'tags' in filters and filters['tags']:
            total_filters += 1
            result_tags = set(result.get('tags') or [])
            query_tags = set(filters['tags'])
            if result_tags and query_tags:
                overlap = len(result_tags & query_tags) / len(query_tags)
                score += overlap
        
        # Date range
        if 'date_from' in filters or 'date_to' in filters:
            total_filters += 1
            created_at = result.get('created_at')
            if created_at:
                in_range = True
                if 'date_from' in filters and created_at < filters['date_from']:
                    in_range = False
                if 'date_to' in filters and created_at > filters['date_to']:
                    in_range = False
                if in_range:
                    score += 1.0
        
        return score / total_filters if total_filters > 0 else 1.0
    
    @staticmethod
    def _format_results(result, metric: DistanceMetric) -> List[Dict]:
        """Format database results as list of dictionaries"""
        results = []
        
        for row in result:
            row_dict = row._asdict()
            distance = row_dict.pop('distance', 0)
            
            # Convert distance to similarity score
            similarity = distance_to_similarity(distance, metric)
            
            results.append({
                **row_dict,
                'distance': distance,
                'similarity': similarity,
                'thumbnail_url': f"/api/media/{row_dict['id']}/thumbnail"
            })
        
        return results
    
    @staticmethod
    def create_vector_indexes():
        """
        Create IVFFlat indexes for faster vector search
        Should be called after populating data
        """
        indexes = [
            """CREATE INDEX IF NOT EXISTS idx_image_features_combined 
               ON image_features USING ivfflat (combined_features vector_l2_ops) 
               WITH (lists = 100)""",
            """CREATE INDEX IF NOT EXISTS idx_audio_features_combined 
               ON audio_features USING ivfflat (combined_features vector_l2_ops) 
               WITH (lists = 100)""",
            """CREATE INDEX IF NOT EXISTS idx_video_features_combined 
               ON video_features USING ivfflat (combined_features vector_l2_ops) 
               WITH (lists = 100)""",
        ]
        
        for index_sql in indexes:
            try:
                db.session.execute(text(index_sql))
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                print(f"Warning: Could not create index: {e}")
