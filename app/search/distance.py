"""
Distance Metrics for Similarity Search
Implements Euclidean and Manhattan distance calculations
"""
import numpy as np
from typing import List, Tuple, Union
from enum import Enum


class DistanceMetric(Enum):
    """Supported distance metrics"""
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
    COSINE = 'cosine'


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors
    
    L2 distance: sqrt(sum((v1 - v2)^2))
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Euclidean distance
    """
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


def manhattan_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate Manhattan distance between two vectors
    
    L1 distance: sum(|v1 - v2|)
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Manhattan distance
    """
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    
    return float(np.sum(np.abs(v1 - v2)))


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate Cosine distance between two vectors
    
    1 - cosine_similarity = 1 - (v1 · v2) / (||v1|| * ||v2||)
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Cosine distance (0 = identical, 2 = opposite)
    """
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 1.0
    
    similarity = np.dot(v1, v2) / (norm1 * norm2)
    return float(1 - similarity)


def calculate_distance(v1: np.ndarray, v2: np.ndarray, metric: Union[str, DistanceMetric]) -> float:
    """
    Calculate distance between two vectors using specified metric
    
    Args:
        v1: First vector
        v2: Second vector
        metric: Distance metric to use
        
    Returns:
        Distance value
    """
    if isinstance(metric, str):
        metric = DistanceMetric(metric.lower())
    
    if metric == DistanceMetric.EUCLIDEAN:
        return euclidean_distance(v1, v2)
    elif metric == DistanceMetric.MANHATTAN:
        return manhattan_distance(v1, v2)
    elif metric == DistanceMetric.COSINE:
        return cosine_distance(v1, v2)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def distance_to_similarity(distance: float, metric: Union[str, DistanceMetric], 
                           max_distance: float = None) -> float:
    """
    Convert distance to similarity score (0-1, where 1 is most similar)
    
    Args:
        distance: Distance value
        metric: Distance metric used
        max_distance: Maximum expected distance (for normalization)
        
    Returns:
        Similarity score between 0 and 1
    """
    if isinstance(metric, str):
        metric = DistanceMetric(metric.lower())
    
    if metric == DistanceMetric.COSINE:
        # Cosine distance is already 0-2, convert to 0-1 similarity
        return max(0, 1 - distance)
    else:
        # For Euclidean and Manhattan, use exponential decay
        # This gives a smooth curve where small distances have high similarity
        if max_distance is None:
            # Default max distance (can be tuned based on data)
            max_distance = 10.0
        
        # Normalize and convert to similarity
        normalized = distance / max_distance
        similarity = np.exp(-normalized)
        return float(min(1, max(0, similarity)))


def batch_distances(query: np.ndarray, candidates: List[np.ndarray], 
                    metric: Union[str, DistanceMetric]) -> List[Tuple[int, float]]:
    """
    Calculate distances from query to all candidates
    
    Args:
        query: Query vector
        candidates: List of candidate vectors
        metric: Distance metric to use
        
    Returns:
        List of (index, distance) tuples, sorted by distance
    """
    distances = []
    
    for i, candidate in enumerate(candidates):
        dist = calculate_distance(query, candidate, metric)
        distances.append((i, dist))
    
    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[1])
    
    return distances
