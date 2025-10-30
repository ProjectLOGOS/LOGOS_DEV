"""
Vector Norms - UIP Step 2 IEL Ontological Synthesis Gateway

Numeric stability utilities for vector operations and normalization.
Provides robust numerical methods for IEL domain vector processing.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger("IEL_ONTO_KIT")


def normalize_vector(vector: Union[List[float], np.ndarray], 
                    method: str = "l2", 
                    epsilon: float = 1e-10) -> Dict[str, Any]:
    """
    Normalize vector using specified method with numerical stability
    
    Args:
        vector: Input vector to normalize
        method: Normalization method ('l2', 'l1', 'max', 'unit')
        epsilon: Small value to prevent division by zero
        
    Returns:
        Dict containing normalized vector and stability metrics
    """
    try:
        logger.debug(f"Normalizing vector with method: {method}")
        
        # Convert to numpy array
        vec_array = np.array(vector, dtype=np.float64)
        
        # Check for invalid values
        if np.any(np.isnan(vec_array)) or np.any(np.isinf(vec_array)):
            logger.warning("Input vector contains NaN or Inf values")
            vec_array = np.nan_to_num(vec_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply normalization method
        if method == "l2":
            normalized_vec, norm_value = _l2_normalize(vec_array, epsilon)
        elif method == "l1":
            normalized_vec, norm_value = _l1_normalize(vec_array, epsilon)
        elif method == "max":
            normalized_vec, norm_value = _max_normalize(vec_array, epsilon)
        elif method == "unit":
            normalized_vec, norm_value = _unit_normalize(vec_array, epsilon)
        else:
            logger.warning(f"Unknown normalization method: {method}, using L2")
            normalized_vec, norm_value = _l2_normalize(vec_array, epsilon)
        
        # Compute stability metrics
        stability_metrics = _compute_vector_stability(vec_array, normalized_vec)
        
        # Verify normalization quality
        quality_score = _assess_normalization_quality(normalized_vec, method)
        
        return {
            "status": "ok",
            "payload": {
                "normalized_vector": normalized_vec.tolist(),
                "original_norm": float(norm_value),
                "normalization_method": method,
                "stability_metrics": stability_metrics,
                "quality_score": quality_score
            },
            "metadata": {
                "stage": "vector_normalization",
                "input_dimension": len(vector),
                "numerical_stability": stability_metrics.get("condition_number", 0.0)
            }
        }
        
    except Exception as e:
        logger.error(f"Vector normalization failed: {e}")
        return {
            "status": "error",
            "payload": {"error": str(e), "fallback_vector": [0.0] * len(vector)},
            "metadata": {"stage": "vector_normalization"}
        }


def compute_stability_metrics(vectors: List[Union[List[float], np.ndarray]]) -> Dict[str, Any]:
    """
    Compute numerical stability metrics for a collection of vectors
    
    Args:
        vectors: Collection of vectors to analyze
        
    Returns:
        Dict containing comprehensive stability analysis
    """
    try:
        logger.debug(f"Computing stability metrics for {len(vectors)} vectors")
        
        if not vectors:
            return {
                "status": "error",
                "payload": {"error": "No vectors provided"},
                "metadata": {"stage": "stability_analysis"}
            }
        
        # Convert vectors to numpy arrays
        vector_arrays = []
        for i, vec in enumerate(vectors):
            try:
                vec_array = np.array(vec, dtype=np.float64)
                # Clean invalid values
                vec_array = np.nan_to_num(vec_array, nan=0.0, posinf=1.0, neginf=-1.0)
                vector_arrays.append(vec_array)
            except Exception as e:
                logger.warning(f"Failed to process vector {i}: {e}")
                continue
        
        if not vector_arrays:
            return {
                "status": "error",
                "payload": {"error": "No valid vectors found"},
                "metadata": {"stage": "stability_analysis"}
            }
        
        # Compute stability metrics
        stability_analysis = {
            "vector_count": len(vector_arrays),
            "dimension_consistency": _check_dimension_consistency(vector_arrays),
            "numerical_stability": _analyze_numerical_stability(vector_arrays),
            "distribution_metrics": _compute_distribution_metrics(vector_arrays),
            "correlation_analysis": _compute_correlation_metrics(vector_arrays),
            "outlier_detection": _detect_outliers(vector_arrays)
        }
        
        # Overall stability score
        overall_stability = _calculate_overall_stability_score(stability_analysis)
        
        logger.info(f"Stability analysis completed: overall_score={overall_stability:.3f}")
        
        return {
            "status": "ok",
            "payload": {
                "stability_analysis": stability_analysis,
                "overall_stability_score": overall_stability,
                "recommendations": _generate_stability_recommendations(stability_analysis)
            },
            "metadata": {
                "stage": "stability_analysis",
                "vectors_analyzed": len(vector_arrays),
                "stability_score": overall_stability
            }
        }
        
    except Exception as e:
        logger.error(f"Stability metrics computation failed: {e}")
        return {
            "status": "error",
            "payload": {"error": str(e)},
            "metadata": {"stage": "stability_analysis"}
        }


def _l2_normalize(vector: np.ndarray, epsilon: float) -> Tuple[np.ndarray, float]:
    """L2 (Euclidean) normalization"""
    norm = np.linalg.norm(vector)
    if norm < epsilon:
        # Handle zero or near-zero vector
        normalized = np.zeros_like(vector)
        if len(vector) > 0:
            normalized[0] = 1.0  # Set first component to 1
        return normalized, epsilon
    return vector / norm, norm


def _l1_normalize(vector: np.ndarray, epsilon: float) -> Tuple[np.ndarray, float]:
    """L1 (Manhattan) normalization"""
    norm = np.sum(np.abs(vector))
    if norm < epsilon:
        # Handle zero or near-zero vector
        normalized = np.zeros_like(vector)
        if len(vector) > 0:
            normalized[0] = 1.0
        return normalized, epsilon
    return vector / norm, norm


def _max_normalize(vector: np.ndarray, epsilon: float) -> Tuple[np.ndarray, float]:
    """Max (infinity norm) normalization"""
    norm = np.max(np.abs(vector))
    if norm < epsilon:
        # Handle zero or near-zero vector
        normalized = np.zeros_like(vector)
        if len(vector) > 0:
            normalized[0] = 1.0
        return normalized, epsilon
    return vector / norm, norm


def _unit_normalize(vector: np.ndarray, epsilon: float) -> Tuple[np.ndarray, float]:
    """Unit sphere normalization (same as L2 but ensures unit length)"""
    return _l2_normalize(vector, epsilon)


def _compute_vector_stability(original: np.ndarray, normalized: np.ndarray) -> Dict[str, float]:
    """Compute stability metrics for vector normalization"""
    try:
        metrics = {}
        
        # Condition number (measure of numerical stability)
        if len(original) > 1:
            # Create matrix from original vector for condition number
            matrix = np.outer(original, original)
            if np.linalg.det(matrix) != 0:
                metrics["condition_number"] = float(np.linalg.cond(matrix))
            else:
                metrics["condition_number"] = float('inf')
        else:
            metrics["condition_number"] = 1.0
        
        # Preservation of direction (cosine similarity)
        orig_norm = np.linalg.norm(original)
        if orig_norm > 0:
            cosine_sim = np.dot(original, normalized) / orig_norm
            metrics["direction_preservation"] = float(np.abs(cosine_sim))
        else:
            metrics["direction_preservation"] = 0.0
        
        # Magnitude ratio
        normalized_norm = np.linalg.norm(normalized)
        if orig_norm > 0:
            metrics["magnitude_ratio"] = float(normalized_norm / orig_norm)
        else:
            metrics["magnitude_ratio"] = 0.0
        
        # Numerical precision loss
        if orig_norm > 0:
            relative_error = np.linalg.norm(original - normalized * orig_norm) / orig_norm
            metrics["precision_loss"] = float(relative_error)
        else:
            metrics["precision_loss"] = 0.0
        
        return metrics
        
    except Exception as e:
        logger.warning(f"Stability computation failed: {e}")
        return {
            "condition_number": float('inf'),
            "direction_preservation": 0.0,
            "magnitude_ratio": 0.0,
            "precision_loss": 1.0
        }


def _assess_normalization_quality(normalized_vector: np.ndarray, method: str) -> float:
    """Assess quality of normalization result"""
    try:
        # Check if vector is properly normalized
        if method in ["l2", "unit"]:
            expected_norm = 1.0
            actual_norm = np.linalg.norm(normalized_vector)
        elif method == "l1":
            expected_norm = 1.0
            actual_norm = np.sum(np.abs(normalized_vector))
        elif method == "max":
            expected_norm = 1.0
            actual_norm = np.max(np.abs(normalized_vector))
        else:
            return 0.5  # Unknown method
        
        # Quality based on proximity to expected norm
        norm_error = abs(actual_norm - expected_norm)
        quality = max(0.0, 1.0 - norm_error * 10.0)  # Scale error
        
        # Penalize for NaN or Inf values
        if np.any(np.isnan(normalized_vector)) or np.any(np.isinf(normalized_vector)):
            quality *= 0.1
        
        return quality
        
    except Exception:
        return 0.0


def _check_dimension_consistency(vectors: List[np.ndarray]) -> Dict[str, Any]:
    """Check dimensional consistency across vectors"""
    try:
        dimensions = [len(vec) for vec in vectors]
        unique_dims = list(set(dimensions))
        
        return {
            "is_consistent": len(unique_dims) == 1,
            "dimensions": dimensions,
            "unique_dimensions": unique_dims,
            "most_common_dimension": max(set(dimensions), key=dimensions.count) if dimensions else 0
        }
        
    except Exception:
        return {
            "is_consistent": False,
            "dimensions": [],
            "unique_dimensions": [],
            "most_common_dimension": 0
        }


def _analyze_numerical_stability(vectors: List[np.ndarray]) -> Dict[str, float]:
    """Analyze numerical stability of vector collection"""
    try:
        if not vectors:
            return {"condition_number": float('inf'), "rank": 0, "determinant": 0.0}
        
        # Stack vectors into matrix (if same dimension)
        dims = [len(vec) for vec in vectors]
        if len(set(dims)) == 1 and dims[0] > 0:
            matrix = np.vstack(vectors)
            
            # Condition number
            cond_num = float(np.linalg.cond(matrix))
            
            # Matrix rank
            rank = int(np.linalg.matrix_rank(matrix))
            
            # Determinant (if square matrix)
            if matrix.shape[0] == matrix.shape[1]:
                det = float(np.linalg.det(matrix))
            else:
                # Use pseudo-determinant for non-square matrices
                det = float(np.linalg.det(np.dot(matrix.T, matrix)))
            
            return {
                "condition_number": cond_num,
                "rank": rank,
                "determinant": det
            }
        else:
            return {
                "condition_number": float('inf'),
                "rank": 0,
                "determinant": 0.0
            }
            
    except Exception:
        return {
            "condition_number": float('inf'),
            "rank": 0,
            "determinant": 0.0
        }


def _compute_distribution_metrics(vectors: List[np.ndarray]) -> Dict[str, Any]:
    """Compute distribution metrics across vectors"""
    try:
        if not vectors:
            return {}
        
        # Flatten all vectors for distribution analysis
        all_values = np.concatenate([vec.flatten() for vec in vectors])
        
        metrics = {
            "mean": float(np.mean(all_values)),
            "std": float(np.std(all_values)),
            "min": float(np.min(all_values)),
            "max": float(np.max(all_values)),
            "median": float(np.median(all_values)),
            "skewness": float(_compute_skewness(all_values)),
            "kurtosis": float(_compute_kurtosis(all_values))
        }
        
        return metrics
        
    except Exception:
        return {}


def _compute_correlation_metrics(vectors: List[np.ndarray]) -> Dict[str, Any]:
    """Compute correlation metrics between vectors"""
    try:
        if len(vectors) < 2:
            return {"pairwise_correlations": [], "average_correlation": 0.0}
        
        # Check dimension consistency
        dims = [len(vec) for vec in vectors]
        if len(set(dims)) != 1:
            return {"pairwise_correlations": [], "average_correlation": 0.0}
        
        # Compute pairwise correlations
        correlations = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                try:
                    corr = np.corrcoef(vectors[i], vectors[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(float(corr))
                except Exception:
                    continue
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        return {
            "pairwise_correlations": correlations,
            "average_correlation": float(avg_correlation),
            "correlation_std": float(np.std(correlations)) if correlations else 0.0
        }
        
    except Exception:
        return {"pairwise_correlations": [], "average_correlation": 0.0}


def _detect_outliers(vectors: List[np.ndarray]) -> Dict[str, Any]:
    """Detect outlier vectors using statistical methods"""
    try:
        if len(vectors) < 3:
            return {"outlier_indices": [], "outlier_scores": []}
        
        # Compute vector norms as outlier detection feature
        norms = [np.linalg.norm(vec) for vec in vectors]
        norms_array = np.array(norms)
        
        # Use IQR method for outlier detection
        q1 = np.percentile(norms_array, 25)
        q3 = np.percentile(norms_array, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_indices = []
        outlier_scores = []
        
        for i, norm in enumerate(norms):
            if norm < lower_bound or norm > upper_bound:
                outlier_indices.append(i)
                # Score based on distance from bounds
                if norm < lower_bound:
                    score = (lower_bound - norm) / iqr if iqr > 0 else 0
                else:
                    score = (norm - upper_bound) / iqr if iqr > 0 else 0
                outlier_scores.append(float(score))
        
        return {
            "outlier_indices": outlier_indices,
            "outlier_scores": outlier_scores,
            "detection_method": "IQR",
            "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
        }
        
    except Exception:
        return {"outlier_indices": [], "outlier_scores": []}


def _compute_skewness(data: np.ndarray) -> float:
    """Compute skewness of data distribution"""
    try:
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skew = np.mean(((data - mean) / std) ** 3)
        return skew
        
    except Exception:
        return 0.0


def _compute_kurtosis(data: np.ndarray) -> float:
    """Compute kurtosis of data distribution"""
    try:
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        kurt = np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        return kurt
        
    except Exception:
        return 0.0


def _calculate_overall_stability_score(stability_analysis: Dict[str, Any]) -> float:
    """Calculate overall stability score from analysis results"""
    try:
        # Extract key metrics
        dimension_consistent = stability_analysis.get("dimension_consistency", {}).get("is_consistent", False)
        condition_number = stability_analysis.get("numerical_stability", {}).get("condition_number", float('inf'))
        avg_correlation = stability_analysis.get("correlation_analysis", {}).get("average_correlation", 0.0)
        outlier_count = len(stability_analysis.get("outlier_detection", {}).get("outlier_indices", []))
        vector_count = stability_analysis.get("vector_count", 1)
        
        # Scoring components
        dimension_score = 1.0 if dimension_consistent else 0.0
        
        # Condition number score (better for smaller condition numbers)
        if np.isinf(condition_number) or condition_number > 1e12:
            condition_score = 0.0
        elif condition_number < 10:
            condition_score = 1.0
        else:
            condition_score = max(0.0, 1.0 - np.log10(condition_number) / 6.0)
        
        # Correlation score (moderate correlation is good)
        correlation_score = 1.0 - abs(avg_correlation)  # Prefer independence
        
        # Outlier score (fewer outliers is better)
        outlier_ratio = outlier_count / vector_count if vector_count > 0 else 0
        outlier_score = max(0.0, 1.0 - outlier_ratio * 2.0)
        
        # Weighted overall score
        overall_score = (
            dimension_score * 0.3 +
            condition_score * 0.3 +
            correlation_score * 0.2 +
            outlier_score * 0.2
        )
        
        return min(max(overall_score, 0.0), 1.0)
        
    except Exception:
        return 0.0


def _generate_stability_recommendations(stability_analysis: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on stability analysis"""
    recommendations = []
    
    try:
        # Dimension consistency
        if not stability_analysis.get("dimension_consistency", {}).get("is_consistent", True):
            recommendations.append("Ensure all vectors have consistent dimensions before processing")
        
        # Numerical stability
        condition_number = stability_analysis.get("numerical_stability", {}).get("condition_number", 1.0)
        if np.isinf(condition_number) or condition_number > 1e6:
            recommendations.append("Consider regularization techniques to improve numerical stability")
        
        # Outliers
        outlier_count = len(stability_analysis.get("outlier_detection", {}).get("outlier_indices", []))
        if outlier_count > 0:
            recommendations.append(f"Investigate {outlier_count} detected outlier vector(s)")
        
        # Distribution
        distribution = stability_analysis.get("distribution_metrics", {})
        std = distribution.get("std", 0.0)
        if std > 10.0:
            recommendations.append("High variance detected; consider normalization or scaling")
        
        if not recommendations:
            recommendations.append("Vector collection appears numerically stable")
        
    except Exception:
        recommendations.append("Unable to generate stability recommendations due to analysis errors")
    
    return recommendations