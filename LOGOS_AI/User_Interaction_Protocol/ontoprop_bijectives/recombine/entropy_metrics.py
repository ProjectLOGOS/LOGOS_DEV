"""
Entropy Metrics - UIP Step 2 IEL Ontological Synthesis Gateway

Computes entropy and information gain metrics for IEL distribution analysis
and mutual information assessment pre-translation.
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import entropy

logger = logging.getLogger("IEL_ONTO_KIT")


def assess_distribution(merged_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess entropy and information distribution of merged IEL outputs

    Args:
        merged_data: Merged IEL output data from recombine_core

    Returns:
        Dict containing entropy metrics and distribution analysis
    """
    try:
        logger.info("Starting entropy distribution assessment")

        # Extract vectors and activations
        payload = merged_data.get("payload", {})
        merged_vectors = payload.get("merged_vectors", {})
        ontological_alignments = payload.get("ontological_alignments", {})

        # Calculate various entropy metrics
        entropy_metrics = {
            "vector_entropy": _calculate_vector_entropy(merged_vectors),
            "activation_entropy": _calculate_activation_entropy(ontological_alignments),
            "domain_distribution_entropy": _calculate_domain_entropy(
                ontological_alignments
            ),
            "mutual_information": _calculate_mutual_information(
                merged_vectors, ontological_alignments
            ),
            "information_gain": _calculate_information_gain(merged_vectors),
        }

        # Calculate overall entropy score
        overall_entropy = _calculate_overall_entropy_score(entropy_metrics)

        # Generate distribution quality assessment
        distribution_quality = _assess_distribution_quality(entropy_metrics)

        logger.info(f"Entropy assessment completed: overall={overall_entropy:.3f}")

        return {
            "status": "ok",
            "payload": {
                "entropy_metrics": entropy_metrics,
                "overall_entropy": overall_entropy,
                "distribution_quality": distribution_quality,
                "assessment_metadata": {
                    "vector_dimensions": len(
                        merged_vectors.get("weighted_average", [])
                    ),
                    "domain_count": len(ontological_alignments),
                    "coherence_factor": merged_vectors.get("coherence_score", 0.0),
                },
            },
            "metadata": {
                "stage": "entropy_assessment",
                "quality_score": distribution_quality,
            },
        }

    except Exception as e:
        logger.error(f"Entropy assessment failed: {e}")
        return {
            "status": "error",
            "payload": {"error": str(e)},
            "metadata": {"stage": "entropy_assessment"},
        }


def _calculate_vector_entropy(merged_vectors: Dict[str, Any]) -> float:
    """Calculate entropy of the merged vector distribution"""
    try:
        weighted_avg = merged_vectors.get("weighted_average", [])
        if not weighted_avg:
            return 0.0

        # Convert to numpy array and normalize
        vector = np.array(weighted_avg)

        # Ensure non-negative values for entropy calculation
        vector_abs = np.abs(vector)

        # Normalize to create probability distribution
        if np.sum(vector_abs) > 0:
            probabilities = vector_abs / np.sum(vector_abs)
        else:
            probabilities = np.ones(len(vector)) / len(vector)

        # Calculate Shannon entropy
        return float(entropy(probabilities, base=2))

    except Exception as e:
        logger.warning(f"Vector entropy calculation failed: {e}")
        return 0.0


def _calculate_activation_entropy(ontological_alignments: Dict[str, Any]) -> float:
    """Calculate entropy of domain activation strengths"""
    try:
        activations = []
        for domain_data in ontological_alignments.values():
            activation = domain_data.get("activation_strength", 0.0)
            activations.append(max(activation, 0.001))  # Avoid log(0)

        if not activations:
            return 0.0

        # Normalize activations to probabilities
        activations = np.array(activations)
        probabilities = activations / np.sum(activations)

        # Calculate Shannon entropy
        return float(entropy(probabilities, base=2))

    except Exception as e:
        logger.warning(f"Activation entropy calculation failed: {e}")
        return 0.0


def _calculate_domain_entropy(ontological_alignments: Dict[str, Any]) -> float:
    """Calculate entropy of domain distribution based on ontological properties"""
    try:
        # Count ontological property types
        properties = []
        for domain_data in ontological_alignments.values():
            prop = domain_data.get("property", "unknown")
            properties.append(prop)

        if not properties:
            return 0.0

        # Calculate frequency distribution
        property_counts = Counter(properties)
        total_count = len(properties)

        # Create probability distribution
        probabilities = np.array(
            [count / total_count for count in property_counts.values()]
        )

        # Calculate Shannon entropy
        return float(entropy(probabilities, base=2))

    except Exception as e:
        logger.warning(f"Domain entropy calculation failed: {e}")
        return 0.0


def _calculate_mutual_information(
    merged_vectors: Dict[str, Any], ontological_alignments: Dict[str, Any]
) -> float:
    """Calculate mutual information between vectors and domain activations"""
    try:
        weighted_avg = merged_vectors.get("weighted_average", [])
        if not weighted_avg or not ontological_alignments:
            return 0.0

        # Get activation strengths
        activations = []
        for domain_data in ontological_alignments.values():
            activation = domain_data.get("activation_strength", 0.0)
            activations.append(activation)

        if len(activations) == 0:
            return 0.0

        # Discretize vector components and activations for MI calculation
        vector = np.array(weighted_avg)
        activations = np.array(activations)

        # Simple discretization (could be improved with more sophisticated binning)
        vector_bins = np.digitize(
            vector, bins=np.linspace(vector.min(), vector.max(), 5)
        )
        activation_bins = np.digitize(activations, bins=np.linspace(0, 1, 5))

        # Calculate joint and marginal distributions
        joint_hist, _, _ = np.histogram2d(vector_bins, activation_bins, bins=5)
        joint_prob = joint_hist / np.sum(joint_hist)

        # Marginal distributions
        vector_prob = np.sum(joint_prob, axis=1)
        activation_prob = np.sum(joint_prob, axis=0)

        # Calculate mutual information
        mi = 0.0
        for i in range(len(vector_prob)):
            for j in range(len(activation_prob)):
                if (
                    joint_prob[i, j] > 0
                    and vector_prob[i] > 0
                    and activation_prob[j] > 0
                ):
                    mi += joint_prob[i, j] * np.log2(
                        joint_prob[i, j] / (vector_prob[i] * activation_prob[j])
                    )

        return float(mi)

    except Exception as e:
        logger.warning(f"Mutual information calculation failed: {e}")
        return 0.0


def _calculate_information_gain(merged_vectors: Dict[str, Any]) -> float:
    """Calculate information gain from vector synthesis"""
    try:
        weighted_avg = merged_vectors.get("weighted_average", [])
        vector_std = merged_vectors.get("vector_std", [])

        if not weighted_avg or not vector_std:
            return 0.0

        # Information gain based on reduction in uncertainty (std dev)
        avg_array = np.array(weighted_avg)
        std_array = np.array(vector_std)

        # Calculate signal-to-noise ratio as proxy for information gain
        snr_components = []
        for i in range(len(avg_array)):
            if std_array[i] > 0:
                snr = abs(avg_array[i]) / std_array[i]
                snr_components.append(snr)

        if snr_components:
            # Average SNR as information gain measure
            return float(np.mean(snr_components))
        else:
            return 0.0

    except Exception as e:
        logger.warning(f"Information gain calculation failed: {e}")
        return 0.0


def _calculate_overall_entropy_score(entropy_metrics: Dict[str, float]) -> float:
    """Calculate overall entropy score combining all metrics"""
    try:
        # Weight different entropy components
        weights = {
            "vector_entropy": 0.3,
            "activation_entropy": 0.25,
            "domain_distribution_entropy": 0.2,
            "mutual_information": 0.15,
            "information_gain": 0.1,
        }

        # Calculate weighted average
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in entropy_metrics:
                value = entropy_metrics[metric]
                if not np.isnan(value) and not np.isinf(value):
                    weighted_sum += value * weight
                    total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0

    except Exception as e:
        logger.warning(f"Overall entropy calculation failed: {e}")
        return 0.0


def _assess_distribution_quality(entropy_metrics: Dict[str, float]) -> float:
    """Assess overall quality of entropy distribution"""
    try:
        # Quality factors
        vector_entropy = entropy_metrics.get("vector_entropy", 0.0)
        activation_entropy = entropy_metrics.get("activation_entropy", 0.0)
        mutual_info = entropy_metrics.get("mutual_information", 0.0)
        info_gain = entropy_metrics.get("information_gain", 0.0)

        # Quality assessment criteria
        # - Moderate vector entropy indicates good diversity without chaos
        # - High activation entropy indicates good domain utilization
        # - High mutual information indicates good correlation
        # - High information gain indicates good synthesis

        vector_quality = _entropy_quality_score(
            vector_entropy, optimal_range=(1.5, 2.5)
        )
        activation_quality = min(
            activation_entropy / 3.0, 1.0
        )  # Normalize to domain count
        correlation_quality = min(mutual_info / 2.0, 1.0)  # Normalize MI
        synthesis_quality = min(info_gain / 5.0, 1.0)  # Normalize IG

        # Overall quality as weighted average
        quality = (
            vector_quality * 0.3
            + activation_quality * 0.3
            + correlation_quality * 0.2
            + synthesis_quality * 0.2
        )

        return min(max(quality, 0.0), 1.0)

    except Exception as e:
        logger.warning(f"Distribution quality assessment failed: {e}")
        return 0.0


def _entropy_quality_score(
    entropy_value: float, optimal_range: tuple = (1.0, 2.0)
) -> float:
    """Calculate quality score for entropy value based on optimal range"""
    try:
        min_optimal, max_optimal = optimal_range

        if min_optimal <= entropy_value <= max_optimal:
            return 1.0
        elif entropy_value < min_optimal:
            # Too low entropy (not diverse enough)
            return entropy_value / min_optimal
        else:
            # Too high entropy (too chaotic)
            decay_factor = max(0.1, 1.0 - (entropy_value - max_optimal) / max_optimal)
            return decay_factor

    except Exception:
        return 0.0
