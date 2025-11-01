"""
Recombine Core - UIP Step 2 IEL Ontological Synthesis Gateway

Central module for IEL output synthesis, merging outputs from multiple IEL domains
into coherent unified representations for downstream lambda processing.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("IEL_ONTO_KIT")


def merge_outputs(
    input_data: Dict[str, Any], active_domains: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge IEL outputs from multiple domains into unified representation

    Args:
        input_data: Processed linguistic context from Step 1
        active_domains: List of active IEL domain configurations

    Returns:
        Dict containing merged IEL outputs ready for lambda processing
    """
    try:
        logger.info(f"Starting IEL output merge for {len(active_domains)} domains")

        # Initialize merge context
        merge_context = {
            "timestamp": datetime.utcnow().isoformat(),
            "source_domains": [domain.get("name") for domain in active_domains],
            "input_complexity": _assess_input_complexity(input_data),
            "merged_vectors": {},
            "ontological_alignments": {},
            "synthesis_metadata": {},
        }

        # Extract Trinity vectors from input
        trinity_vectors = input_data.get("trinity_vectors", {})

        # Process each active domain
        domain_outputs = {}
        for domain in active_domains:
            domain_name = domain.get("name")
            ontological_prop = domain.get("ontological_property")

            # Generate domain-specific output
            domain_output = _process_domain_output(
                domain_name, ontological_prop, trinity_vectors, input_data
            )

            domain_outputs[domain_name] = domain_output

            # Store ontological alignment
            merge_context["ontological_alignments"][domain_name] = {
                "property": ontological_prop,
                "trinity_weights": domain.get("trinity_weights", {}),
                "activation_strength": domain_output.get("activation_strength", 0.0),
            }

        # Synthesize merged vectors
        merged_vectors = _synthesize_vectors(domain_outputs, active_domains)
        merge_context["merged_vectors"] = merged_vectors

        # Calculate synthesis quality metrics
        synthesis_quality = _calculate_synthesis_quality(domain_outputs, merged_vectors)
        merge_context["synthesis_metadata"] = {
            "quality_score": synthesis_quality,
            "vector_coherence": _measure_vector_coherence(merged_vectors),
            "domain_coverage": len(domain_outputs) / len(active_domains),
            "processing_time": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"IEL merge completed: quality={synthesis_quality:.3f}, "
            f"domains={len(domain_outputs)}"
        )

        return {
            "status": "ok",
            "payload": merge_context,
            "metadata": {
                "stage": "recombine",
                "domains_processed": len(domain_outputs),
                "synthesis_quality": synthesis_quality,
            },
        }

    except Exception as e:
        logger.error(f"IEL output merge failed: {e}")
        raise


def _assess_input_complexity(input_data: Dict[str, Any]) -> float:
    """Assess complexity of input linguistic data"""
    try:
        # Factors for complexity assessment
        factors = {
            "entity_count": len(input_data.get("entities", [])),
            "intent_diversity": len(input_data.get("intent_classification", {})),
            "linguistic_depth": input_data.get("processing_depth", 1),
            "semantic_richness": len(input_data.get("semantic_relations", [])),
        }

        # Weighted complexity calculation
        weights = {
            "entity_count": 0.3,
            "intent_diversity": 0.25,
            "linguistic_depth": 0.25,
            "semantic_richness": 0.2,
        }

        complexity = sum(factors[k] * weights[k] for k in factors.keys())
        return min(complexity / 10.0, 1.0)  # Normalize to [0,1]

    except Exception:
        return 0.5  # Default medium complexity


def _process_domain_output(
    domain_name: str,
    ontological_prop: str,
    trinity_vectors: Dict[str, Any],
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Process output for a specific IEL domain"""
    try:
        # Extract relevant Trinity components
        e_vector = trinity_vectors.get("ethical", np.array([0.0, 0.0, 0.0]))
        g_vector = trinity_vectors.get("gnoseological", np.array([0.0, 0.0, 0.0]))
        t_vector = trinity_vectors.get("teleological", np.array([0.0, 0.0, 0.0]))

        # Calculate domain activation based on ontological property alignment
        activation_strength = _calculate_domain_activation(
            domain_name, ontological_prop, e_vector, g_vector, t_vector
        )

        # Generate domain-specific representation
        domain_vector = _generate_domain_vector(
            domain_name, activation_strength, trinity_vectors
        )

        return {
            "domain_name": domain_name,
            "ontological_property": ontological_prop,
            "activation_strength": activation_strength,
            "domain_vector": domain_vector.tolist(),
            "trinity_alignment": {
                "ethical": float(
                    np.dot(
                        domain_vector[: min(len(domain_vector), len(e_vector))],
                        e_vector[: min(len(domain_vector), len(e_vector))],
                    )
                ),
                "gnoseological": float(
                    np.dot(
                        domain_vector[: min(len(domain_vector), len(g_vector))],
                        g_vector[: min(len(domain_vector), len(g_vector))],
                    )
                ),
                "teleological": float(
                    np.dot(
                        domain_vector[: min(len(domain_vector), len(t_vector))],
                        t_vector[: min(len(domain_vector), len(t_vector))],
                    )
                ),
            },
        }

    except Exception as e:
        logger.warning(f"Domain output processing failed for {domain_name}: {e}")
        return {
            "domain_name": domain_name,
            "ontological_property": ontological_prop,
            "activation_strength": 0.0,
            "domain_vector": [0.0] * 8,  # Default 8-dimensional vector
            "trinity_alignment": {
                "ethical": 0.0,
                "gnoseological": 0.0,
                "teleological": 0.0,
            },
        }


def _calculate_domain_activation(
    domain_name: str,
    ontological_prop: str,
    e_vector: np.ndarray,
    g_vector: np.ndarray,
    t_vector: np.ndarray,
) -> float:
    """Calculate activation strength for IEL domain"""
    try:
        # Base activation from domain-property alignment
        base_activation = 0.5

        # Trinity alignment factors
        trinity_norms = {
            "ethical": np.linalg.norm(e_vector),
            "gnoseological": np.linalg.norm(g_vector),
            "teleological": np.linalg.norm(t_vector),
        }

        # Domain-specific Trinity weighting
        domain_weights = _get_domain_trinity_weights(domain_name)

        # Calculate weighted Trinity activation
        trinity_activation = sum(
            trinity_norms[dim] * domain_weights.get(dim, 0.33)
            for dim in trinity_norms.keys()
        )

        # Combine base and Trinity activations
        final_activation = base_activation * 0.4 + trinity_activation * 0.6

        return min(max(final_activation, 0.0), 1.0)  # Clamp to [0,1]

    except Exception:
        return 0.5  # Default medium activation


def _get_domain_trinity_weights(domain_name: str) -> Dict[str, float]:
    """Get Trinity weighting for specific IEL domain"""
    # Default balanced weights
    default_weights = {"ethical": 0.33, "gnoseological": 0.33, "teleological": 0.34}

    # Domain-specific Trinity emphasis
    domain_emphasis = {
        "ModalPraxis": {"gnoseological": 0.5, "ethical": 0.3, "teleological": 0.2},
        "GnosiPraxis": {"gnoseological": 0.6, "ethical": 0.2, "teleological": 0.2},
        "ThemiPraxis": {"ethical": 0.6, "gnoseological": 0.2, "teleological": 0.2},
        "DynaPraxis": {"teleological": 0.5, "ethical": 0.25, "gnoseological": 0.25},
        "HexiPraxis": {"ethical": 0.4, "teleological": 0.4, "gnoseological": 0.2},
    }

    return domain_emphasis.get(domain_name, default_weights)


def _generate_domain_vector(
    domain_name: str, activation_strength: float, trinity_vectors: Dict[str, Any]
) -> np.ndarray:
    """Generate normalized domain representation vector"""
    try:
        # Create 8-dimensional domain vector
        base_vector = np.random.normal(0.0, 0.1, 8)  # Small random initialization

        # Scale by activation strength
        domain_vector = base_vector * activation_strength

        # Add Trinity influence
        trinity_influence = np.zeros(8)
        for i, (dim, vector) in enumerate(trinity_vectors.items()):
            if isinstance(vector, (list, np.ndarray)) and len(vector) >= 3:
                # Map Trinity components to domain vector dimensions
                trinity_influence[i % 8] += np.mean(vector[:3]) * 0.1

        # Combine base vector with Trinity influence
        final_vector = domain_vector + trinity_influence

        # Normalize to unit length
        norm = np.linalg.norm(final_vector)
        if norm > 0:
            final_vector = final_vector / norm

        return final_vector

    except Exception:
        # Return normalized default vector
        return np.array([0.125] * 8)  # Uniform 8D unit vector


def _synthesize_vectors(
    domain_outputs: Dict[str, Dict[str, Any]], active_domains: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Synthesize merged vectors from all domain outputs"""
    try:
        # Extract all domain vectors
        vectors = []
        weights = []

        for domain_name, output in domain_outputs.items():
            domain_vector = np.array(output.get("domain_vector", [0.0] * 8))
            activation = output.get("activation_strength", 0.0)

            vectors.append(domain_vector)
            weights.append(activation)

        # Convert to numpy arrays
        vectors_array = np.array(vectors)
        weights_array = np.array(weights)

        # Calculate weighted average vector
        if len(vectors) > 0 and np.sum(weights_array) > 0:
            weighted_avg = np.average(vectors_array, axis=0, weights=weights_array)
        else:
            weighted_avg = np.zeros(8)

        # Calculate synthesis statistics
        vector_std = np.std(vectors_array, axis=0) if len(vectors) > 1 else np.zeros(8)
        coherence_score = _measure_vector_coherence({"weighted_average": weighted_avg})

        return {
            "weighted_average": weighted_avg.tolist(),
            "vector_std": vector_std.tolist(),
            "coherence_score": coherence_score,
            "synthesis_confidence": min(np.mean(weights_array), 1.0),
        }

    except Exception as e:
        logger.warning(f"Vector synthesis failed: {e}")
        return {
            "weighted_average": [0.0] * 8,
            "vector_std": [0.0] * 8,
            "coherence_score": 0.0,
            "synthesis_confidence": 0.0,
        }


def _calculate_synthesis_quality(
    domain_outputs: Dict[str, Dict[str, Any]], merged_vectors: Dict[str, Any]
) -> float:
    """Calculate overall synthesis quality score"""
    try:
        # Quality factors
        activation_scores = [
            output.get("activation_strength", 0.0) for output in domain_outputs.values()
        ]

        # Mean activation strength
        mean_activation = np.mean(activation_scores) if activation_scores else 0.0

        # Vector coherence
        coherence = merged_vectors.get("coherence_score", 0.0)

        # Synthesis confidence
        confidence = merged_vectors.get("synthesis_confidence", 0.0)

        # Domain coverage (proportion of domains with good activation)
        good_activations = sum(1 for score in activation_scores if score > 0.3)
        coverage = (
            good_activations / len(activation_scores) if activation_scores else 0.0
        )

        # Weighted quality calculation
        quality = (
            mean_activation * 0.3
            + coherence * 0.25
            + confidence * 0.25
            + coverage * 0.2
        )

        return min(max(quality, 0.0), 1.0)

    except Exception:
        return 0.0


def _measure_vector_coherence(vectors: Dict[str, Any]) -> float:
    """Measure coherence of synthesized vectors"""
    try:
        # For now, use norm of weighted average as coherence measure
        weighted_avg = vectors.get("weighted_average", [0.0] * 8)
        norm = np.linalg.norm(weighted_avg)

        # Normalize coherence to [0,1]
        return min(norm, 1.0)

    except Exception:
        return 0.0
