"""
Bayesian Resolver - UIP Step 1 Component
=========================================

Bayesian inference for trinitarian vectors and uncertainty resolution.
Integrates advanced Bayesian reasoning for linguistic ambiguity resolution.

Adapted from: V2_Possible_Gap_Fillers/bayesian predictor/bayesian_inferencer.py
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from protocols.shared.system_imports import *


@dataclass
class BayesianResult:
    """Result from Bayesian inference process"""

    trinity_vector: Tuple[float, float, float]  # (E, G, T)
    complex_value: complex
    confidence: float
    source_terms: List[str]
    prior_strength: float
    posterior_distribution: Dict[str, float]
    evidence_strength: float
    uncertainty_reduced: float


class TrinityPriorDatabase:
    """Database of Trinity priors for Bayesian inference"""

    def __init__(self, prior_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.priors: Dict[str, Dict[str, float]] = {}

        # Load priors from configuration or use defaults
        if prior_path and Path(prior_path).exists():
            self._load_priors_from_file(prior_path)
        else:
            self._initialize_default_priors()

    def _load_priors_from_file(self, prior_path: str):
        """Load priors from JSON configuration file"""
        try:
            with open(prior_path, "r") as f:
                self.priors = json.load(f)
            self.logger.info(f"Loaded {len(self.priors)} priors from {prior_path}")
        except Exception as e:
            self.logger.error(f"Failed to load priors from {prior_path}: {e}")
            self._initialize_default_priors()

    def _initialize_default_priors(self):
        """Initialize default Trinity priors for common terms"""
        self.priors = {
            # Existence-heavy terms
            "being": {"E": 0.9, "G": 0.5, "T": 0.7},
            "exist": {"E": 0.95, "G": 0.4, "T": 0.6},
            "reality": {"E": 0.85, "G": 0.6, "T": 0.8},
            "substance": {"E": 0.8, "G": 0.5, "T": 0.6},
            # Goodness-heavy terms
            "good": {"E": 0.6, "G": 0.95, "T": 0.7},
            "virtue": {"E": 0.5, "G": 0.9, "T": 0.8},
            "moral": {"E": 0.4, "G": 0.85, "T": 0.7},
            "ethical": {"E": 0.5, "G": 0.88, "T": 0.75},
            "beauty": {"E": 0.7, "G": 0.9, "T": 0.8},
            # Truth-heavy terms
            "true": {"E": 0.7, "G": 0.6, "T": 0.95},
            "knowledge": {"E": 0.6, "G": 0.7, "T": 0.9},
            "wisdom": {"E": 0.7, "G": 0.8, "T": 0.95},
            "logic": {"E": 0.5, "G": 0.6, "T": 0.9},
            "proof": {"E": 0.6, "G": 0.7, "T": 0.95},
            # Balanced terms
            "unity": {"E": 0.8, "G": 0.8, "T": 0.8},
            "harmony": {"E": 0.7, "G": 0.85, "T": 0.8},
            "balance": {"E": 0.6, "G": 0.8, "T": 0.7},
            # Domain-specific terms
            "modal": {"E": 0.7, "G": 0.6, "T": 0.85},
            "necessity": {"E": 0.8, "G": 0.7, "T": 0.9},
            "possibility": {"E": 0.6, "G": 0.5, "T": 0.8},
            "temporal": {"E": 0.7, "G": 0.6, "T": 0.75},
            "eternal": {"E": 0.9, "G": 0.8, "T": 0.95},
            # Negative/uncertain terms
            "false": {"E": 0.3, "G": 0.2, "T": 0.1},
            "evil": {"E": 0.4, "G": 0.1, "T": 0.3},
            "doubt": {"E": 0.4, "G": 0.3, "T": 0.2},
            "uncertainty": {"E": 0.3, "G": 0.4, "T": 0.3},
        }

        self.logger.info(f"Initialized {len(self.priors)} default Trinity priors")

    def get_prior(self, term: str) -> Optional[Dict[str, float]]:
        """Get Trinity prior for specific term"""
        return self.priors.get(term.lower())

    def add_prior(self, term: str, e_val: float, g_val: float, t_val: float):
        """Add new Trinity prior"""
        self.priors[term.lower()] = {
            "E": max(0.0, min(1.0, e_val)),
            "G": max(0.0, min(1.0, g_val)),
            "T": max(0.0, min(1.0, t_val)),
        }

    def update_prior(
        self, term: str, evidence: Dict[str, float], learning_rate: float = 0.1
    ):
        """Update prior based on new evidence (Bayesian learning)"""
        if term.lower() not in self.priors:
            return

        prior = self.priors[term.lower()]

        # Bayesian update: posterior ∝ likelihood × prior
        for dimension in ["E", "G", "T"]:
            if dimension in evidence:
                likelihood = evidence[dimension]
                prior_val = prior[dimension]

                # Simple Bayesian update with learning rate
                updated_val = prior_val + learning_rate * (likelihood - prior_val)
                prior[dimension] = max(0.0, min(1.0, updated_val))


class BayesianTrinityInferencer:
    """Advanced Bayesian inference for Trinity vectors"""

    def __init__(self, prior_database: Optional[TrinityPriorDatabase] = None):
        self.logger = logging.getLogger(__name__)
        self.prior_db = prior_database or TrinityPriorDatabase()

        # Inference parameters
        self.confidence_threshold = 0.6
        self.min_evidence_terms = 1
        self.uncertainty_penalty = 0.1

    def infer_trinity_vector(
        self,
        keywords: List[str],
        weights: Optional[List[float]] = None,
        context_boost: Optional[Dict[str, float]] = None,
    ) -> BayesianResult:
        """
        Infer Trinity vector from keywords using Bayesian inference

        Args:
            keywords: List of keywords to analyze
            weights: Optional weights for each keyword
            context_boost: Optional contextual boosting factors

        Returns:
            BayesianResult: Complete inference results
        """
        if not keywords:
            raise ValueError("At least one keyword required for inference")

        # Normalize inputs
        keywords = [k.lower().strip() for k in keywords if k.strip()]
        weights = (
            weights
            if weights and len(weights) == len(keywords)
            else [1.0] * len(keywords)
        )
        context_boost = context_boost or {}

        # Collect evidence from priors
        evidence_terms = []
        e_evidence, g_evidence, t_evidence = [], [], []
        total_weight = 0.0

        for i, keyword in enumerate(keywords):
            prior = self.prior_db.get_prior(keyword)
            if prior:
                weight = weights[i]

                # Apply contextual boosting
                boost = context_boost.get(keyword, 1.0)
                effective_weight = weight * boost

                e_evidence.append(prior["E"] * effective_weight)
                g_evidence.append(prior["G"] * effective_weight)
                t_evidence.append(prior["T"] * effective_weight)

                total_weight += effective_weight
                evidence_terms.append(keyword)

        if not evidence_terms:
            raise ValueError("No valid priors found for given keywords")

        # Bayesian inference: compute posterior means
        e_posterior = sum(e_evidence) / total_weight
        g_posterior = sum(g_evidence) / total_weight
        t_posterior = sum(t_evidence) / total_weight

        # Normalize to [0,1] bounds
        e_final = max(0.0, min(1.0, e_posterior))
        g_final = max(0.0, min(1.0, g_posterior))
        t_final = max(0.0, min(1.0, t_posterior))

        # Calculate confidence metrics
        confidence = self._calculate_confidence(evidence_terms, total_weight, weights)
        prior_strength = min(1.0, total_weight / len(keywords))
        evidence_strength = len(evidence_terms) / len(keywords)

        # Calculate uncertainty reduction
        prior_uncertainty = self._calculate_prior_uncertainty()
        posterior_uncertainty = self._calculate_posterior_uncertainty(
            [e_final, g_final, t_final]
        )
        uncertainty_reduced = max(0.0, prior_uncertainty - posterior_uncertainty)

        # Create complex representation
        complex_val = complex(e_final * t_final, g_final)

        # Build posterior distribution summary
        posterior_dist = {
            "E_mean": e_final,
            "G_mean": g_final,
            "T_mean": t_final,
            "E_variance": self._calculate_dimension_variance(e_evidence, e_final),
            "G_variance": self._calculate_dimension_variance(g_evidence, g_final),
            "T_variance": self._calculate_dimension_variance(t_evidence, t_final),
        }

        result = BayesianResult(
            trinity_vector=(e_final, g_final, t_final),
            complex_value=complex_val,
            confidence=confidence,
            source_terms=evidence_terms,
            prior_strength=prior_strength,
            posterior_distribution=posterior_dist,
            evidence_strength=evidence_strength,
            uncertainty_reduced=uncertainty_reduced,
        )

        self.logger.debug(
            f"Bayesian inference completed: T=({e_final:.3f}, {g_final:.3f}, {t_final:.3f}), confidence={confidence:.3f}"
        )

        return result

    def resolve_ambiguity(
        self, ambiguous_terms: List[str], context_keywords: List[str]
    ) -> Dict[str, BayesianResult]:
        """
        Resolve ambiguous terms using contextual Bayesian inference

        Args:
            ambiguous_terms: Terms with uncertain meaning
            context_keywords: Context terms for disambiguation

        Returns:
            Dict mapping terms to their resolved Trinity vectors
        """
        results = {}

        # Build context boost from context keywords
        context_inference = self.infer_trinity_vector(context_keywords)
        context_boost = {
            "existence_context": context_inference.trinity_vector[0],
            "goodness_context": context_inference.trinity_vector[1],
            "truth_context": context_inference.trinity_vector[2],
        }

        # Resolve each ambiguous term
        for term in ambiguous_terms:
            try:
                # Use term + context for disambiguation
                combined_keywords = [term] + context_keywords[:3]  # Limit context

                result = self.infer_trinity_vector(
                    combined_keywords,
                    weights=[2.0]
                    + [0.5] * (len(combined_keywords) - 1),  # Boost target term
                    context_boost=context_boost,
                )

                results[term] = result

            except Exception as e:
                self.logger.error(f"Failed to resolve ambiguity for '{term}': {e}")

        return results

    def _calculate_confidence(
        self,
        evidence_terms: List[str],
        total_weight: float,
        original_weights: List[float],
    ) -> float:
        """Calculate confidence in inference results"""

        # Base confidence from evidence coverage
        coverage = (
            len(evidence_terms) / len(original_weights) if original_weights else 0
        )

        # Weight-based confidence
        weight_confidence = min(1.0, total_weight / (len(original_weights) * 2.0))

        # Evidence term quality (number of matched priors)
        quality_confidence = min(1.0, len(evidence_terms) / max(1, len(evidence_terms)))

        # Combined confidence with geometric mean
        confidence = (coverage * weight_confidence * quality_confidence) ** (1 / 3)

        return max(0.0, min(1.0, confidence))

    def _calculate_prior_uncertainty(self) -> float:
        """Calculate baseline uncertainty (maximum entropy)"""
        return -3 * (0.5 * np.log2(0.5))  # Max entropy for 3 dimensions

    def _calculate_posterior_uncertainty(self, trinity_values: List[float]) -> float:
        """Calculate posterior uncertainty"""
        # Shannon entropy-based uncertainty measure
        entropy_sum = 0.0
        for val in trinity_values:
            if 0 < val < 1:
                entropy_sum += -(val * np.log2(val) + (1 - val) * np.log2(1 - val))

        return entropy_sum

    def _calculate_dimension_variance(
        self, evidence_list: List[float], mean_val: float
    ) -> float:
        """Calculate variance for a Trinity dimension"""
        if len(evidence_list) < 2:
            return 0.0

        variance = sum((x - mean_val) ** 2 for x in evidence_list) / len(evidence_list)
        return variance


class BayesianResolver:
    """Main Bayesian resolution engine for UIP Step 1"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize components
        prior_path = self.config.get("prior_database_path")
        self.prior_db = TrinityPriorDatabase(prior_path)
        self.inferencer = BayesianTrinityInferencer(self.prior_db)

        self.logger.info("Bayesian resolver initialized")

    def resolve_linguistic_uncertainty(
        self, linguistic_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve linguistic uncertainty using Bayesian inference

        Args:
            linguistic_results: Results from linguistic analysis step

        Returns:
            Enhanced results with Bayesian uncertainty resolution
        """
        try:
            # Extract keywords and entities from linguistic analysis
            keywords = linguistic_results.get("keywords", [])
            entities = linguistic_results.get("entities", [])
            ambiguous_terms = linguistic_results.get("ambiguous_terms", [])

            # Combine for comprehensive analysis
            all_terms = keywords + [entity.get("text", "") for entity in entities]

            # Perform primary Trinity inference
            primary_result = self.inferencer.infer_trinity_vector(all_terms)

            # Resolve ambiguous terms if present
            disambiguation_results = {}
            if ambiguous_terms:
                disambiguation_results = self.inferencer.resolve_ambiguity(
                    ambiguous_terms, keywords
                )

            # Build enhanced results
            enhanced_results = {
                **linguistic_results,
                "bayesian_analysis": {
                    "primary_trinity_vector": primary_result.trinity_vector,
                    "complex_representation": str(primary_result.complex_value),
                    "inference_confidence": primary_result.confidence,
                    "evidence_terms": primary_result.source_terms,
                    "uncertainty_reduced": primary_result.uncertainty_reduced,
                    "posterior_distribution": primary_result.posterior_distribution,
                },
                "disambiguation": {
                    term: {
                        "trinity_vector": result.trinity_vector,
                        "confidence": result.confidence,
                    }
                    for term, result in disambiguation_results.items()
                },
                "resolution_metadata": {
                    "resolver_version": "1.0.0",
                    "prior_database_size": len(self.prior_db.priors),
                    "terms_analyzed": len(all_terms),
                    "terms_resolved": len(primary_result.source_terms),
                    "ambiguities_resolved": len(disambiguation_results),
                },
            }

            self.logger.info(
                f"Resolved linguistic uncertainty: {len(primary_result.source_terms)} terms, "
                f"confidence {primary_result.confidence:.3f}"
            )

            return enhanced_results

        except Exception as e:
            self.logger.error(f"Bayesian resolution failed: {e}")
            # Return original results with error metadata
            return {
                **linguistic_results,
                "bayesian_analysis": {"error": str(e)},
                "disambiguation": {},
                "resolution_metadata": {"error": True},
            }


# Global resolver instance
bayesian_resolver = BayesianResolver()


__all__ = [
    "BayesianResult",
    "TrinityPriorDatabase",
    "BayesianTrinityInferencer",
    "BayesianResolver",
    "bayesian_resolver",
]
