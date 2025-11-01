"""
PXL Consistency Checker - UIP Step 2 Component
==============================================

Advanced consistency validation system for PXL (Philosophically Extended Logic) analysis.
Detects logical contradictions, Trinity vector inconsistencies, and modal logic violations.

Integrates with: PXL relation mapper, modal validator, ontological validator, Trinity processor
Dependencies: V2 framework protocols, mathematical frameworks, formal logic systems
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
from LOGOS_AI.Advanced_Reasoning_Protocol.system_utillities.system_imports import *


class ConsistencyViolationType(Enum):
    """Types of consistency violations"""

    LOGICAL_CONTRADICTION = "logical_contradiction"
    MODAL_INCONSISTENCY = "modal_inconsistency"
    TRINITY_VECTOR_INCOHERENCE = "trinity_vector_incoherence"
    ONTOLOGICAL_VIOLATION = "ontological_violation"
    SEMANTIC_CONTRADICTION = "semantic_contradiction"
    DEFINITIONAL_CIRCULARITY = "definitional_circularity"
    HIERARCHICAL_INCONSISTENCY = "hierarchical_inconsistency"
    RELATIONAL_ASYMMETRY = "relational_asymmetry"
    CAUSAL_LOOP = "causal_loop"
    TEMPORAL_PARADOX = "temporal_paradox"


class ViolationSeverity(Enum):
    """Severity levels of violations"""

    CRITICAL = "critical"  # System-breaking contradictions
    MAJOR = "major"  # Significant logical problems
    MODERATE = "moderate"  # Concerning but not fatal
    MINOR = "minor"  # Weak inconsistencies
    WARNING = "warning"  # Potential issues


class ConsistencyScope(Enum):
    """Scope of consistency checking"""

    LOCAL = "local"  # Single concept or relation
    CLUSTER = "cluster"  # Within concept clusters
    NETWORK = "network"  # Across semantic networks
    GLOBAL = "global"  # Entire system


@dataclass
class ConsistencyViolation:
    """Individual consistency violation"""

    violation_type: ConsistencyViolationType
    severity: ViolationSeverity
    scope: ConsistencyScope
    description: str
    involved_concepts: List[str]
    involved_relations: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_resolution: Optional[str] = None
    confidence: float = 1.0
    trinity_analysis: Optional[Dict[str, Any]] = None
    modal_analysis: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f"{self.severity.value.upper()}: {self.description}"


@dataclass
class ConsistencyReport:
    """Comprehensive consistency analysis report"""

    total_violations: int
    violations_by_type: Dict[ConsistencyViolationType, List[ConsistencyViolation]]
    violations_by_severity: Dict[ViolationSeverity, List[ConsistencyViolation]]
    critical_violations: List[ConsistencyViolation]
    global_consistency_score: float
    local_consistency_scores: Dict[str, float]
    cluster_consistency_scores: Dict[str, float]
    resolution_recommendations: List[str]
    consistency_metrics: Dict[str, float]
    analysis_metadata: Dict[str, Any]


class LogicalConsistencyAnalyzer:
    """Logical contradiction and inference consistency analysis"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def check_logical_consistency(
        self,
        relations: List[Any],  # PXLRelation objects
        concept_properties: Dict[str, Dict[str, Any]],
    ) -> List[ConsistencyViolation]:
        """Check for logical contradictions and inconsistencies"""

        violations = []

        # Check for direct contradictions
        contradictions = self._detect_direct_contradictions(relations)
        violations.extend(contradictions)

        # Check for inference inconsistencies
        inference_violations = self._check_inference_consistency(relations)
        violations.extend(inference_violations)

        # Check for definitional circularity
        circular_violations = self._detect_definitional_circularity(relations)
        violations.extend(circular_violations)

        # Check property consistency
        property_violations = self._check_property_consistency(concept_properties)
        violations.extend(property_violations)

        self.logger.debug(f"Found {len(violations)} logical consistency violations")
        return violations

    def _detect_direct_contradictions(
        self, relations: List[Any]
    ) -> List[ConsistencyViolation]:
        """Detect direct logical contradictions"""
        violations = []

        # Build assertion map
        assertions = defaultdict(set)
        negations = defaultdict(set)

        for relation in relations:
            if hasattr(relation, "relation_type") and hasattr(
                relation, "source_concept"
            ):
                source = relation.source_concept
                target = relation.target_concept

                # Check for contradictory assertions
                key = f"{source}→{target}"

                if relation.relation_type.value in ["logical_implication", "entails"]:
                    assertions[key].add(relation)
                elif (
                    "not_" in relation.relation_type.value
                    or "contradiction" in relation.relation_type.value
                ):
                    negations[key].add(relation)

        # Find contradictions
        for key in assertions:
            if key in negations and assertions[key] and negations[key]:
                concepts = key.split("→")

                violation = ConsistencyViolation(
                    violation_type=ConsistencyViolationType.LOGICAL_CONTRADICTION,
                    severity=ViolationSeverity.CRITICAL,
                    scope=ConsistencyScope.LOCAL,
                    description=f"Direct contradiction: {key} both asserted and negated",
                    involved_concepts=concepts,
                    evidence={
                        "assertions": len(assertions[key]),
                        "negations": len(negations[key]),
                    },
                    suggested_resolution="Review and resolve contradictory statements",
                )
                violations.append(violation)

        return violations

    def _check_inference_consistency(
        self, relations: List[Any]
    ) -> List[ConsistencyViolation]:
        """Check consistency of logical inferences"""
        violations = []

        # Build inference chains
        inference_graph = self._build_inference_graph(relations)

        # Check for inconsistent inference chains
        try:
            # Detect cycles in inference graph
            if hasattr(nx, "simple_cycles"):
                cycles = list(nx.simple_cycles(inference_graph))

                for cycle in cycles[:10]:  # Limit to prevent performance issues
                    if len(cycle) > 1:  # Ignore self-loops
                        violation = ConsistencyViolation(
                            violation_type=ConsistencyViolationType.LOGICAL_CONTRADICTION,
                            severity=ViolationSeverity.MAJOR,
                            scope=ConsistencyScope.NETWORK,
                            description=f"Circular inference detected: {' → '.join(cycle)}",
                            involved_concepts=cycle,
                            evidence={"cycle_length": len(cycle)},
                            suggested_resolution="Break circular reasoning chain",
                        )
                        violations.append(violation)
        except Exception as e:
            self.logger.warning(f"Cycle detection failed: {e}")

        return violations

    def _detect_definitional_circularity(
        self, relations: List[Any]
    ) -> List[ConsistencyViolation]:
        """Detect circular definitions"""
        violations = []

        # Build definition graph
        definition_graph = nx.DiGraph()

        for relation in relations:
            if (
                hasattr(relation, "relation_type")
                and relation.relation_type.value == "definitional_relation"
            ):

                definition_graph.add_edge(
                    relation.source_concept, relation.target_concept, relation=relation
                )

        # Find strongly connected components (cycles)
        try:
            scc = list(nx.strongly_connected_components(definition_graph))

            for component in scc:
                if len(component) > 1:  # Circular definition
                    violation = ConsistencyViolation(
                        violation_type=ConsistencyViolationType.DEFINITIONAL_CIRCULARITY,
                        severity=ViolationSeverity.MAJOR,
                        scope=ConsistencyScope.CLUSTER,
                        description=f"Circular definition among: {', '.join(component)}",
                        involved_concepts=list(component),
                        suggested_resolution="Provide non-circular definitions for these concepts",
                    )
                    violations.append(violation)
        except Exception as e:
            self.logger.warning(f"SCC detection failed: {e}")

        return violations

    def _check_property_consistency(
        self, concept_properties: Dict[str, Dict[str, Any]]
    ) -> List[ConsistencyViolation]:
        """Check consistency of concept properties"""
        violations = []

        for concept, properties in concept_properties.items():
            # Check for contradictory boolean properties
            boolean_props = {k: v for k, v in properties.items() if isinstance(v, bool)}

            # Example: A concept cannot be both temporal and eternal
            if boolean_props.get("temporal", False) and boolean_props.get(
                "eternal", False
            ):
                violation = ConsistencyViolation(
                    violation_type=ConsistencyViolationType.ONTOLOGICAL_VIOLATION,
                    severity=ViolationSeverity.MAJOR,
                    scope=ConsistencyScope.LOCAL,
                    description=f"Concept '{concept}' cannot be both temporal and eternal",
                    involved_concepts=[concept],
                    evidence={"temporal": True, "eternal": True},
                    suggested_resolution="Clarify temporal nature of concept",
                )
                violations.append(violation)

        return violations

    def _build_inference_graph(self, relations: List[Any]) -> nx.DiGraph:
        """Build directed graph of logical inferences"""
        graph = nx.DiGraph()

        for relation in relations:
            if hasattr(relation, "relation_type") and hasattr(
                relation, "source_concept"
            ):
                if relation.relation_type.value in [
                    "logical_implication",
                    "entails",
                    "causal_relation",
                ]:
                    graph.add_edge(
                        relation.source_concept,
                        relation.target_concept,
                        relation=relation,
                    )

        return graph


class TrinityConsistencyAnalyzer:
    """Trinity vector consistency analysis"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.coherence_threshold = 0.3  # Minimum acceptable coherence

    def check_trinity_consistency(
        self,
        trinity_vectors: Dict[str, Tuple[float, float, float]],
        relations: List[Any],
    ) -> List[ConsistencyViolation]:
        """Check Trinity vector consistency"""

        violations = []

        # Check individual vector validity
        vector_violations = self._check_vector_validity(trinity_vectors)
        violations.extend(vector_violations)

        # Check relational coherence
        coherence_violations = self._check_relational_coherence(
            trinity_vectors, relations
        )
        violations.extend(coherence_violations)

        # Check Trinity-specific theological constraints
        theological_violations = self._check_theological_constraints(trinity_vectors)
        violations.extend(theological_violations)

        self.logger.debug(f"Found {len(violations)} Trinity consistency violations")
        return violations

    def _check_vector_validity(
        self, trinity_vectors: Dict[str, Tuple[float, float, float]]
    ) -> List[ConsistencyViolation]:
        """Check validity of individual Trinity vectors"""
        violations = []

        for concept, vector in trinity_vectors.items():
            e, g, t = vector

            # Check bounds
            if not all(0 <= v <= 1 for v in vector):
                violation = ConsistencyViolation(
                    violation_type=ConsistencyViolationType.TRINITY_VECTOR_INCOHERENCE,
                    severity=ViolationSeverity.CRITICAL,
                    scope=ConsistencyScope.LOCAL,
                    description=f"Trinity vector for '{concept}' has out-of-bounds values: {vector}",
                    involved_concepts=[concept],
                    evidence={"vector": vector},
                    suggested_resolution="Ensure Trinity dimensions are in [0,1] range",
                )
                violations.append(violation)

            # Check for degenerate vectors (all zeros)
            if all(v == 0 for v in vector):
                violation = ConsistencyViolation(
                    violation_type=ConsistencyViolationType.TRINITY_VECTOR_INCOHERENCE,
                    severity=ViolationSeverity.MAJOR,
                    scope=ConsistencyScope.LOCAL,
                    description=f"Trinity vector for '{concept}' is degenerate (all zeros)",
                    involved_concepts=[concept],
                    evidence={"vector": vector},
                    suggested_resolution="Assign meaningful Trinity dimensions",
                )
                violations.append(violation)

            # Check for extreme imbalance
            if max(vector) - min(vector) > 0.9:
                violation = ConsistencyViolation(
                    violation_type=ConsistencyViolationType.TRINITY_VECTOR_INCOHERENCE,
                    severity=ViolationSeverity.WARNING,
                    scope=ConsistencyScope.LOCAL,
                    description=f"Trinity vector for '{concept}' is highly imbalanced: {vector}",
                    involved_concepts=[concept],
                    evidence={"vector": vector, "imbalance": max(vector) - min(vector)},
                    suggested_resolution="Consider balancing Trinity dimensions",
                )
                violations.append(violation)

        return violations

    def _check_relational_coherence(
        self,
        trinity_vectors: Dict[str, Tuple[float, float, float]],
        relations: List[Any],
    ) -> List[ConsistencyViolation]:
        """Check coherence between related concepts' Trinity vectors"""
        violations = []

        for relation in relations:
            if (
                hasattr(relation, "source_concept")
                and hasattr(relation, "target_concept")
                and relation.source_concept in trinity_vectors
                and relation.target_concept in trinity_vectors
            ):

                source_vec = np.array(trinity_vectors[relation.source_concept])
                target_vec = np.array(trinity_vectors[relation.target_concept])

                # Calculate coherence
                coherence = self._calculate_trinity_coherence(source_vec, target_vec)

                # Check if coherence matches relation strength
                if hasattr(relation, "strength"):
                    expected_coherence = relation.strength

                    if (
                        abs(coherence - expected_coherence) > 0.4
                    ):  # Significant mismatch
                        violation = ConsistencyViolation(
                            violation_type=ConsistencyViolationType.TRINITY_VECTOR_INCOHERENCE,
                            severity=ViolationSeverity.MODERATE,
                            scope=ConsistencyScope.LOCAL,
                            description=f"Trinity coherence mismatch for relation {relation.source_concept}→{relation.target_concept}",
                            involved_concepts=[
                                relation.source_concept,
                                relation.target_concept,
                            ],
                            evidence={
                                "calculated_coherence": coherence,
                                "relation_strength": expected_coherence,
                                "mismatch": abs(coherence - expected_coherence),
                            },
                            suggested_resolution="Align Trinity vectors with relation strength",
                        )
                        violations.append(violation)

        return violations

    def _check_theological_constraints(
        self, trinity_vectors: Dict[str, Tuple[float, float, float]]
    ) -> List[ConsistencyViolation]:
        """Check Trinity-specific theological constraints"""
        violations = []

        # Check for divine concepts
        divine_concepts = [
            concept
            for concept in trinity_vectors.keys()
            if "divine" in concept.lower() or "god" in concept.lower()
        ]

        for concept in divine_concepts:
            vector = trinity_vectors[concept]

            # Divine concepts should have high Trinity values
            if any(v < 0.7 for v in vector):
                violation = ConsistencyViolation(
                    violation_type=ConsistencyViolationType.TRINITY_VECTOR_INCOHERENCE,
                    severity=ViolationSeverity.MODERATE,
                    scope=ConsistencyScope.LOCAL,
                    description=f"Divine concept '{concept}' has unexpectedly low Trinity dimensions: {vector}",
                    involved_concepts=[concept],
                    evidence={"vector": vector},
                    suggested_resolution="Divine concepts should have high E, G, T values",
                )
                violations.append(violation)

        return violations

    def _calculate_trinity_coherence(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Trinity coherence between two vectors"""
        # Cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)


class ModalConsistencyAnalyzer:
    """Modal logic consistency analysis"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def check_modal_consistency(
        self, modal_statements: Dict[str, Dict[str, Any]], relations: List[Any]
    ) -> List[ConsistencyViolation]:
        """Check modal logic consistency"""

        violations = []

        # Check modal axiom consistency
        axiom_violations = self._check_modal_axioms(modal_statements)
        violations.extend(axiom_violations)

        # Check necessity/possibility consistency
        necessity_violations = self._check_necessity_consistency(modal_statements)
        violations.extend(necessity_violations)

        # Check accessibility relation consistency
        accessibility_violations = self._check_accessibility_consistency(relations)
        violations.extend(accessibility_violations)

        self.logger.debug(f"Found {len(violations)} modal consistency violations")
        return violations

    def _check_modal_axioms(
        self, modal_statements: Dict[str, Dict[str, Any]]
    ) -> List[ConsistencyViolation]:
        """Check consistency with modal logic axioms"""
        violations = []

        for concept, modal_props in modal_statements.items():
            # Check K axiom: □(p → q) → (□p → □q)
            # If something necessarily implies q, and p is necessary, then q is necessary

            if modal_props.get("necessary", False) and modal_props.get(
                "impossible", False
            ):
                violation = ConsistencyViolation(
                    violation_type=ConsistencyViolationType.MODAL_INCONSISTENCY,
                    severity=ViolationSeverity.CRITICAL,
                    scope=ConsistencyScope.LOCAL,
                    description=f"Concept '{concept}' cannot be both necessary and impossible",
                    involved_concepts=[concept],
                    evidence={"necessary": True, "impossible": True},
                    suggested_resolution="Resolve modal status contradiction",
                )
                violations.append(violation)

        return violations

    def _check_necessity_consistency(
        self, modal_statements: Dict[str, Dict[str, Any]]
    ) -> List[ConsistencyViolation]:
        """Check necessity/possibility consistency"""
        violations = []

        for concept, modal_props in modal_statements.items():
            # If something is necessary, it must also be possible
            if modal_props.get("necessary", False) and not modal_props.get(
                "possible", True
            ):
                violation = ConsistencyViolation(
                    violation_type=ConsistencyViolationType.MODAL_INCONSISTENCY,
                    severity=ViolationSeverity.MAJOR,
                    scope=ConsistencyScope.LOCAL,
                    description=f"Necessary concept '{concept}' must also be possible",
                    involved_concepts=[concept],
                    evidence={"necessary": True, "possible": False},
                    suggested_resolution="Necessary concepts are necessarily possible",
                )
                violations.append(violation)

        return violations

    def _check_accessibility_consistency(
        self, relations: List[Any]
    ) -> List[ConsistencyViolation]:
        """Check accessibility relation consistency"""
        violations = []

        # This is a simplified check - full modal logic would require more sophisticated analysis
        modal_relations = [
            r
            for r in relations
            if hasattr(r, "relation_type") and "modal" in r.relation_type.value
        ]

        # Check for inconsistent modal relations
        # (Implementation would depend on specific modal system)

        return violations


class PXLConsistencyChecker:
    """Main PXL consistency checking engine"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize analyzers
        self.logical_analyzer = LogicalConsistencyAnalyzer()
        self.trinity_analyzer = TrinityConsistencyAnalyzer()
        self.modal_analyzer = ModalConsistencyAnalyzer()

        # Configuration
        self.severity_weights = {
            ViolationSeverity.CRITICAL: 1.0,
            ViolationSeverity.MAJOR: 0.8,
            ViolationSeverity.MODERATE: 0.5,
            ViolationSeverity.MINOR: 0.2,
            ViolationSeverity.WARNING: 0.1,
        }

        # Caching
        self._consistency_cache: Dict[str, ConsistencyReport] = {}

        self.logger.info("PXL consistency checker initialized")

    def check_consistency(
        self,
        relations: List[Any],
        trinity_vectors: Optional[Dict[str, Tuple[float, float, float]]] = None,
        concept_properties: Optional[Dict[str, Dict[str, Any]]] = None,
        modal_statements: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> ConsistencyReport:
        """
        Comprehensive PXL consistency checking

        Args:
            relations: List of PXL relations to check
            trinity_vectors: Trinity vector representations
            concept_properties: Properties of concepts
            modal_statements: Modal logic statements

        Returns:
            ConsistencyReport: Comprehensive consistency analysis
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(
                relations, trinity_vectors, concept_properties, modal_statements
            )
            if cache_key in self._consistency_cache:
                return self._consistency_cache[cache_key]

            all_violations = []

            # Logical consistency analysis
            logical_violations = self.logical_analyzer.check_logical_consistency(
                relations, concept_properties or {}
            )
            all_violations.extend(logical_violations)
            self.logger.debug(f"Found {len(logical_violations)} logical violations")

            # Trinity consistency analysis
            if trinity_vectors:
                trinity_violations = self.trinity_analyzer.check_trinity_consistency(
                    trinity_vectors, relations
                )
                all_violations.extend(trinity_violations)
                self.logger.debug(f"Found {len(trinity_violations)} Trinity violations")

            # Modal consistency analysis
            if modal_statements:
                modal_violations = self.modal_analyzer.check_modal_consistency(
                    modal_statements, relations
                )
                all_violations.extend(modal_violations)
                self.logger.debug(f"Found {len(modal_violations)} modal violations")

            # Organize violations
            violations_by_type = self._organize_violations_by_type(all_violations)
            violations_by_severity = self._organize_violations_by_severity(
                all_violations
            )

            # Identify critical violations
            critical_violations = violations_by_severity.get(
                ViolationSeverity.CRITICAL, []
            )

            # Calculate consistency scores
            global_score = self._calculate_global_consistency_score(
                all_violations, len(relations)
            )
            local_scores = self._calculate_local_consistency_scores(all_violations)
            cluster_scores = self._calculate_cluster_consistency_scores(all_violations)

            # Generate resolution recommendations
            recommendations = self._generate_resolution_recommendations(all_violations)

            # Calculate consistency metrics
            metrics = self._calculate_consistency_metrics(all_violations, relations)

            # Analysis metadata
            analysis_metadata = {
                "total_relations_analyzed": len(relations),
                "concepts_with_trinity_vectors": (
                    len(trinity_vectors) if trinity_vectors else 0
                ),
                "concepts_with_properties": (
                    len(concept_properties) if concept_properties else 0
                ),
                "modal_statements_analyzed": (
                    len(modal_statements) if modal_statements else 0
                ),
                "processing_timestamp": time.time(),
            }

            report = ConsistencyReport(
                total_violations=len(all_violations),
                violations_by_type=violations_by_type,
                violations_by_severity=violations_by_severity,
                critical_violations=critical_violations,
                global_consistency_score=global_score,
                local_consistency_scores=local_scores,
                cluster_consistency_scores=cluster_scores,
                resolution_recommendations=recommendations,
                consistency_metrics=metrics,
                analysis_metadata=analysis_metadata,
            )

            # Cache report
            self._consistency_cache[cache_key] = report

            self.logger.info(
                f"Consistency check completed: {len(all_violations)} violations found"
            )
            return report

        except Exception as e:
            self.logger.error(f"Consistency checking failed: {e}")
            raise

    def _organize_violations_by_type(
        self, violations: List[ConsistencyViolation]
    ) -> Dict[ConsistencyViolationType, List[ConsistencyViolation]]:
        """Organize violations by type"""

        by_type = defaultdict(list)
        for violation in violations:
            by_type[violation.violation_type].append(violation)

        return dict(by_type)

    def _organize_violations_by_severity(
        self, violations: List[ConsistencyViolation]
    ) -> Dict[ViolationSeverity, List[ConsistencyViolation]]:
        """Organize violations by severity"""

        by_severity = defaultdict(list)
        for violation in violations:
            by_severity[violation.severity].append(violation)

        # Sort within each severity level
        for severity in by_severity:
            by_severity[severity].sort(key=lambda v: v.confidence, reverse=True)

        return dict(by_severity)

    def _calculate_global_consistency_score(
        self, violations: List[ConsistencyViolation], total_relations: int
    ) -> float:
        """Calculate global consistency score (0-1, higher is better)"""

        if total_relations == 0:
            return 1.0

        # Calculate weighted violation penalty
        total_penalty = 0.0
        for violation in violations:
            penalty = self.severity_weights[violation.severity] * violation.confidence
            total_penalty += penalty

        # Normalize by total relations
        normalized_penalty = total_penalty / max(total_relations, 1)

        # Convert to consistency score (1 - penalty, but ensure non-negative)
        consistency_score = max(0.0, 1.0 - normalized_penalty)

        return consistency_score

    def _calculate_local_consistency_scores(
        self, violations: List[ConsistencyViolation]
    ) -> Dict[str, float]:
        """Calculate consistency scores for individual concepts"""

        concept_violations = defaultdict(list)

        # Group violations by concept
        for violation in violations:
            for concept in violation.involved_concepts:
                concept_violations[concept].append(violation)

        # Calculate scores
        scores = {}
        for concept, concept_viols in concept_violations.items():
            penalty = sum(
                self.severity_weights[v.severity] * v.confidence for v in concept_viols
            )
            # Normalize by number of violations + 1 to prevent division by zero
            normalized_penalty = penalty / (len(concept_viols) + 1)
            scores[concept] = max(0.0, 1.0 - normalized_penalty)

        return scores

    def _calculate_cluster_consistency_scores(
        self, violations: List[ConsistencyViolation]
    ) -> Dict[str, float]:
        """Calculate consistency scores for concept clusters"""

        cluster_violations = defaultdict(list)

        # Group violations by scope (cluster-level violations)
        for violation in violations:
            if violation.scope == ConsistencyScope.CLUSTER:
                cluster_key = "_".join(sorted(violation.involved_concepts))
                cluster_violations[cluster_key].append(violation)

        # Calculate scores
        scores = {}
        for cluster, cluster_viols in cluster_violations.items():
            penalty = sum(
                self.severity_weights[v.severity] * v.confidence for v in cluster_viols
            )
            normalized_penalty = penalty / (len(cluster_viols) + 1)
            scores[cluster] = max(0.0, 1.0 - normalized_penalty)

        return scores

    def _generate_resolution_recommendations(
        self, violations: List[ConsistencyViolation]
    ) -> List[str]:
        """Generate prioritized resolution recommendations"""

        recommendations = []

        # Prioritize critical violations
        critical_violations = [
            v for v in violations if v.severity == ViolationSeverity.CRITICAL
        ]

        for violation in critical_violations[:5]:  # Top 5 critical
            if violation.suggested_resolution:
                recommendations.append(f"CRITICAL: {violation.suggested_resolution}")

        # Add major violations
        major_violations = [
            v for v in violations if v.severity == ViolationSeverity.MAJOR
        ]

        for violation in major_violations[:3]:  # Top 3 major
            if violation.suggested_resolution:
                recommendations.append(f"MAJOR: {violation.suggested_resolution}")

        # General recommendations
        if len(violations) > 10:
            recommendations.append(
                "Consider systematic review of concept definitions and relations"
            )

        if any(
            v.violation_type == ConsistencyViolationType.TRINITY_VECTOR_INCOHERENCE
            for v in violations
        ):
            recommendations.append(
                "Review Trinity vector assignments for theological accuracy"
            )

        if any(
            v.violation_type == ConsistencyViolationType.MODAL_INCONSISTENCY
            for v in violations
        ):
            recommendations.append(
                "Check modal logic statements for consistency with axioms"
            )

        return recommendations

    def _calculate_consistency_metrics(
        self, violations: List[ConsistencyViolation], relations: List[Any]
    ) -> Dict[str, float]:
        """Calculate detailed consistency metrics"""

        total_violations = len(violations)
        total_relations = len(relations)

        return {
            "violation_rate": total_violations / max(total_relations, 1),
            "critical_violation_rate": len(
                [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
            )
            / max(total_violations, 1),
            "avg_violation_confidence": sum(v.confidence for v in violations)
            / max(total_violations, 1),
            "logical_violation_ratio": len(
                [
                    v
                    for v in violations
                    if v.violation_type
                    == ConsistencyViolationType.LOGICAL_CONTRADICTION
                ]
            )
            / max(total_violations, 1),
            "trinity_violation_ratio": len(
                [
                    v
                    for v in violations
                    if v.violation_type
                    == ConsistencyViolationType.TRINITY_VECTOR_INCOHERENCE
                ]
            )
            / max(total_violations, 1),
            "modal_violation_ratio": len(
                [
                    v
                    for v in violations
                    if v.violation_type == ConsistencyViolationType.MODAL_INCONSISTENCY
                ]
            )
            / max(total_violations, 1),
        }

    def _generate_cache_key(
        self,
        relations: List[Any],
        trinity_vectors: Optional[Dict[str, Tuple[float, float, float]]],
        concept_properties: Optional[Dict[str, Dict[str, Any]]],
        modal_statements: Optional[Dict[str, Dict[str, Any]]],
    ) -> str:
        """Generate cache key for consistency checking"""

        import hashlib

        key_data = {
            "relations_count": len(relations),
            "trinity_concepts": (
                sorted(trinity_vectors.keys()) if trinity_vectors else []
            ),
            "property_concepts": (
                sorted(concept_properties.keys()) if concept_properties else []
            ),
            "modal_concepts": (
                sorted(modal_statements.keys()) if modal_statements else []
            ),
        }

        key_string = str(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()


# Global PXL consistency checker instance
pxl_consistency_checker = PXLConsistencyChecker()


__all__ = [
    "ConsistencyViolationType",
    "ViolationSeverity",
    "ConsistencyScope",
    "ConsistencyViolation",
    "ConsistencyReport",
    "LogicalConsistencyAnalyzer",
    "TrinityConsistencyAnalyzer",
    "ModalConsistencyAnalyzer",
    "PXLConsistencyChecker",
    "pxl_consistency_checker",
]
