"""
PXL Schema Definitions - UIP Step 2 Component
============================================

Protocol definitions and data structures for PXL (Philosophically Extended Logic) components.
Defines interfaces, validation rules, and data contracts for relation mapping, consistency checking,
and postprocessing integration with V2 framework protocols.

Integrates with: All PXL components, V2 framework protocols, Trinity systems, modal logic systems
Dependencies: Pydantic for validation, typing protocols, dataclasses for structured data
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
from protocols.shared.system_imports import *
from pydantic import BaseModel, Field, validator

# ========================= CORE ENUMERATIONS =========================


class PXLRelationType(Enum):
    """Types of PXL relations"""

    # Logical relations
    LOGICAL_IMPLICATION = "logical_implication"
    LOGICAL_EQUIVALENCE = "logical_equivalence"
    LOGICAL_CONTRADICTION = "logical_contradiction"
    DEFINITIONAL_RELATION = "definitional_relation"

    # Trinity-based relations
    TRINITY_COHERENCE = "trinity_coherence"
    TRINITY_COMPLEMENTARITY = "trinity_complementarity"
    ESSENCE_RELATION = "essence_relation"
    GENERATION_RELATION = "generation_relation"
    TEMPORAL_RELATION = "temporal_relation"

    # Semantic relations
    SEMANTIC_SIMILARITY = "semantic_similarity"
    SEMANTIC_OPPOSITION = "semantic_opposition"
    HIERARCHICAL_RELATION = "hierarchical_relation"
    CAUSAL_RELATION = "causal_relation"

    # Modal relations
    NECESSITY_RELATION = "necessity_relation"
    POSSIBILITY_RELATION = "possibility_relation"
    CONTINGENCY_RELATION = "contingency_relation"

    # Ontological relations
    SUBSUMPTION_RELATION = "subsumption_relation"
    INSTANTIATION_RELATION = "instantiation_relation"
    COMPOSITION_RELATION = "composition_relation"


class PXLConsistencyLevel(Enum):
    """Levels of PXL consistency"""

    CRITICAL_VIOLATION = "critical_violation"
    MAJOR_INCONSISTENCY = "major_inconsistency"
    MODERATE_TENSION = "moderate_tension"
    MINOR_DISCREPANCY = "minor_discrepancy"
    FULLY_CONSISTENT = "fully_consistent"


class PXLAnalysisScope(Enum):
    """Scope of PXL analysis"""

    LOCAL_CONCEPT = "local_concept"
    CONCEPT_CLUSTER = "concept_cluster"
    SEMANTIC_NETWORK = "semantic_network"
    GLOBAL_SYSTEM = "global_system"


class TrinityDimension(Enum):
    """Trinity vector dimensions"""

    ESSENCE = "essence"  # E - Essential/Being dimension
    GENERATION = "generation"  # G - Generative/Becoming dimension
    TEMPORAL = "temporal"  # T - Temporal/Process dimension


class ModalOperator(Enum):
    """Modal logic operators"""

    NECESSITY = "necessity"  # □ (box)
    POSSIBILITY = "possibility"  # ◊ (diamond)
    CONTINGENCY = "contingency"  # Neither necessary nor impossible
    IMPOSSIBILITY = "impossibility"  # ¬◊ (not possible)


class ValidationSeverity(Enum):
    """Severity levels for validation"""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ========================= BASE PROTOCOLS =========================


@runtime_checkable
class PXLValidatable(Protocol):
    """Protocol for validatable PXL objects"""

    def validate(self) -> "ValidationResult":
        """Validate the object and return validation result"""
        ...

    def is_valid(self) -> bool:
        """Check if object is valid"""
        ...


@runtime_checkable
class PXLAnalyzable(Protocol):
    """Protocol for analyzable PXL objects"""

    def analyze(self) -> "PXLAnalysisResult":
        """Perform analysis and return results"""
        ...

    def get_analysis_metadata(self) -> Dict[str, Any]:
        """Get analysis metadata"""
        ...


@runtime_checkable
class TrinityVectorizable(Protocol):
    """Protocol for objects that can be Trinity vectorized"""

    def get_trinity_vector(self) -> Tuple[float, float, float]:
        """Get Trinity vector (E, G, T)"""
        ...

    def set_trinity_vector(self, vector: Tuple[float, float, float]) -> None:
        """Set Trinity vector"""
        ...


@runtime_checkable
class ModalAnalyzable(Protocol):
    """Protocol for modal logic analyzable objects"""

    def get_modal_properties(self) -> Dict[ModalOperator, bool]:
        """Get modal properties"""
        ...

    def check_modal_consistency(self) -> "ModalConsistencyResult":
        """Check modal consistency"""
        ...


# ========================= CORE DATA STRUCTURES =========================


@dataclass
class TrinityVector:
    """Trinity vector representation"""

    essence: float = field(default=0.0)
    generation: float = field(default=0.0)
    temporal: float = field(default=0.0)

    def __post_init__(self):
        """Validate Trinity vector"""
        self._validate_bounds()

    def _validate_bounds(self):
        """Ensure all dimensions are in [0,1] range"""
        for dim_name, value in [
            ("essence", self.essence),
            ("generation", self.generation),
            ("temporal", self.temporal),
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Trinity dimension '{dim_name}' must be in [0,1], got {value}"
                )

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple"""
        return (self.essence, self.generation, self.temporal)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.essence, self.generation, self.temporal])

    def magnitude(self) -> float:
        """Calculate vector magnitude"""
        return np.linalg.norm(self.to_array())

    def normalized(self) -> "TrinityVector":
        """Return normalized Trinity vector"""
        mag = self.magnitude()
        if mag == 0:
            return TrinityVector(0, 0, 0)

        arr = self.to_array() / mag
        return TrinityVector(arr[0], arr[1], arr[2])

    def coherence_with(self, other: "TrinityVector") -> float:
        """Calculate coherence with another Trinity vector"""
        if self.magnitude() == 0 or other.magnitude() == 0:
            return 0.0

        return np.dot(self.to_array(), other.to_array()) / (
            self.magnitude() * other.magnitude()
        )

    def __str__(self) -> str:
        return f"Trinity(E={self.essence:.3f}, G={self.generation:.3f}, T={self.temporal:.3f})"


@dataclass
class ModalProperties:
    """Modal logic properties"""

    necessary: bool = False
    possible: bool = True
    contingent: bool = True
    impossible: bool = False

    def __post_init__(self):
        """Validate modal properties"""
        self._validate_modal_consistency()

    def _validate_modal_consistency(self):
        """Check consistency of modal properties"""
        # Necessary implies possible
        if self.necessary and not self.possible:
            raise ValueError("Necessary statements must be possible")

        # Impossible implies not possible
        if self.impossible and self.possible:
            raise ValueError("Impossible statements cannot be possible")

        # Cannot be both necessary and impossible
        if self.necessary and self.impossible:
            raise ValueError("Cannot be both necessary and impossible")

        # If necessary, not contingent
        if self.necessary:
            self.contingent = False

        # If impossible, not contingent
        if self.impossible:
            self.contingent = False

    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary"""
        return {
            "necessary": self.necessary,
            "possible": self.possible,
            "contingent": self.contingent,
            "impossible": self.impossible,
        }


@dataclass
class ValidationIssue:
    """Individual validation issue"""

    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    code: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        prefix = f"[{self.severity.value.upper()}]"
        field_info = f" {self.field}:" if self.field else ""
        return f"{prefix}{field_info} {self.message}"


@dataclass
class ValidationResult:
    """Result of validation operation"""

    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def add_issue(
        self,
        severity: ValidationSeverity,
        message: str,
        field: Optional[str] = None,
        code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Add validation issue"""
        issue = ValidationIssue(
            severity=severity,
            message=message,
            field=field,
            code=code,
            context=context or {},
        )
        self.issues.append(issue)

        # Mark as invalid if error severity
        if severity == ValidationSeverity.ERROR:
            self.valid = False

    def has_errors(self) -> bool:
        """Check if has error-level issues"""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)

    def has_warnings(self) -> bool:
        """Check if has warning-level issues"""
        return any(
            issue.severity == ValidationSeverity.WARNING for issue in self.issues
        )

    def get_summary(self) -> str:
        """Get validation summary"""
        if self.valid:
            return "Valid"

        error_count = len(
            [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
        )
        warning_count = len(
            [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
        )

        return f"Invalid: {error_count} errors, {warning_count} warnings"


# ========================= PXL RELATION SCHEMA =========================


class PXLRelationModel(BaseModel):
    """Pydantic model for PXL relations"""

    source_concept: str = Field(
        ..., min_length=1, description="Source concept identifier"
    )
    target_concept: str = Field(
        ..., min_length=1, description="Target concept identifier"
    )
    relation_type: PXLRelationType = Field(..., description="Type of PXL relation")
    strength: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Relation strength [0,1]"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in relation [0,1]"
    )

    trinity_coherence: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Trinity coherence [-1,1]"
    )
    modal_necessity: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Modal necessity [0,1]"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        use_enum_values = True

    @validator("source_concept", "target_concept")
    def validate_concepts(cls, v):
        """Validate concept identifiers"""
        if not v.strip():
            raise ValueError("Concept identifiers cannot be empty")
        return v.strip()

    @validator("strength", "confidence")
    def validate_probabilities(cls, v):
        """Validate probability values"""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Probability values must be in [0,1]")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.dict()


@dataclass
class PXLRelation:
    """Core PXL relation structure"""

    source_concept: str
    target_concept: str
    relation_type: PXLRelationType
    strength: float = 1.0
    confidence: float = 1.0
    trinity_coherence: Optional[float] = None
    modal_necessity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> ValidationResult:
        """Validate PXL relation"""
        result = ValidationResult(valid=True)

        # Check concept identifiers
        if not self.source_concept.strip():
            result.add_issue(
                ValidationSeverity.ERROR,
                "Source concept cannot be empty",
                "source_concept",
            )

        if not self.target_concept.strip():
            result.add_issue(
                ValidationSeverity.ERROR,
                "Target concept cannot be empty",
                "target_concept",
            )

        # Check strength and confidence bounds
        if not (0.0 <= self.strength <= 1.0):
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Strength must be in [0,1], got {self.strength}",
                "strength",
            )

        if not (0.0 <= self.confidence <= 1.0):
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Confidence must be in [0,1], got {self.confidence}",
                "confidence",
            )

        # Check optional fields
        if self.trinity_coherence is not None and not (
            -1.0 <= self.trinity_coherence <= 1.0
        ):
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Trinity coherence must be in [-1,1], got {self.trinity_coherence}",
                "trinity_coherence",
            )

        if self.modal_necessity is not None and not (
            0.0 <= self.modal_necessity <= 1.0
        ):
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Modal necessity must be in [0,1], got {self.modal_necessity}",
                "modal_necessity",
            )

        return result

    def is_valid(self) -> bool:
        """Check if relation is valid"""
        return self.validate().valid

    def to_model(self) -> PXLRelationModel:
        """Convert to Pydantic model"""
        return PXLRelationModel(
            source_concept=self.source_concept,
            target_concept=self.target_concept,
            relation_type=self.relation_type,
            strength=self.strength,
            confidence=self.confidence,
            trinity_coherence=self.trinity_coherence,
            modal_necessity=self.modal_necessity,
            metadata=self.metadata,
        )


# ========================= PXL CONSISTENCY SCHEMA =========================


@dataclass
class ConsistencyViolation:
    """PXL consistency violation"""

    violation_type: str
    severity: PXLConsistencyLevel
    scope: PXLAnalysisScope
    description: str
    involved_concepts: List[str]
    involved_relations: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_resolution: Optional[str] = None
    confidence: float = 1.0

    def validate(self) -> ValidationResult:
        """Validate consistency violation"""
        result = ValidationResult(valid=True)

        if not self.description.strip():
            result.add_issue(
                ValidationSeverity.ERROR, "Description cannot be empty", "description"
            )

        if not self.involved_concepts:
            result.add_issue(
                ValidationSeverity.ERROR,
                "Must have at least one involved concept",
                "involved_concepts",
            )

        if not (0.0 <= self.confidence <= 1.0):
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Confidence must be in [0,1], got {self.confidence}",
                "confidence",
            )

        return result


@dataclass
class ConsistencyReport:
    """PXL consistency analysis report"""

    total_violations: int
    violations_by_type: Dict[str, List[ConsistencyViolation]]
    violations_by_severity: Dict[PXLConsistencyLevel, List[ConsistencyViolation]]
    global_consistency_score: float
    local_consistency_scores: Dict[str, float]
    resolution_recommendations: List[str]
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> ValidationResult:
        """Validate consistency report"""
        result = ValidationResult(valid=True)

        if self.total_violations < 0:
            result.add_issue(
                ValidationSeverity.ERROR,
                "Total violations cannot be negative",
                "total_violations",
            )

        if not (0.0 <= self.global_consistency_score <= 1.0):
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Global consistency score must be in [0,1], got {self.global_consistency_score}",
                "global_consistency_score",
            )

        # Check local scores
        for concept, score in self.local_consistency_scores.items():
            if not (0.0 <= score <= 1.0):
                result.add_issue(
                    ValidationSeverity.ERROR,
                    f"Local consistency score for '{concept}' must be in [0,1], got {score}",
                    "local_consistency_scores",
                )

        return result


# ========================= PXL ANALYSIS SCHEMA =========================


@dataclass
class PXLAnalysisConfig:
    """Configuration for PXL analysis"""

    enable_trinity_analysis: bool = True
    enable_modal_analysis: bool = True
    enable_consistency_checking: bool = True
    enable_semantic_clustering: bool = True

    trinity_coherence_threshold: float = 0.3
    consistency_threshold: float = 0.7
    relation_strength_threshold: float = 0.1

    max_analysis_depth: int = 10
    max_cluster_size: int = 50

    cache_results: bool = True
    detailed_logging: bool = False

    def validate(self) -> ValidationResult:
        """Validate analysis configuration"""
        result = ValidationResult(valid=True)

        # Check thresholds
        for threshold_name, threshold_value in [
            ("trinity_coherence_threshold", self.trinity_coherence_threshold),
            ("consistency_threshold", self.consistency_threshold),
            ("relation_strength_threshold", self.relation_strength_threshold),
        ]:
            if not (0.0 <= threshold_value <= 1.0):
                result.add_issue(
                    ValidationSeverity.ERROR,
                    f"{threshold_name} must be in [0,1], got {threshold_value}",
                    threshold_name,
                )

        # Check limits
        if self.max_analysis_depth <= 0:
            result.add_issue(
                ValidationSeverity.ERROR,
                "Max analysis depth must be positive",
                "max_analysis_depth",
            )

        if self.max_cluster_size <= 0:
            result.add_issue(
                ValidationSeverity.ERROR,
                "Max cluster size must be positive",
                "max_cluster_size",
            )

        return result


@dataclass
class PXLAnalysisResult:
    """Result of PXL analysis"""

    relations: List[PXLRelation]
    consistency_report: ConsistencyReport
    trinity_analysis: Dict[str, TrinityVector]
    modal_analysis: Dict[str, ModalProperties]
    semantic_clusters: Dict[str, List[str]]
    analysis_config: PXLAnalysisConfig
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> ValidationResult:
        """Validate analysis result"""
        result = ValidationResult(valid=True)

        # Validate all relations
        for i, relation in enumerate(self.relations):
            relation_result = relation.validate()
            if not relation_result.valid:
                for issue in relation_result.issues:
                    result.add_issue(
                        issue.severity,
                        f"Relation {i}: {issue.message}",
                        (
                            f"relations[{i}].{issue.field}"
                            if issue.field
                            else f"relations[{i}]"
                        ),
                    )

        # Validate consistency report
        consistency_result = self.consistency_report.validate()
        if not consistency_result.valid:
            for issue in consistency_result.issues:
                result.add_issue(
                    issue.severity,
                    f"Consistency report: {issue.message}",
                    (
                        f"consistency_report.{issue.field}"
                        if issue.field
                        else "consistency_report"
                    ),
                )

        # Validate config
        config_result = self.analysis_config.validate()
        if not config_result.valid:
            for issue in config_result.issues:
                result.add_issue(
                    issue.severity,
                    f"Analysis config: {issue.message}",
                    (
                        f"analysis_config.{issue.field}"
                        if issue.field
                        else "analysis_config"
                    ),
                )

        return result


# ========================= PXL PROCESSING INTERFACES =========================


@runtime_checkable
class PXLRelationMapper(Protocol):
    """Protocol for PXL relation mapping"""

    def map_relations(
        self, concepts: List[str], context: Dict[str, Any]
    ) -> List[PXLRelation]:
        """Map relations between concepts"""
        ...

    def analyze_trinity_coherence(
        self, relations: List[PXLRelation]
    ) -> Dict[str, float]:
        """Analyze Trinity coherence of relations"""
        ...


@runtime_checkable
class PXLConsistencyChecker(Protocol):
    """Protocol for PXL consistency checking"""

    def check_consistency(
        self,
        relations: List[PXLRelation],
        trinity_vectors: Optional[Dict[str, TrinityVector]] = None,
    ) -> ConsistencyReport:
        """Check consistency of PXL relations"""
        ...

    def validate_relations(self, relations: List[PXLRelation]) -> ValidationResult:
        """Validate list of relations"""
        ...


@runtime_checkable
class PXLPostprocessor(Protocol):
    """Protocol for PXL postprocessing"""

    def process_analysis_result(self, result: PXLAnalysisResult) -> Dict[str, Any]:
        """Process analysis result for output"""
        ...

    def generate_report(
        self, result: PXLAnalysisResult, format_type: str = "json"
    ) -> str:
        """Generate formatted report"""
        ...


# ========================= UTILITY FUNCTIONS =========================


def validate_trinity_vector(vector: Tuple[float, float, float]) -> ValidationResult:
    """Validate Trinity vector tuple"""
    result = ValidationResult(valid=True)

    if len(vector) != 3:
        result.add_issue(
            ValidationSeverity.ERROR,
            f"Trinity vector must have 3 dimensions, got {len(vector)}",
        )
        return result

    for i, value in enumerate(vector):
        if not isinstance(value, (int, float)):
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Trinity dimension {i} must be numeric, got {type(value)}",
            )
        elif not (0.0 <= value <= 1.0):
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Trinity dimension {i} must be in [0,1], got {value}",
            )

    return result


def create_default_analysis_config() -> PXLAnalysisConfig:
    """Create default analysis configuration"""
    return PXLAnalysisConfig()


def merge_validation_results(results: List[ValidationResult]) -> ValidationResult:
    """Merge multiple validation results"""
    merged = ValidationResult(valid=True)

    for result in results:
        merged.issues.extend(result.issues)
        if not result.valid:
            merged.valid = False

    return merged


def relation_to_json(relation: PXLRelation) -> str:
    """Convert PXL relation to JSON"""
    model = relation.to_model()
    return model.json()


def relation_from_json(json_str: str) -> PXLRelation:
    """Create PXL relation from JSON"""
    model = PXLRelationModel.parse_raw(json_str)
    return PXLRelation(
        source_concept=model.source_concept,
        target_concept=model.target_concept,
        relation_type=model.relation_type,
        strength=model.strength,
        confidence=model.confidence,
        trinity_coherence=model.trinity_coherence,
        modal_necessity=model.modal_necessity,
        metadata=model.metadata,
    )


# ========================= SCHEMA VALIDATORS =========================


class PXLSchemaValidator:
    """Comprehensive PXL schema validation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_relation_set(self, relations: List[PXLRelation]) -> ValidationResult:
        """Validate a set of PXL relations"""
        result = ValidationResult(valid=True)

        # Validate individual relations
        for i, relation in enumerate(relations):
            relation_result = relation.validate()
            if not relation_result.valid:
                for issue in relation_result.issues:
                    result.add_issue(
                        issue.severity,
                        f"Relation {i}: {issue.message}",
                        f"relations[{i}]",
                    )

        # Check for duplicate relations
        seen_relations = set()
        for i, relation in enumerate(relations):
            relation_key = (
                relation.source_concept,
                relation.target_concept,
                relation.relation_type.value,
            )
            if relation_key in seen_relations:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Duplicate relation at index {i}: {relation_key}",
                    f"relations[{i}]",
                )
            seen_relations.add(relation_key)

        return result

    def validate_trinity_vector_set(
        self, vectors: Dict[str, TrinityVector]
    ) -> ValidationResult:
        """Validate a set of Trinity vectors"""
        result = ValidationResult(valid=True)

        for concept, vector in vectors.items():
            try:
                # Trinity vector validates itself in __post_init__
                vector._validate_bounds()
            except ValueError as e:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    f"Invalid Trinity vector for '{concept}': {e}",
                    concept,
                )

        return result

    def validate_analysis_result(
        self, analysis_result: PXLAnalysisResult
    ) -> ValidationResult:
        """Validate complete analysis result"""
        return analysis_result.validate()


# Global schema validator instance
pxl_schema_validator = PXLSchemaValidator()


__all__ = [
    # Enums
    "PXLRelationType",
    "PXLConsistencyLevel",
    "PXLAnalysisScope",
    "TrinityDimension",
    "ModalOperator",
    "ValidationSeverity",
    # Protocols
    "PXLValidatable",
    "PXLAnalyzable",
    "TrinityVectorizable",
    "ModalAnalyzable",
    "PXLRelationMapper",
    "PXLConsistencyChecker",
    "PXLPostprocessor",
    # Core data structures
    "TrinityVector",
    "ModalProperties",
    "ValidationIssue",
    "ValidationResult",
    # Relation schema
    "PXLRelationModel",
    "PXLRelation",
    # Consistency schema
    "ConsistencyViolation",
    "ConsistencyReport",
    # Analysis schema
    "PXLAnalysisConfig",
    "PXLAnalysisResult",
    # Utility functions
    "validate_trinity_vector",
    "create_default_analysis_config",
    "merge_validation_results",
    "relation_to_json",
    "relation_from_json",
    # Validators
    "PXLSchemaValidator",
    "pxl_schema_validator",
]
