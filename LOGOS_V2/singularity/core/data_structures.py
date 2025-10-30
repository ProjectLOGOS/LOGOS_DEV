"""
Core Data Structures for Singularity AGI System
===============================================

Defines the fundamental data structures used throughout the Singularity system,
integrating with existing LOGOS V2 Trinity vector mathematics and PXL core.

Key Structures:
- MVSCoordinate: Fractal coordinates in Modal Vector Space
- BDNGenealogy: Banach Data Node genealogy and transformation history  
- ModalInferenceResult: Results from S5 modal logic inference
- CreativeHypothesis: Creative hypothesis with BDN fusion metadata
- NovelProblem: Novel problems discovered through MVS exploration
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import numpy as np
from enum import Enum

# Import existing LOGOS V2 components (maintain existing integrations)
try:
    from intelligence.trinity.trinity_vector_processor import TrinityVector, TrinityVectorAnalysis
except ImportError:
    # Fallback for development/testing
    class TrinityVector:
        def __init__(self, existence=0.5, goodness=0.5, truth=0.5):
            self.existence, self.goodness, self.truth = existence, goodness, truth


class MVSRegionType(Enum):
    """Types of regions in Modal Vector Space"""
    MANDELBROT_SET = "mandelbrot_set"
    JULIA_SET = "julia_set" 
    ESCAPE_REGION = "escape_region"
    BOUNDARY_REGION = "boundary_region"
    CONVERGENT_BASIN = "convergent_basin"
    CHAOTIC_REGION = "chaotic_region"
    UNKNOWN_TERRITORY = "unknown_territory"


class BDNTransformationType(Enum):
    """Types of Banach Data Node transformations"""
    DECOMPOSITION = "banach_decomposition"
    RECOMPOSITION = "banach_recomposition"
    CREATIVE_FUSION = "creative_fusion" 
    CAUSAL_EXTENSION = "causal_extension"
    MODAL_INFERENCE = "modal_inference"
    ORBITAL_PREDICTION = "orbital_prediction"


class NoveltyLevel(Enum):
    """Levels of novelty for generated problems/hypotheses"""
    DERIVATIVE = "derivative"        # Based on existing patterns
    COMBINATORIAL = "combinatorial"  # Novel combination of known elements
    STRUCTURAL = "structural"        # Novel structural relationships  
    PARADIGMATIC = "paradigmatic"    # Novel conceptual paradigm
    TRANSCENDENT = "transcendent"    # Beyond current understanding


@dataclass
class MVSCoordinate:
    """
    Coordinate in Fractal Modal Vector Space
    
    Represents position in infinite-dimensional fractal space with:
    - Complex coordinate for Mandelbrot/Julia set positioning
    - Trinity vector for structural alignment
    - Regional classification and properties
    """
    
    # Core coordinate data
    complex_position: complex
    trinity_vector: Tuple[float, float, float]  # (E, G, T)
    
    # Fractal properties
    region_type: MVSRegionType
    iteration_depth: int
    escape_radius: Optional[float] = None
    convergence_rate: Optional[float] = None
    
    # Metadata
    coordinate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_timestamp: datetime = field(default_factory=datetime.now)
    parent_coordinate_id: Optional[str] = None
    
    # Cached computations
    _orbital_properties_cache: Optional[Dict] = field(default=None, repr=False)
    _stability_cache: Optional[float] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate coordinate data on initialization"""
        # Ensure Trinity vector is normalized
        e, g, t = self.trinity_vector
        if not (0.0 <= e <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= t <= 1.0):
            # Normalize if outside bounds
            max_val = max(e, g, t, 1.0)
            self.trinity_vector = (e/max_val, g/max_val, t/max_val)
    
    def distance_to(self, other: 'MVSCoordinate') -> float:
        """Calculate distance to another coordinate in MVS"""
        # Complex distance
        complex_dist = abs(self.complex_position - other.complex_position)
        
        # Trinity vector distance  
        e1, g1, t1 = self.trinity_vector
        e2, g2, t2 = other.trinity_vector
        trinity_dist = np.sqrt((e1-e2)**2 + (g1-g2)**2 + (t1-t2)**2)
        
        # Combined distance (weighted)
        return np.sqrt(complex_dist**2 + trinity_dist**2)
    
    def to_trinity_vector(self) -> TrinityVector:
        """Convert to LOGOS V2 TrinityVector for integration"""
        e, g, t = self.trinity_vector
        return TrinityVector(existence=e, goodness=g, truth=t)
    
    def get_orbital_properties(self) -> Dict[str, Any]:
        """Get orbital properties (cached computation)"""
        if self._orbital_properties_cache is None:
            self._orbital_properties_cache = self._compute_orbital_properties()
        return self._orbital_properties_cache
    
    def _compute_orbital_properties(self) -> Dict[str, Any]:
        """Compute orbital properties for this coordinate"""
        # Mandelbrot iteration analysis
        z = 0
        c = self.complex_position
        orbit = []
        
        for i in range(self.iteration_depth):
            z = z*z + c
            orbit.append(z)
            
            if abs(z) > 2.0:  # Escape radius
                return {
                    "type": "divergent",
                    "escape_iteration": i,
                    "escape_velocity": abs(z),
                    "orbit": orbit[:min(10, len(orbit))]  # First 10 iterations
                }
        
        # Check for periodic behavior
        period = self._detect_period(orbit)
        if period:
            return {
                "type": "periodic", 
                "period": period,
                "attracting_cycle": orbit[-period:],
                "orbit": orbit[:min(10, len(orbit))]
            }
        
        return {
            "type": "convergent",
            "final_value": orbit[-1] if orbit else None,
            "orbit": orbit[:min(10, len(orbit))]
        }
    
    def _detect_period(self, orbit: List[complex], tolerance: float = 1e-6) -> Optional[int]:
        """Detect periodic behavior in orbit"""
        if len(orbit) < 4:
            return None
        
        # Check for periods up to half the orbit length
        max_period = min(20, len(orbit) // 2)
        
        for period in range(1, max_period + 1):
            is_periodic = True
            
            # Check if last 'period' values repeat
            for i in range(period):
                if len(orbit) >= 2*period:
                    val1 = orbit[-(period + i)]
                    val2 = orbit[-(i + 1)]
                    if abs(val1 - val2) > tolerance:
                        is_periodic = False
                        break
            
            if is_periodic:
                return period
        
        return None


@dataclass  
class BDNGenealogy:
    """
    Genealogy tracking for Banach Data Node transformations
    
    Maintains complete audit trail of all Banach-Tarski decompositions
    and recompositions to ensure fidelity preservation.
    """
    
    # Node identification
    node_id: str
    parent_node_id: Optional[str] = None
    root_node_id: Optional[str] = None
    generation: int = 0
    
    # Transformation history  
    transformation_chain: List[Dict[str, Any]] = field(default_factory=list)
    creation_method: BDNTransformationType = BDNTransformationType.DECOMPOSITION
    
    # Fidelity tracking
    original_trinity_vector: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    current_trinity_vector: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    fidelity_score: float = 1.0
    information_preservation_verified: bool = True
    
    # Metadata
    created_timestamp: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    
    def add_transformation(self, 
                          transformation_type: BDNTransformationType,
                          source_coordinates: MVSCoordinate,
                          target_coordinates: MVSCoordinate,
                          transformation_data: Dict[str, Any]):
        """Add transformation to genealogy chain"""
        
        transformation_record = {
            "timestamp": datetime.now(),
            "transformation_type": transformation_type.value,
            "source_mvs_coordinate": {
                "complex_position": str(source_coordinates.complex_position),
                "trinity_vector": source_coordinates.trinity_vector,
                "coordinate_id": source_coordinates.coordinate_id
            },
            "target_mvs_coordinate": {
                "complex_position": str(target_coordinates.complex_position), 
                "trinity_vector": target_coordinates.trinity_vector,
                "coordinate_id": target_coordinates.coordinate_id
            },
            "transformation_metadata": transformation_data,
            "pre_transformation_fidelity": self.fidelity_score
        }
        
        self.transformation_chain.append(transformation_record)
        self.last_modified = datetime.now()
        
        # Update current trinity vector
        self.current_trinity_vector = target_coordinates.trinity_vector
        
        # Recalculate fidelity score
        self._update_fidelity_score()
    
    def _update_fidelity_score(self):
        """Update fidelity score based on Trinity vector preservation"""
        if not self.transformation_chain:
            self.fidelity_score = 1.0
            return
        
        # Calculate Trinity vector drift
        original = np.array(self.original_trinity_vector)
        current = np.array(self.current_trinity_vector) 
        
        # Fidelity based on Trinity alignment preservation
        trinity_distance = np.linalg.norm(current - original)
        self.fidelity_score = max(0.0, 1.0 - trinity_distance)
        
        # Update information preservation status
        self.information_preservation_verified = (self.fidelity_score > 0.95)
    
    def get_genealogy_summary(self) -> Dict[str, Any]:
        """Get summary of genealogy for analysis"""
        return {
            "node_id": self.node_id,
            "generation": self.generation,
            "transformation_count": len(self.transformation_chain),
            "fidelity_score": self.fidelity_score,
            "information_preserved": self.information_preservation_verified,
            "trinity_drift": np.linalg.norm(
                np.array(self.current_trinity_vector) - 
                np.array(self.original_trinity_vector)
            ),
            "creation_age": (datetime.now() - self.created_timestamp).total_seconds(),
            "last_transformation": (
                self.transformation_chain[-1]["timestamp"]
                if self.transformation_chain else None
            )
        }


@dataclass
class ModalInferenceResult:
    """
    Result from S5 Modal Logic Inference
    
    Contains modal logic evaluation results with Trinity alignment
    and BDN integration metadata.
    """
    
    # Modal logic results
    formula_evaluated: str
    truth_value: bool
    modal_status: str  # "necessary", "possible", "contingent", "impossible"
    possible_worlds_count: int
    
    # S5 specific properties
    accessibility_relation: str  # "reflexive", "symmetric", "transitive", "universal"
    kripke_model_size: int
    model_consistency_verified: bool
    
    # Trinity integration
    trinity_vector: Tuple[float, float, float]
    mvs_coordinate: MVSCoordinate
    trinity_alignment_preserved: bool
    
    # BDN chain information
    inference_chain_length: int
    bdn_nodes_involved: List[str] = field(default_factory=list)
    genealogy_depth: int = 0
    
    # Computational metadata
    computation_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    convergence_achieved: bool = True
    
    # Inference confidence and quality
    confidence_score: float = 1.0
    logical_consistency_score: float = 1.0
    
    def __post_init__(self):
        """Validate modal inference result"""
        # Ensure confidence scores are in valid range
        self.confidence_score = max(0.0, min(1.0, self.confidence_score))
        self.logical_consistency_score = max(0.0, min(1.0, self.logical_consistency_score))


@dataclass
class CreativeHypothesis:
    """
    Creative Hypothesis generated through cross-domain BDN fusion
    
    Represents novel hypotheses generated by fusing distant domains
    in MVS space through Banach-Tarski creative recombination.
    """
    
    # Hypothesis content
    hypothesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hypothesis_content: str = ""
    formal_representation: Optional[str] = None
    
    # Creative generation metadata
    source_domains: List[str] = field(default_factory=list)
    fusion_coordinates: List[MVSCoordinate] = field(default_factory=list)
    creative_leap_distance: float = 0.0
    
    # BDN fusion information
    parent_bdn_ids: List[str] = field(default_factory=list)
    fusion_genealogy: Optional[BDNGenealogy] = None
    banach_transformation_applied: bool = False
    
    # Quality metrics
    novelty_level: NoveltyLevel = NoveltyLevel.DERIVATIVE
    confidence_score: float = 0.0
    feasibility_score: float = 0.0
    potential_impact_score: float = 0.0
    
    # Validation results
    modal_validation_result: Optional[ModalInferenceResult] = None
    trinity_alignment_verified: bool = False
    logical_consistency_verified: bool = False
    
    # Implementation suggestions
    implementation_approach: Optional[str] = None
    required_resources: List[str] = field(default_factory=list)
    estimated_difficulty: str = "unknown"  # "trivial", "easy", "moderate", "hard", "extreme"
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    generation_method: str = "creative_hypothesis_engine"
    
    # Metadata
    generated_timestamp: datetime = field(default_factory=datetime.now)
    generator_engine: str = "creative_hypothesis_engine"
    
    def calculate_overall_score(self) -> float:
        """Calculate overall hypothesis quality score"""
        weights = {
            "confidence": 0.3,
            "feasibility": 0.2, 
            "impact": 0.25,
            "novelty": 0.25
        }
        
        novelty_score = {
            NoveltyLevel.DERIVATIVE: 0.2,
            NoveltyLevel.COMBINATORIAL: 0.4,
            NoveltyLevel.STRUCTURAL: 0.6,
            NoveltyLevel.PARADIGMATIC: 0.8,
            NoveltyLevel.TRANSCENDENT: 1.0
        }[self.novelty_level]
        
        return (
            weights["confidence"] * self.confidence_score +
            weights["feasibility"] * self.feasibility_score +
            weights["impact"] * self.potential_impact_score +
            weights["novelty"] * novelty_score
        )


@dataclass
class NovelProblem:
    """
    Novel Problem discovered through MVS exploration
    
    Represents entirely new problem categories discovered by exploring
    uncharted regions of Modal Vector Space.
    """
    
    # Problem identification
    problem_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    problem_title: str = ""
    problem_description: str = ""
    formal_specification: Optional[str] = None
    
    # Discovery metadata
    discovery_coordinates: MVSCoordinate = None
    exploration_path: List[MVSCoordinate] = field(default_factory=list)
    discovery_method: str = "mvs_exploration"
    
    # Problem classification
    domain_classification: Optional[str] = None
    complexity_level: str = "unknown"  # "polynomial", "exponential", "undecidable", "open"
    problem_type: str = "optimization"  # "decision", "search", "optimization", "construction"
    
    # Novelty assessment
    novelty_level: NoveltyLevel = NoveltyLevel.DERIVATIVE
    distance_from_known_problems: float = 0.0
    unprecedented_aspects: List[str] = field(default_factory=list)
    
    # Solution approach suggestions
    potential_approaches: List[str] = field(default_factory=list)
    required_mathematical_tools: List[str] = field(default_factory=list)
    estimated_research_time: Optional[str] = None
    
    # Trinity alignment and validation
    trinity_vector: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    trinity_alignment_verified: bool = False
    problem_consistency_verified: bool = False
    
    # Discovery context
    discovery_context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    discovered_timestamp: datetime = field(default_factory=datetime.now)
    discoverer_engine: str = "novel_problem_generator"
    
    def assess_research_value(self) -> float:
        """Assess potential research value of this novel problem"""
        novelty_weight = {
            NoveltyLevel.DERIVATIVE: 0.1,
            NoveltyLevel.COMBINATORIAL: 0.3,
            NoveltyLevel.STRUCTURAL: 0.6, 
            NoveltyLevel.PARADIGMATIC: 0.8,
            NoveltyLevel.TRANSCENDENT: 1.0
        }[self.novelty_level]
        
        # Distance from known problems (normalized)
        distance_score = min(1.0, self.distance_from_known_problems / 10.0)
        
        # Number of unprecedented aspects
        unprecedented_score = min(1.0, len(self.unprecedented_aspects) / 5.0)
        
        return (novelty_weight * 0.4 + distance_score * 0.3 + unprecedented_score * 0.3)


# Type aliases for convenience
MVSCoordinateList = List[MVSCoordinate]
BDNGenealogyChain = List[BDNGenealogy] 
ModalInferenceChain = List[ModalInferenceResult]
CreativeHypothesisList = List[CreativeHypothesis]
NovelProblemList = List[NovelProblem]


# Export all data structures
__all__ = [
    # Enums
    "MVSRegionType",
    "BDNTransformationType", 
    "NoveltyLevel",
    
    # Core data structures
    "MVSCoordinate",
    "BDNGenealogy", 
    "ModalInferenceResult",
    "CreativeHypothesis",
    "NovelProblem",
    
    # Type aliases
    "MVSCoordinateList",
    "BDNGenealogyChain",
    "ModalInferenceChain", 
    "CreativeHypothesisList",
    "NovelProblemList"
]