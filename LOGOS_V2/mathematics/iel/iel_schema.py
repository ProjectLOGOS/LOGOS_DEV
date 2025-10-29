"""
IEL Schema Definitions - UIP Step 3 Component
============================================

Protocol definitions for IEL (Integrated Epistemic Logic) framework components.
Defines interfaces for domain synthesis, error handling, and integration with existing 
UIP foundation and V2 framework protocols.

Integrates with: All IEL components, UIP foundation, V2 framework protocols, modal systems
Dependencies: Pydantic validation, typing protocols, abstract base classes, dataclasses
"""

from protocols.shared.system_imports import *
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Protocol, runtime_checkable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, validator, Field, root_validator
import json


# Import from other IEL components for type consistency
try:
    from .iel_synthesizer import DomainType, SynthesisStrategy, KnowledgeLevel, SynthesisQuality
    from .iel_error_handler import IELErrorType, ErrorSeverity, ErrorRecoveryStrategy, ErrorContext
except ImportError:
    # Fallback definitions if imports fail
    from enum import Enum
    
    class DomainType(Enum):
        LOGICAL_DOMAIN = "logical_domain"
        MODAL_DOMAIN = "modal_domain"
        ONTOLOGICAL_DOMAIN = "ontological_domain"
        EPISTEMIC_DOMAIN = "epistemic_domain"
        TRINITY_DOMAIN = "trinity_domain"
    
    class SynthesisStrategy(Enum):
        HIERARCHICAL_SYNTHESIS = "hierarchical_synthesis"
        NETWORK_SYNTHESIS = "network_synthesis"
        MODAL_SYNTHESIS = "modal_synthesis"
        TRINITY_SYNTHESIS = "trinity_synthesis"
    
    class ErrorSeverity(Enum):
        CRITICAL = "critical"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        WARNING = "warning"


# Type variables for generic protocols
T = TypeVar('T')
K = TypeVar('K')  # Knowledge type
D = TypeVar('D')  # Domain type
R = TypeVar('R')  # Result type


# ========================= CORE ENUMERATIONS =========================

class IELProcessingPhase(Enum):
    """Phases of IEL processing"""
    INITIALIZATION = "initialization"
    DOMAIN_ANALYSIS = "domain_analysis"
    MODAL_PROCESSING = "modal_processing"
    ONTOLOGICAL_INTEGRATION = "ontological_integration"
    TRINITY_SYNTHESIS = "trinity_synthesis"
    CONSISTENCY_VALIDATION = "consistency_validation"
    QUALITY_ASSESSMENT = "quality_assessment"
    OUTPUT_GENERATION = "output_generation"
    ERROR_HANDLING = "error_handling"
    FINALIZATION = "finalization"


class IELDataType(Enum):
    """Types of data in IEL processing"""
    CONCEPT = "concept"
    RELATION = "relation"
    AXIOM = "axiom"
    CONSTRAINT = "constraint"
    MODAL_PROPERTY = "modal_property"
    TRINITY_VECTOR = "trinity_vector"
    SYNTHESIS_RESULT = "synthesis_result"
    ERROR_REPORT = "error_report"
    VALIDATION_RESULT = "validation_result"


class IELIntegrationLevel(Enum):
    """Levels of IEL framework integration"""
    COMPONENT_LEVEL = "component_level"      # Individual component integration
    MODULE_LEVEL = "module_level"           # Module-to-module integration
    FRAMEWORK_LEVEL = "framework_level"     # Framework-wide integration
    SYSTEM_LEVEL = "system_level"          # System-wide integration


class IELValidationLevel(Enum):
    """Levels of IEL validation"""
    SYNTACTIC = "syntactic"        # Syntax and format validation
    SEMANTIC = "semantic"          # Semantic correctness validation
    PRAGMATIC = "pragmatic"        # Practical usage validation
    HOLISTIC = "holistic"          # Complete system validation


# ========================= BASE PROTOCOLS =========================

@runtime_checkable
class IELValidatable(Protocol):
    """Base protocol for validatable IEL objects"""
    
    def validate(self) -> 'IELValidationResult':
        """Validate object and return comprehensive result"""
        ...
    
    def is_valid(self) -> bool:
        """Quick validity check"""
        ...
    
    def get_validation_metadata(self) -> Dict[str, Any]:
        """Get validation metadata"""
        ...


@runtime_checkable
class IELProcessable(Protocol[T]):
    """Base protocol for processable IEL objects"""
    
    def process(self, context: 'IELProcessingContext') -> T:
        """Process object in given context"""
        ...
    
    def get_processing_requirements(self) -> List[str]:
        """Get processing requirements"""
        ...
    
    def supports_async_processing(self) -> bool:
        """Check if supports asynchronous processing"""
        ...


@runtime_checkable
class IELSynthesizable(Protocol[K, D]):
    """Protocol for synthesizable knowledge domains"""
    
    def synthesize_with(self, other: K, strategy: SynthesisStrategy) -> D:
        """Synthesize with another knowledge domain"""
        ...
    
    def get_synthesis_compatibility(self, other: K) -> float:
        """Get synthesis compatibility score [0,1]"""
        ...
    
    def prepare_for_synthesis(self) -> K:
        """Prepare domain for synthesis"""
        ...


@runtime_checkable
class IELRecoverable(Protocol):
    """Protocol for error-recoverable components"""
    
    def attempt_recovery(self, error: 'IELError', strategy: ErrorRecoveryStrategy) -> 'IELRecoveryResult':
        """Attempt recovery from error"""
        ...
    
    def get_recovery_capabilities(self) -> List[ErrorRecoveryStrategy]:
        """Get supported recovery strategies"""
        ...
    
    def can_recover_from(self, error_type: 'IELErrorType') -> bool:
        """Check if can recover from specific error type"""
        ...


@runtime_checkable
class IELConfigurable(Protocol):
    """Protocol for configurable IEL components"""
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure component with parameters"""
        ...
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration"""
        ...
    
    def validate_configuration(self, config: Dict[str, Any]) -> 'IELValidationResult':
        """Validate configuration parameters"""
        ...


@runtime_checkable
class IELMonitorable(Protocol):
    """Protocol for monitorable IEL components"""
    
    def get_status(self) -> 'IELComponentStatus':
        """Get current component status"""
        ...
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        ...
    
    def get_health_check(self) -> 'IELHealthCheck':
        """Get health check result"""
        ...


# ========================= CORE DATA STRUCTURES =========================

@dataclass
class IELValidationIssue:
    """Individual validation issue in IEL processing"""
    level: IELValidationLevel
    severity: ErrorSeverity
    message: str
    context: str
    field: Optional[str] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.level.value}: {self.message}"


@dataclass
class IELValidationResult:
    """Comprehensive validation result"""
    valid: bool
    validation_level: IELValidationLevel
    issues: List[IELValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_time: float = 0.0
    
    def add_issue(
        self,
        level: IELValidationLevel,
        severity: ErrorSeverity,
        message: str,
        context: str,
        field: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        """Add validation issue"""
        issue = IELValidationIssue(
            level=level,
            severity=severity,
            message=message,
            context=context,
            field=field,
            suggestion=suggestion
        )
        self.issues.append(issue)
        
        # Mark as invalid if error severity
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.valid = False
    
    def has_critical_issues(self) -> bool:
        """Check for critical validation issues"""
        return any(issue.severity == ErrorSeverity.CRITICAL for issue in self.issues)
    
    def get_issues_by_severity(self, severity: ErrorSeverity) -> List[IELValidationIssue]:
        """Get issues by severity level"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_summary(self) -> str:
        """Get validation summary"""
        if self.valid:
            return f"Valid ({len(self.issues)} warnings)"
        else:
            error_count = len([i for i in self.issues if i.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]])
            warning_count = len([i for i in self.issues if i.severity in [ErrorSeverity.MEDIUM, ErrorSeverity.LOW]])
            return f"Invalid: {error_count} errors, {warning_count} warnings"


@dataclass
class IELProcessingContext:
    """Context for IEL processing operations"""
    phase: IELProcessingPhase
    integration_level: IELIntegrationLevel
    processing_id: str
    parent_context: Optional['IELProcessingContext'] = None
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # State tracking
    start_time: float = field(default_factory=time.time)
    current_step: str = ""
    steps_completed: List[str] = field(default_factory=list)
    
    # Data tracking
    input_data: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    
    # Error tracking
    errors: List['IELError'] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_step(self, step_name: str):
        """Add completed step"""
        self.steps_completed.append(step_name)
        self.current_step = step_name
    
    def add_intermediate_result(self, key: str, value: Any):
        """Add intermediate result"""
        self.intermediate_results[key] = value
    
    def get_processing_time(self) -> float:
        """Get total processing time"""
        return time.time() - self.start_time
    
    def create_child_context(self, phase: IELProcessingPhase, processing_id: str) -> 'IELProcessingContext':
        """Create child processing context"""
        return IELProcessingContext(
            phase=phase,
            integration_level=self.integration_level,
            processing_id=processing_id,
            parent_context=self,
            config=self.config.copy()
        )


@dataclass
class IELComponentStatus:
    """Status information for IEL component"""
    component_name: str
    status: str  # "active", "idle", "processing", "error", "disabled"
    last_activity: float
    processing_count: int = 0
    error_count: int = 0
    
    # Performance metrics
    average_processing_time: float = 0.0
    success_rate: float = 1.0
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def is_healthy(self) -> bool:
        """Check if component is healthy"""
        return (self.status not in ["error", "disabled"] and
                self.success_rate > 0.8 and
                self.cpu_usage_percent < 90.0)


@dataclass
class IELHealthCheck:
    """Health check result for IEL component"""
    healthy: bool
    component_name: str
    timestamp: float = field(default_factory=time.time)
    
    # Detailed health information
    status_checks: Dict[str, bool] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Issues and recommendations
    health_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def add_check(self, check_name: str, passed: bool, details: str = ""):
        """Add health check result"""
        self.status_checks[check_name] = passed
        if not passed:
            self.healthy = False
            self.health_issues.append(f"{check_name}: {details}")
    
    def add_recommendation(self, recommendation: str):
        """Add health recommendation"""
        self.recommendations.append(recommendation)


# ========================= PYDANTIC MODELS =========================

class IELKnowledgeModel(BaseModel):
    """Pydantic model for IEL knowledge representation"""
    
    concepts: Set[str] = Field(..., min_items=1, description="Set of concepts")
    relations: List[Dict[str, Any]] = Field(default_factory=list, description="Knowledge relations")
    axioms: List[str] = Field(default_factory=list, description="Logical axioms")
    constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Knowledge constraints")
    
    # Modal properties
    modal_properties: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Modal properties")
    
    # Trinity vectors
    trinity_vectors: Dict[str, Tuple[float, float, float]] = Field(default_factory=dict, description="Trinity vectors")
    
    # Quality metrics
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Knowledge confidence")
    completeness: float = Field(default=0.0, ge=0.0, le=1.0, description="Knowledge completeness")
    consistency_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Consistency score")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        extra = "allow"
        
    @validator('concepts')
    def validate_concepts(cls, v):
        """Validate concepts are non-empty strings"""
        for concept in v:
            if not isinstance(concept, str) or not concept.strip():
                raise ValueError("Concepts must be non-empty strings")
        return v
    
    @validator('trinity_vectors')
    def validate_trinity_vectors(cls, v):
        """Validate Trinity vectors"""
        for concept, vector in v.items():
            if len(vector) != 3:
                raise ValueError(f"Trinity vector for '{concept}' must have 3 dimensions")
            if not all(0.0 <= dim <= 1.0 for dim in vector):
                raise ValueError(f"Trinity vector dimensions for '{concept}' must be in [0,1]")
        return v
    
    @root_validator
    def validate_consistency(cls, values):
        """Validate overall consistency"""
        concepts = values.get('concepts', set())
        trinity_vectors = values.get('trinity_vectors', {})
        
        # Check Trinity vectors reference valid concepts
        for concept in trinity_vectors:
            if concept not in concepts:
                raise ValueError(f"Trinity vector concept '{concept}' not in concept set")
        
        return values


class IELSynthesisConfigModel(BaseModel):
    """Pydantic model for synthesis configuration"""
    
    synthesis_strategy: SynthesisStrategy = Field(..., description="Synthesis strategy")
    target_quality: float = Field(default=0.7, ge=0.0, le=1.0, description="Target synthesis quality")
    max_iterations: int = Field(default=10, ge=1, le=100, description="Maximum synthesis iterations")
    
    # Feature toggles
    enable_modal_analysis: bool = Field(default=True, description="Enable modal analysis")
    enable_trinity_processing: bool = Field(default=True, description="Enable Trinity processing")
    enable_ontological_integration: bool = Field(default=True, description="Enable ontological integration")
    enable_consistency_checking: bool = Field(default=True, description="Enable consistency checking")
    
    # Thresholds
    convergence_threshold: float = Field(default=0.01, ge=0.0, le=1.0, description="Convergence threshold")
    compatibility_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Compatibility threshold")
    quality_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Quality threshold")
    
    # Processing limits
    max_domains: int = Field(default=20, ge=1, le=100, description="Maximum domains to synthesize")
    max_concepts_per_domain: int = Field(default=1000, ge=1, description="Maximum concepts per domain")
    max_processing_time: float = Field(default=300.0, gt=0.0, description="Maximum processing time (seconds)")
    
    class Config:
        use_enum_values = True


class IELErrorModel(BaseModel):
    """Pydantic model for IEL errors"""
    
    error_type: str = Field(..., description="Error type identifier")
    severity: str = Field(..., description="Error severity level")
    context: str = Field(..., description="Error context")
    message: str = Field(..., min_length=1, description="Error message")
    
    # Optional fields
    details: str = Field(default="", description="Additional error details")
    function_name: Optional[str] = Field(None, description="Function where error occurred")
    module_name: Optional[str] = Field(None, description="Module where error occurred")
    line_number: Optional[int] = Field(None, ge=1, description="Line number where error occurred")
    
    # Recovery information
    suggested_recovery: Optional[str] = Field(None, description="Suggested recovery strategy")
    recovery_attempts: int = Field(default=0, ge=0, description="Number of recovery attempts")
    max_recovery_attempts: int = Field(default=3, ge=1, description="Maximum recovery attempts")
    
    # Timing
    timestamp: float = Field(default_factory=time.time, description="Error timestamp")
    resolution_time: Optional[float] = Field(None, description="Resolution timestamp")
    
    # Additional data
    error_data: Dict[str, Any] = Field(default_factory=dict, description="Additional error data")


# ========================= DOMAIN-SPECIFIC PROTOCOLS =========================

@runtime_checkable
class IELDomainSynthesizer(Protocol):
    """Protocol for IEL domain synthesis"""
    
    def synthesize_domains(
        self,
        domains: List['IELDomainKnowledge'],
        strategy: SynthesisStrategy,
        config: Optional[IELSynthesisConfigModel] = None
    ) -> 'IELSynthesisResult':
        """Synthesize multiple knowledge domains"""
        ...
    
    def analyze_domain_compatibility(
        self,
        domain1: 'IELDomainKnowledge',
        domain2: 'IELDomainKnowledge'
    ) -> float:
        """Analyze compatibility between domains"""
        ...
    
    def validate_synthesis_input(
        self,
        domains: List['IELDomainKnowledge']
    ) -> IELValidationResult:
        """Validate synthesis input"""
        ...


@runtime_checkable
class IELErrorHandler(Protocol):
    """Protocol for IEL error handling"""
    
    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        recovery_enabled: bool = True
    ) -> 'IELErrorHandlingResult':
        """Handle error with optional recovery"""
        ...
    
    def classify_error(
        self,
        error: Exception,
        context_hint: Optional[ErrorContext] = None
    ) -> 'IELError':
        """Classify exception into IEL error"""
        ...
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics"""
        ...


@runtime_checkable
class IELModalAnalyzer(Protocol):
    """Protocol for modal logic analysis"""
    
    def analyze_modal_structure(
        self,
        knowledge: 'IELDomainKnowledge'
    ) -> Dict[str, Any]:
        """Analyze modal structure of knowledge"""
        ...
    
    def validate_modal_consistency(
        self,
        modal_properties: Dict[str, Dict[str, Any]]
    ) -> IELValidationResult:
        """Validate modal consistency"""
        ...
    
    def extract_modal_relations(
        self,
        knowledge: 'IELDomainKnowledge'
    ) -> List[Dict[str, Any]]:
        """Extract modal relations from knowledge"""
        ...


@runtime_checkable
class IELTrinityProcessor(Protocol):
    """Protocol for Trinity processing"""
    
    def process_trinity_vectors(
        self,
        trinity_data: Dict[str, Tuple[float, float, float]]
    ) -> Dict[str, Any]:
        """Process Trinity vector data"""
        ...
    
    def calculate_trinity_coherence(
        self,
        vectors: Dict[str, Tuple[float, float, float]]
    ) -> float:
        """Calculate Trinity coherence across vectors"""
        ...
    
    def validate_trinity_vectors(
        self,
        vectors: Dict[str, Tuple[float, float, float]]
    ) -> IELValidationResult:
        """Validate Trinity vectors"""
        ...


@runtime_checkable
class IELOntologyIntegrator(Protocol):
    """Protocol for ontology integration"""
    
    def integrate_ontologies(
        self,
        domains: List['IELDomainKnowledge']
    ) -> Dict[str, Any]:
        """Integrate ontologies from multiple domains"""
        ...
    
    def detect_ontological_conflicts(
        self,
        domains: List['IELDomainKnowledge']
    ) -> List[Dict[str, Any]]:
        """Detect conflicts between ontologies"""
        ...
    
    def resolve_ontological_conflicts(
        self,
        conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Resolve ontological conflicts"""
        ...


# ========================= COMPOSITE PROTOCOLS =========================

@runtime_checkable
class IELFrameworkComponent(IELValidatable, IELConfigurable, IELMonitorable, Protocol):
    """Comprehensive IEL framework component protocol"""
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize component"""
        ...
    
    def process(self, input_data: Any, context: IELProcessingContext) -> Any:
        """Process input data"""
        ...
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize component processing"""
        ...


@runtime_checkable
class IELIntegratedSystem(Protocol):
    """Protocol for integrated IEL system"""
    
    def register_component(self, component: IELFrameworkComponent, name: str) -> bool:
        """Register framework component"""
        ...
    
    def execute_pipeline(
        self,
        input_data: Any,
        pipeline_config: Dict[str, Any]
    ) -> 'IELPipelineResult':
        """Execute complete processing pipeline"""
        ...
    
    def get_system_status(self) -> Dict[str, IELComponentStatus]:
        """Get status of all components"""
        ...
    
    def perform_system_health_check(self) -> Dict[str, IELHealthCheck]:
        """Perform comprehensive system health check"""
        ...


# ========================= RESULT STRUCTURES =========================

@dataclass
class IELSynthesisResult:
    """Result of IEL domain synthesis"""
    success: bool
    synthesized_knowledge: Optional['IELDomainKnowledge']
    synthesis_quality: float
    processing_time: float
    
    # Analysis results
    modal_analysis: Dict[str, Any] = field(default_factory=dict)
    trinity_analysis: Dict[str, Any] = field(default_factory=dict)
    ontological_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Diagnostics
    synthesis_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    resolution_strategies: List[str] = field(default_factory=list)
    emergence_indicators: List[str] = field(default_factory=list)
    
    # Metadata
    synthesis_strategy: Optional[SynthesisStrategy] = None
    source_domain_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_synthesis_summary(self) -> str:
        """Get synthesis result summary"""
        status = "SUCCESS" if self.success else "FAILURE"
        return f"Synthesis {status}: Quality {self.synthesis_quality:.2f}, {self.source_domain_count} domains, {self.processing_time:.2f}s"


@dataclass
class IELErrorHandlingResult:
    """Result of IEL error handling"""
    handled: bool
    recovery_attempted: bool
    recovery_successful: bool
    
    original_error: Exception
    iel_error: 'IELError'
    recovery_result: Optional[Any] = None
    recovery_strategy: Optional[ErrorRecoveryStrategy] = None
    
    processing_time: float = 0.0
    additional_errors: List['IELError'] = field(default_factory=list)
    
    def get_handling_summary(self) -> str:
        """Get error handling summary"""
        if self.recovery_successful:
            return f"Error handled successfully via {self.recovery_strategy.value if self.recovery_strategy else 'unknown'}"
        elif self.recovery_attempted:
            return f"Recovery attempted but failed: {self.iel_error.message}"
        else:
            return f"Error handled without recovery: {self.iel_error.message}"


@dataclass
class IELPipelineResult:
    """Result of complete IEL pipeline execution"""
    success: bool
    pipeline_id: str
    processing_time: float
    
    # Phase results
    phase_results: Dict[IELProcessingPhase, Any] = field(default_factory=dict)
    
    # Final output
    final_output: Optional[Any] = None
    
    # Quality metrics
    overall_quality: float = 0.0
    phase_qualities: Dict[IELProcessingPhase, float] = field(default_factory=dict)
    
    # Error information
    errors: List['IELError'] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    def get_pipeline_summary(self) -> str:
        """Get pipeline execution summary"""
        status = "SUCCESS" if self.success else "FAILURE"
        error_count = len(self.errors)
        warning_count = len(self.warnings)
        return f"Pipeline {status}: Quality {self.overall_quality:.2f}, {error_count} errors, {warning_count} warnings, {self.processing_time:.2f}s"


# ========================= UTILITY FUNCTIONS =========================

def create_iel_processing_context(
    phase: IELProcessingPhase,
    integration_level: IELIntegrationLevel = IELIntegrationLevel.COMPONENT_LEVEL,
    config: Optional[Dict[str, Any]] = None
) -> IELProcessingContext:
    """Create IEL processing context"""
    
    import uuid
    
    return IELProcessingContext(
        phase=phase,
        integration_level=integration_level,
        processing_id=str(uuid.uuid4()),
        config=config or {}
    )


def validate_iel_knowledge(knowledge: IELKnowledgeModel) -> IELValidationResult:
    """Validate IEL knowledge using Pydantic model"""
    
    result = IELValidationResult(
        valid=True,
        validation_level=IELValidationLevel.SEMANTIC
    )
    
    try:
        # Pydantic validation
        knowledge.dict()
        result.metadata['pydantic_validation'] = 'passed'
    except Exception as e:
        result.add_issue(
            level=IELValidationLevel.SYNTACTIC,
            severity=ErrorSeverity.HIGH,
            message=f"Pydantic validation failed: {e}",
            context="knowledge_validation"
        )
    
    return result


def merge_iel_validation_results(results: List[IELValidationResult]) -> IELValidationResult:
    """Merge multiple IEL validation results"""
    
    merged = IELValidationResult(
        valid=all(r.valid for r in results),
        validation_level=IELValidationLevel.HOLISTIC
    )
    
    # Collect all issues
    for result in results:
        merged.issues.extend(result.issues)
    
    # Merge metadata
    for result in results:
        merged.metadata.update(result.metadata)
    
    # Calculate total validation time
    merged.validation_time = sum(r.validation_time for r in results)
    
    return merged


def create_iel_health_check(
    component_name: str,
    status_checks: Dict[str, bool],
    performance_metrics: Optional[Dict[str, float]] = None,
    resource_usage: Optional[Dict[str, float]] = None
) -> IELHealthCheck:
    """Create IEL health check result"""
    
    health_check = IELHealthCheck(
        healthy=all(status_checks.values()),
        component_name=component_name,
        status_checks=status_checks,
        performance_metrics=performance_metrics or {},
        resource_usage=resource_usage or {}
    )
    
    # Add recommendations based on checks
    for check_name, passed in status_checks.items():
        if not passed:
            health_check.add_recommendation(f"Address issue with {check_name}")
    
    return health_check


def serialize_iel_result(result: Union[IELSynthesisResult, IELErrorHandlingResult, IELPipelineResult]) -> str:
    """Serialize IEL result to JSON"""
    
    def serialize_object(obj):
        """Custom serializer for complex objects"""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)
    
    try:
        if hasattr(result, '__dict__'):
            data = result.__dict__
        else:
            data = result
        
        return json.dumps(data, default=serialize_object, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({'serialization_error': str(e)})


def deserialize_iel_result(json_str: str, result_type: Type[T]) -> Optional[T]:
    """Deserialize JSON to IEL result object"""
    
    try:
        data = json.loads(json_str)
        
        # Simple reconstruction (can be enhanced)
        if hasattr(result_type, '__annotations__'):
            return result_type(**data)
        else:
            return data
    except Exception as e:
        logging.error(f"Failed to deserialize IEL result: {e}")
        return None


# ========================= SCHEMA VALIDATORS =========================

class IELSchemaValidator:
    """Comprehensive IEL schema validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_synthesis_config(self, config: Dict[str, Any]) -> IELValidationResult:
        """Validate synthesis configuration"""
        
        result = IELValidationResult(
            valid=True,
            validation_level=IELValidationLevel.SYNTACTIC
        )
        
        try:
            # Use Pydantic model for validation
            config_model = IELSynthesisConfigModel(**config)
            result.metadata['validated_config'] = config_model.dict()
        except Exception as e:
            result.add_issue(
                level=IELValidationLevel.SYNTACTIC,
                severity=ErrorSeverity.HIGH,
                message=f"Configuration validation failed: {e}",
                context="synthesis_config"
            )
        
        return result
    
    def validate_knowledge_domain(self, domain_data: Dict[str, Any]) -> IELValidationResult:
        """Validate knowledge domain data"""
        
        result = IELValidationResult(
            valid=True,
            validation_level=IELValidationLevel.SEMANTIC
        )
        
        try:
            # Use Pydantic model for validation
            knowledge_model = IELKnowledgeModel(**domain_data)
            result.metadata['validated_knowledge'] = knowledge_model.dict()
        except Exception as e:
            result.add_issue(
                level=IELValidationLevel.SEMANTIC,
                severity=ErrorSeverity.HIGH,
                message=f"Knowledge validation failed: {e}",
                context="knowledge_domain"
            )
        
        return result
    
    def validate_processing_context(self, context: IELProcessingContext) -> IELValidationResult:
        """Validate processing context"""
        
        result = IELValidationResult(
            valid=True,
            validation_level=IELValidationLevel.PRAGMATIC
        )
        
        # Check required fields
        if not context.processing_id:
            result.add_issue(
                level=IELValidationLevel.SYNTACTIC,
                severity=ErrorSeverity.HIGH,
                message="Processing context missing processing_id",
                context="processing_context"
            )
        
        # Check phase validity
        if not isinstance(context.phase, IELProcessingPhase):
            result.add_issue(
                level=IELValidationLevel.SYNTACTIC,
                severity=ErrorSeverity.HIGH,
                message="Invalid processing phase",
                context="processing_context"
            )
        
        # Check integration level
        if not isinstance(context.integration_level, IELIntegrationLevel):
            result.add_issue(
                level=IELValidationLevel.SYNTACTIC,
                severity=ErrorSeverity.MEDIUM,
                message="Invalid integration level",
                context="processing_context"
            )
        
        return result
    
    def validate_component_protocol_compliance(
        self,
        component: Any,
        expected_protocols: List[Type]
    ) -> IELValidationResult:
        """Validate component protocol compliance"""
        
        result = IELValidationResult(
            valid=True,
            validation_level=IELValidationLevel.HOLISTIC
        )
        
        for protocol in expected_protocols:
            if not isinstance(component, protocol):
                result.add_issue(
                    level=IELValidationLevel.SYNTACTIC,
                    severity=ErrorSeverity.HIGH,
                    message=f"Component does not implement protocol {protocol.__name__}",
                    context="protocol_compliance"
                )
        
        return result


# Global schema validator instance
iel_schema_validator = IELSchemaValidator()


__all__ = [
    # Enums
    'IELProcessingPhase',
    'IELDataType', 
    'IELIntegrationLevel',
    'IELValidationLevel',
    
    # Base protocols
    'IELValidatable',
    'IELProcessable',
    'IELSynthesizable',
    'IELRecoverable',
    'IELConfigurable',
    'IELMonitorable',
    
    # Domain-specific protocols
    'IELDomainSynthesizer',
    'IELErrorHandler',
    'IELModalAnalyzer',
    'IELTrinityProcessor',
    'IELOntologyIntegrator',
    
    # Composite protocols
    'IELFrameworkComponent',
    'IELIntegratedSystem',
    
    # Data structures
    'IELValidationIssue',
    'IELValidationResult',
    'IELProcessingContext',
    'IELComponentStatus',
    'IELHealthCheck',
    
    # Pydantic models
    'IELKnowledgeModel',
    'IELSynthesisConfigModel',
    'IELErrorModel',
    
    # Result structures
    'IELSynthesisResult',
    'IELErrorHandlingResult',
    'IELPipelineResult',
    
    # Utility functions
    'create_iel_processing_context',
    'validate_iel_knowledge',
    'merge_iel_validation_results',
    'create_iel_health_check',
    'serialize_iel_result',
    'deserialize_iel_result',
    
    # Validator
    'IELSchemaValidator',
    'iel_schema_validator'
]