"""
IEL Error Handler - UIP Step 3 Component
========================================

Comprehensive error handling and recovery for IEL (Integrated Epistemic Logic) components.
Handles modal logic errors, synthesis failures, integration issues with graceful degradation
and detailed diagnostics.

Integrates with: IEL synthesizer, modal validators, ontological validators, V2 framework protocols
Dependencies: Error classification systems, recovery strategies, diagnostic tools, logging frameworks
"""

import inspect
import signal
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from protocols.shared.system_imports import *


class IELErrorType(Enum):
    """Types of IEL errors"""

    # Modal logic errors
    MODAL_CONSISTENCY_ERROR = "modal_consistency_error"
    MODAL_AXIOM_VIOLATION = "modal_axiom_violation"
    ACCESSIBILITY_ERROR = "accessibility_error"
    NECESSITY_IMPOSSIBILITY_CONFLICT = "necessity_impossibility_conflict"

    # Synthesis errors
    DOMAIN_SYNTHESIS_FAILURE = "domain_synthesis_failure"
    INTEGRATION_CONFLICT = "integration_conflict"
    CONVERGENCE_FAILURE = "convergence_failure"
    QUALITY_THRESHOLD_VIOLATION = "quality_threshold_violation"

    # Ontological errors
    CONCEPT_HIERARCHY_VIOLATION = "concept_hierarchy_violation"
    DEFINITIONAL_CIRCULARITY = "definitional_circularity"
    ONTOLOGICAL_INCONSISTENCY = "ontological_inconsistency"
    CATEGORY_ERROR = "category_error"

    # Trinity errors
    TRINITY_VECTOR_INVALID = "trinity_vector_invalid"
    TRINITY_COHERENCE_FAILURE = "trinity_coherence_failure"
    TRINITY_DIMENSION_ERROR = "trinity_dimension_error"

    # Integration errors
    FRAMEWORK_INTEGRATION_ERROR = "framework_integration_error"
    PROTOCOL_VIOLATION = "protocol_violation"
    VALIDATION_ERROR = "validation_error"

    # System errors
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    COMPUTATION_OVERFLOW = "computation_overflow"

    # Data errors
    INVALID_INPUT_FORMAT = "invalid_input_format"
    MISSING_REQUIRED_DATA = "missing_required_data"
    DATA_CORRUPTION = "data_corruption"


class ErrorSeverity(Enum):
    """Severity levels for IEL errors"""

    CRITICAL = "critical"  # System-breaking, requires immediate attention
    HIGH = "high"  # Major functionality impacted
    MEDIUM = "medium"  # Moderate impact, workaround possible
    LOW = "low"  # Minor issue, minimal impact
    WARNING = "warning"  # Potential issue, monitoring needed


class ErrorRecoveryStrategy(Enum):
    """Recovery strategies for different error types"""

    IMMEDIATE_RETRY = "immediate_retry"
    DELAYED_RETRY = "delayed_retry"
    ALTERNATIVE_METHOD = "alternative_method"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    PARTIAL_RECOVERY = "partial_recovery"
    FALLBACK_MODE = "fallback_mode"
    MANUAL_INTERVENTION = "manual_intervention"
    SYSTEM_RESTART = "system_restart"


class ErrorContext(Enum):
    """Context where error occurred"""

    MODAL_ANALYSIS = "modal_analysis"
    DOMAIN_SYNTHESIS = "domain_synthesis"
    ONTOLOGICAL_INTEGRATION = "ontological_integration"
    TRINITY_PROCESSING = "trinity_processing"
    CONSISTENCY_CHECKING = "consistency_checking"
    VALIDATION = "validation"
    INPUT_PROCESSING = "input_processing"
    OUTPUT_GENERATION = "output_generation"


@dataclass
class IELError:
    """Comprehensive IEL error representation"""

    error_type: IELErrorType
    severity: ErrorSeverity
    context: ErrorContext
    message: str
    details: str = ""

    # Error location information
    function_name: Optional[str] = None
    module_name: Optional[str] = None
    line_number: Optional[int] = None

    # Error data
    error_data: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

    # Recovery information
    suggested_recovery: Optional[ErrorRecoveryStrategy] = None
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3

    # Timing information
    timestamp: float = field(default_factory=time.time)
    resolution_time: Optional[float] = None

    # Related errors
    caused_by: Optional["IELError"] = None
    related_errors: List["IELError"] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"[{self.severity.value.upper()}] {self.error_type.value}: {self.message}"
        )

    def get_full_description(self) -> str:
        """Get comprehensive error description"""
        desc = [
            f"Error Type: {self.error_type.value}",
            f"Severity: {self.severity.value}",
            f"Context: {self.context.value}",
            f"Message: {self.message}",
        ]

        if self.details:
            desc.append(f"Details: {self.details}")

        if self.function_name:
            desc.append(f"Function: {self.function_name}")

        if self.module_name:
            desc.append(f"Module: {self.module_name}")

        if self.line_number:
            desc.append(f"Line: {self.line_number}")

        if self.suggested_recovery:
            desc.append(f"Suggested Recovery: {self.suggested_recovery.value}")

        return "\n".join(desc)

    def is_recoverable(self) -> bool:
        """Check if error is recoverable"""
        return (
            self.recovery_attempts < self.max_recovery_attempts
            and self.severity != ErrorSeverity.CRITICAL
            and self.suggested_recovery
            not in [
                ErrorRecoveryStrategy.MANUAL_INTERVENTION,
                ErrorRecoveryStrategy.SYSTEM_RESTART,
            ]
        )

    def mark_resolved(self):
        """Mark error as resolved"""
        self.resolution_time = time.time()


@dataclass
class ErrorRecoveryResult:
    """Result of error recovery attempt"""

    success: bool
    recovery_strategy: ErrorRecoveryStrategy
    recovery_data: Any = None
    new_errors: List[IELError] = field(default_factory=list)
    recovery_time: float = 0.0
    recovery_message: str = ""

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"Recovery {status} using {self.recovery_strategy.value}: {self.recovery_message}"


class ErrorClassifier:
    """Classifies and analyzes IEL errors"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Error classification rules
        self.classification_rules = {
            # Modal logic error patterns
            "modal": {
                "keywords": [
                    "modal",
                    "necessity",
                    "possibility",
                    "accessibility",
                    "world",
                    "frame",
                ],
                "context": ErrorContext.MODAL_ANALYSIS,
                "base_severity": ErrorSeverity.MEDIUM,
            },
            # Synthesis error patterns
            "synthesis": {
                "keywords": [
                    "synthesis",
                    "integration",
                    "domain",
                    "convergence",
                    "quality",
                ],
                "context": ErrorContext.DOMAIN_SYNTHESIS,
                "base_severity": ErrorSeverity.HIGH,
            },
            # Ontological error patterns
            "ontological": {
                "keywords": [
                    "ontology",
                    "concept",
                    "hierarchy",
                    "definition",
                    "category",
                ],
                "context": ErrorContext.ONTOLOGICAL_INTEGRATION,
                "base_severity": ErrorSeverity.MEDIUM,
            },
            # Trinity error patterns
            "trinity": {
                "keywords": [
                    "trinity",
                    "vector",
                    "coherence",
                    "essence",
                    "generation",
                    "temporal",
                ],
                "context": ErrorContext.TRINITY_PROCESSING,
                "base_severity": ErrorSeverity.MEDIUM,
            },
            # System error patterns
            "system": {
                "keywords": ["memory", "timeout", "resource", "overflow", "crash"],
                "context": ErrorContext.INPUT_PROCESSING,
                "base_severity": ErrorSeverity.CRITICAL,
            },
        }

    def classify_error(
        self, exception: Exception, context_hint: Optional[ErrorContext] = None
    ) -> IELError:
        """Classify an exception into an IEL error"""

        error_message = str(exception)
        error_type_name = type(exception).__name__

        # Get stack trace information
        tb = traceback.extract_tb(exception.__traceback__)
        if tb:
            frame = tb[-1]
            function_name = frame.name
            module_name = (
                frame.filename.split("/")[-1]
                if "/" in frame.filename
                else frame.filename.split("\\")[-1]
            )
            line_number = frame.lineno
        else:
            function_name = None
            module_name = None
            line_number = None

        # Determine error type based on exception type and message
        error_type = self._determine_error_type(exception, error_message)

        # Determine severity
        severity = self._determine_severity(exception, error_type, error_message)

        # Determine context
        context = context_hint or self._determine_context(error_message, function_name)

        # Suggest recovery strategy
        suggested_recovery = self._suggest_recovery_strategy(
            error_type, severity, exception
        )

        # Create IEL error
        iel_error = IELError(
            error_type=error_type,
            severity=severity,
            context=context,
            message=error_message,
            details=f"Exception type: {error_type_name}",
            function_name=function_name,
            module_name=module_name,
            line_number=line_number,
            stack_trace=traceback.format_exc(),
            suggested_recovery=suggested_recovery,
            error_data={
                "exception_type": error_type_name,
                "exception_args": getattr(exception, "args", []),
            },
        )

        return iel_error

    def _determine_error_type(self, exception: Exception, message: str) -> IELErrorType:
        """Determine IEL error type from exception"""

        exception_type = type(exception).__name__
        message_lower = message.lower()

        # Exception type mappings
        if isinstance(exception, (MemoryError, OSError)):
            return IELErrorType.MEMORY_ERROR
        elif isinstance(exception, TimeoutError):
            return IELErrorType.TIMEOUT_ERROR
        elif isinstance(exception, (ValueError, TypeError)):
            if any(
                keyword in message_lower
                for keyword in ["modal", "necessity", "possibility"]
            ):
                return IELErrorType.MODAL_CONSISTENCY_ERROR
            elif any(
                keyword in message_lower
                for keyword in ["trinity", "vector", "coherence"]
            ):
                return IELErrorType.TRINITY_VECTOR_INVALID
            elif any(
                keyword in message_lower
                for keyword in ["validation", "invalid", "format"]
            ):
                return IELErrorType.VALIDATION_ERROR
            else:
                return IELErrorType.INVALID_INPUT_FORMAT
        elif isinstance(exception, RecursionError):
            return IELErrorType.COMPUTATION_OVERFLOW
        elif isinstance(exception, KeyError):
            return IELErrorType.MISSING_REQUIRED_DATA

        # Message-based classification
        if any(keyword in message_lower for keyword in ["synthesis", "integration"]):
            if "convergence" in message_lower:
                return IELErrorType.CONVERGENCE_FAILURE
            elif "conflict" in message_lower:
                return IELErrorType.INTEGRATION_CONFLICT
            else:
                return IELErrorType.DOMAIN_SYNTHESIS_FAILURE

        elif any(
            keyword in message_lower
            for keyword in ["modal", "necessity", "possibility"]
        ):
            if "axiom" in message_lower:
                return IELErrorType.MODAL_AXIOM_VIOLATION
            elif "accessibility" in message_lower:
                return IELErrorType.ACCESSIBILITY_ERROR
            else:
                return IELErrorType.MODAL_CONSISTENCY_ERROR

        elif any(
            keyword in message_lower for keyword in ["ontology", "concept", "hierarchy"]
        ):
            if "circular" in message_lower:
                return IELErrorType.DEFINITIONAL_CIRCULARITY
            elif "hierarchy" in message_lower:
                return IELErrorType.CONCEPT_HIERARCHY_VIOLATION
            else:
                return IELErrorType.ONTOLOGICAL_INCONSISTENCY

        elif any(
            keyword in message_lower for keyword in ["trinity", "vector", "coherence"]
        ):
            if "coherence" in message_lower:
                return IELErrorType.TRINITY_COHERENCE_FAILURE
            elif "dimension" in message_lower:
                return IELErrorType.TRINITY_DIMENSION_ERROR
            else:
                return IELErrorType.TRINITY_VECTOR_INVALID

        # Default to framework integration error
        return IELErrorType.FRAMEWORK_INTEGRATION_ERROR

    def _determine_severity(
        self, exception: Exception, error_type: IELErrorType, message: str
    ) -> ErrorSeverity:
        """Determine error severity"""

        # Critical errors
        if isinstance(exception, (MemoryError, SystemError)) or error_type in [
            IELErrorType.MEMORY_ERROR,
            IELErrorType.COMPUTATION_OVERFLOW,
            IELErrorType.SYSTEM_RESTART,
        ]:
            return ErrorSeverity.CRITICAL

        # High severity errors
        if error_type in [
            IELErrorType.DOMAIN_SYNTHESIS_FAILURE,
            IELErrorType.INTEGRATION_CONFLICT,
            IELErrorType.MODAL_AXIOM_VIOLATION,
            IELErrorType.ONTOLOGICAL_INCONSISTENCY,
        ]:
            return ErrorSeverity.HIGH

        # Medium severity errors
        if error_type in [
            IELErrorType.MODAL_CONSISTENCY_ERROR,
            IELErrorType.TRINITY_COHERENCE_FAILURE,
            IELErrorType.CONVERGENCE_FAILURE,
            IELErrorType.VALIDATION_ERROR,
        ]:
            return ErrorSeverity.MEDIUM

        # Low severity errors
        if error_type in [
            IELErrorType.TRINITY_DIMENSION_ERROR,
            IELErrorType.QUALITY_THRESHOLD_VIOLATION,
            IELErrorType.INVALID_INPUT_FORMAT,
        ]:
            return ErrorSeverity.LOW

        # Default to medium severity
        return ErrorSeverity.MEDIUM

    def _determine_context(
        self, message: str, function_name: Optional[str]
    ) -> ErrorContext:
        """Determine error context"""

        message_lower = message.lower()
        function_lower = function_name.lower() if function_name else ""

        # Context mapping based on keywords
        if any(
            keyword in message_lower or keyword in function_lower
            for keyword in ["modal", "necessity", "possibility", "accessibility"]
        ):
            return ErrorContext.MODAL_ANALYSIS

        elif any(
            keyword in message_lower or keyword in function_lower
            for keyword in ["synthesis", "synthesize", "integrate"]
        ):
            return ErrorContext.DOMAIN_SYNTHESIS

        elif any(
            keyword in message_lower or keyword in function_lower
            for keyword in ["ontology", "ontological", "concept", "hierarchy"]
        ):
            return ErrorContext.ONTOLOGICAL_INTEGRATION

        elif any(
            keyword in message_lower or keyword in function_lower
            for keyword in ["trinity", "vector", "coherence"]
        ):
            return ErrorContext.TRINITY_PROCESSING

        elif any(
            keyword in message_lower or keyword in function_lower
            for keyword in ["consistency", "validate", "check"]
        ):
            return ErrorContext.CONSISTENCY_CHECKING

        elif any(
            keyword in message_lower or keyword in function_lower
            for keyword in ["input", "parse", "load"]
        ):
            return ErrorContext.INPUT_PROCESSING

        elif any(
            keyword in message_lower or keyword in function_lower
            for keyword in ["output", "format", "generate", "render"]
        ):
            return ErrorContext.OUTPUT_GENERATION

        # Default context
        return ErrorContext.VALIDATION

    def _suggest_recovery_strategy(
        self, error_type: IELErrorType, severity: ErrorSeverity, exception: Exception
    ) -> ErrorRecoveryStrategy:
        """Suggest recovery strategy based on error characteristics"""

        # Critical errors need manual intervention or system restart
        if severity == ErrorSeverity.CRITICAL:
            if isinstance(exception, MemoryError):
                return ErrorRecoveryStrategy.SYSTEM_RESTART
            else:
                return ErrorRecoveryStrategy.MANUAL_INTERVENTION

        # Recovery strategies by error type
        recovery_mapping = {
            # Modal errors - try alternative methods
            IELErrorType.MODAL_CONSISTENCY_ERROR: ErrorRecoveryStrategy.ALTERNATIVE_METHOD,
            IELErrorType.MODAL_AXIOM_VIOLATION: ErrorRecoveryStrategy.GRACEFUL_DEGRADATION,
            IELErrorType.ACCESSIBILITY_ERROR: ErrorRecoveryStrategy.ALTERNATIVE_METHOD,
            # Synthesis errors - retry or degrade
            IELErrorType.DOMAIN_SYNTHESIS_FAILURE: ErrorRecoveryStrategy.ALTERNATIVE_METHOD,
            IELErrorType.INTEGRATION_CONFLICT: ErrorRecoveryStrategy.PARTIAL_RECOVERY,
            IELErrorType.CONVERGENCE_FAILURE: ErrorRecoveryStrategy.DELAYED_RETRY,
            # Ontological errors - degrade gracefully
            IELErrorType.CONCEPT_HIERARCHY_VIOLATION: ErrorRecoveryStrategy.GRACEFUL_DEGRADATION,
            IELErrorType.DEFINITIONAL_CIRCULARITY: ErrorRecoveryStrategy.ALTERNATIVE_METHOD,
            IELErrorType.ONTOLOGICAL_INCONSISTENCY: ErrorRecoveryStrategy.PARTIAL_RECOVERY,
            # Trinity errors - fallback mode
            IELErrorType.TRINITY_VECTOR_INVALID: ErrorRecoveryStrategy.FALLBACK_MODE,
            IELErrorType.TRINITY_COHERENCE_FAILURE: ErrorRecoveryStrategy.GRACEFUL_DEGRADATION,
            IELErrorType.TRINITY_DIMENSION_ERROR: ErrorRecoveryStrategy.FALLBACK_MODE,
            # System errors
            IELErrorType.TIMEOUT_ERROR: ErrorRecoveryStrategy.DELAYED_RETRY,
            IELErrorType.RESOURCE_EXHAUSTION: ErrorRecoveryStrategy.GRACEFUL_DEGRADATION,
            # Data errors
            IELErrorType.INVALID_INPUT_FORMAT: ErrorRecoveryStrategy.ALTERNATIVE_METHOD,
            IELErrorType.MISSING_REQUIRED_DATA: ErrorRecoveryStrategy.PARTIAL_RECOVERY,
            IELErrorType.VALIDATION_ERROR: ErrorRecoveryStrategy.IMMEDIATE_RETRY,
        }

        return recovery_mapping.get(
            error_type, ErrorRecoveryStrategy.GRACEFUL_DEGRADATION
        )


class ErrorRecoveryEngine:
    """Engine for executing error recovery strategies"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.recovery_handlers: Dict[ErrorRecoveryStrategy, Callable] = {
            ErrorRecoveryStrategy.IMMEDIATE_RETRY: self._immediate_retry,
            ErrorRecoveryStrategy.DELAYED_RETRY: self._delayed_retry,
            ErrorRecoveryStrategy.ALTERNATIVE_METHOD: self._alternative_method,
            ErrorRecoveryStrategy.GRACEFUL_DEGRADATION: self._graceful_degradation,
            ErrorRecoveryStrategy.PARTIAL_RECOVERY: self._partial_recovery,
            ErrorRecoveryStrategy.FALLBACK_MODE: self._fallback_mode,
            ErrorRecoveryStrategy.MANUAL_INTERVENTION: self._manual_intervention,
            ErrorRecoveryStrategy.SYSTEM_RESTART: self._system_restart,
        }

    def attempt_recovery(
        self, error: IELError, operation_func: Callable, *args, **kwargs
    ) -> ErrorRecoveryResult:
        """Attempt to recover from error using suggested strategy"""

        start_time = time.time()

        if not error.is_recoverable():
            return ErrorRecoveryResult(
                success=False,
                recovery_strategy=error.suggested_recovery
                or ErrorRecoveryStrategy.MANUAL_INTERVENTION,
                recovery_message="Error is not recoverable",
                recovery_time=time.time() - start_time,
            )

        # Increment recovery attempt counter
        error.recovery_attempts += 1

        # Get recovery handler
        recovery_strategy = (
            error.suggested_recovery or ErrorRecoveryStrategy.GRACEFUL_DEGRADATION
        )
        recovery_handler = self.recovery_handlers.get(
            recovery_strategy, self._graceful_degradation
        )

        try:
            # Execute recovery strategy
            recovery_result = recovery_handler(error, operation_func, *args, **kwargs)
            recovery_result.recovery_time = time.time() - start_time

            if recovery_result.success:
                error.mark_resolved()
                self.logger.info(f"Successfully recovered from error: {error}")
            else:
                self.logger.warning(f"Recovery failed for error: {error}")

            return recovery_result

        except Exception as recovery_exception:
            self.logger.error(
                f"Recovery attempt failed with exception: {recovery_exception}"
            )

            return ErrorRecoveryResult(
                success=False,
                recovery_strategy=recovery_strategy,
                recovery_message=f"Recovery failed: {recovery_exception}",
                recovery_time=time.time() - start_time,
            )

    def _immediate_retry(
        self, error: IELError, operation_func: Callable, *args, **kwargs
    ) -> ErrorRecoveryResult:
        """Immediate retry of the failed operation"""

        try:
            result = operation_func(*args, **kwargs)
            return ErrorRecoveryResult(
                success=True,
                recovery_strategy=ErrorRecoveryStrategy.IMMEDIATE_RETRY,
                recovery_data=result,
                recovery_message="Immediate retry successful",
            )
        except Exception as e:
            return ErrorRecoveryResult(
                success=False,
                recovery_strategy=ErrorRecoveryStrategy.IMMEDIATE_RETRY,
                recovery_message=f"Immediate retry failed: {e}",
            )

    def _delayed_retry(
        self, error: IELError, operation_func: Callable, *args, **kwargs
    ) -> ErrorRecoveryResult:
        """Delayed retry with exponential backoff"""

        delay = min(2**error.recovery_attempts, 30)  # Max 30 seconds delay

        try:
            time.sleep(delay)
            result = operation_func(*args, **kwargs)
            return ErrorRecoveryResult(
                success=True,
                recovery_strategy=ErrorRecoveryStrategy.DELAYED_RETRY,
                recovery_data=result,
                recovery_message=f"Delayed retry successful after {delay}s",
            )
        except Exception as e:
            return ErrorRecoveryResult(
                success=False,
                recovery_strategy=ErrorRecoveryStrategy.DELAYED_RETRY,
                recovery_message=f"Delayed retry failed: {e}",
            )

    def _alternative_method(
        self, error: IELError, operation_func: Callable, *args, **kwargs
    ) -> ErrorRecoveryResult:
        """Try alternative method based on error type"""

        alternative_data = {}

        # Alternative methods based on error type
        if error.error_type in [
            IELErrorType.MODAL_CONSISTENCY_ERROR,
            IELErrorType.MODAL_AXIOM_VIOLATION,
        ]:
            # Use simplified modal analysis
            alternative_data = {
                "use_simplified_modal": True,
                "skip_axiom_checking": True,
            }

        elif error.error_type == IELErrorType.DOMAIN_SYNTHESIS_FAILURE:
            # Use simpler synthesis strategy
            alternative_data = {
                "synthesis_strategy": "simple_merge",
                "skip_optimization": True,
            }

        elif error.error_type == IELErrorType.TRINITY_COHERENCE_FAILURE:
            # Use average Trinity vectors
            alternative_data = {
                "use_average_trinity": True,
                "skip_coherence_check": True,
            }

        elif error.error_type == IELErrorType.ONTOLOGICAL_INCONSISTENCY:
            # Skip ontological validation
            alternative_data = {"skip_ontological_validation": True}

        try:
            # Update kwargs with alternative parameters
            updated_kwargs = {**kwargs, **alternative_data}
            result = operation_func(*args, **updated_kwargs)

            return ErrorRecoveryResult(
                success=True,
                recovery_strategy=ErrorRecoveryStrategy.ALTERNATIVE_METHOD,
                recovery_data=result,
                recovery_message="Alternative method successful",
            )
        except Exception as e:
            return ErrorRecoveryResult(
                success=False,
                recovery_strategy=ErrorRecoveryStrategy.ALTERNATIVE_METHOD,
                recovery_message=f"Alternative method failed: {e}",
            )

    def _graceful_degradation(
        self, error: IELError, operation_func: Callable, *args, **kwargs
    ) -> ErrorRecoveryResult:
        """Graceful degradation - return partial or simplified result"""

        degraded_result = self._create_degraded_result(error, *args, **kwargs)

        return ErrorRecoveryResult(
            success=True,
            recovery_strategy=ErrorRecoveryStrategy.GRACEFUL_DEGRADATION,
            recovery_data=degraded_result,
            recovery_message="Gracefully degraded result provided",
        )

    def _partial_recovery(
        self, error: IELError, operation_func: Callable, *args, **kwargs
    ) -> ErrorRecoveryResult:
        """Partial recovery - process what we can"""

        try:
            # Try to process subset of data
            partial_args, partial_kwargs = self._create_partial_inputs(
                error, *args, **kwargs
            )
            result = operation_func(*partial_args, **partial_kwargs)

            return ErrorRecoveryResult(
                success=True,
                recovery_strategy=ErrorRecoveryStrategy.PARTIAL_RECOVERY,
                recovery_data=result,
                recovery_message="Partial recovery successful",
            )
        except Exception as e:
            # Fall back to degraded result
            degraded_result = self._create_degraded_result(error, *args, **kwargs)
            return ErrorRecoveryResult(
                success=True,
                recovery_strategy=ErrorRecoveryStrategy.PARTIAL_RECOVERY,
                recovery_data=degraded_result,
                recovery_message=f"Partial recovery fell back to degradation: {e}",
            )

    def _fallback_mode(
        self, error: IELError, operation_func: Callable, *args, **kwargs
    ) -> ErrorRecoveryResult:
        """Fallback mode - use minimal functionality"""

        fallback_result = self._create_fallback_result(error, *args, **kwargs)

        return ErrorRecoveryResult(
            success=True,
            recovery_strategy=ErrorRecoveryStrategy.FALLBACK_MODE,
            recovery_data=fallback_result,
            recovery_message="Fallback mode activated",
        )

    def _manual_intervention(
        self, error: IELError, operation_func: Callable, *args, **kwargs
    ) -> ErrorRecoveryResult:
        """Manual intervention required"""

        return ErrorRecoveryResult(
            success=False,
            recovery_strategy=ErrorRecoveryStrategy.MANUAL_INTERVENTION,
            recovery_message="Manual intervention required - error cannot be automatically resolved",
        )

    def _system_restart(
        self, error: IELError, operation_func: Callable, *args, **kwargs
    ) -> ErrorRecoveryResult:
        """System restart required (not actually implemented for safety)"""

        return ErrorRecoveryResult(
            success=False,
            recovery_strategy=ErrorRecoveryStrategy.SYSTEM_RESTART,
            recovery_message="System restart required - please restart the application",
        )

    def _create_degraded_result(self, error: IELError, *args, **kwargs) -> Any:
        """Create degraded result based on error context"""

        if error.context == ErrorContext.DOMAIN_SYNTHESIS:
            # Return empty synthesis result
            return {
                "synthesized_domain": None,
                "quality_score": 0.0,
                "synthesis_status": "degraded",
                "error_info": error.get_full_description(),
            }

        elif error.context == ErrorContext.MODAL_ANALYSIS:
            # Return minimal modal analysis
            return {
                "modal_properties": {},
                "consistency_score": 0.5,
                "analysis_status": "degraded",
                "error_info": error.get_full_description(),
            }

        elif error.context == ErrorContext.TRINITY_PROCESSING:
            # Return zero Trinity vectors
            return {
                "trinity_vectors": {},
                "coherence_score": 0.0,
                "processing_status": "degraded",
                "error_info": error.get_full_description(),
            }

        else:
            # Generic degraded result
            return {
                "result": None,
                "status": "degraded",
                "error_info": error.get_full_description(),
            }

    def _create_partial_inputs(
        self, error: IELError, *args, **kwargs
    ) -> Tuple[Tuple, Dict]:
        """Create partial inputs for recovery attempt"""

        # Remove problematic parts of input based on error type
        partial_args = args
        partial_kwargs = kwargs.copy()

        if error.error_type == IELErrorType.DOMAIN_SYNTHESIS_FAILURE:
            # Reduce number of domains to synthesize
            if args and isinstance(args[0], list):
                partial_args = (args[0][: len(args[0]) // 2],) + args[1:]

        elif error.error_type == IELErrorType.MODAL_CONSISTENCY_ERROR:
            # Simplify modal analysis
            partial_kwargs["skip_complex_modal_analysis"] = True

        elif error.error_type == IELErrorType.TRINITY_COHERENCE_FAILURE:
            # Skip Trinity analysis
            partial_kwargs["skip_trinity_analysis"] = True

        return partial_args, partial_kwargs

    def _create_fallback_result(self, error: IELError, *args, **kwargs) -> Any:
        """Create minimal fallback result"""

        return {
            "result": "fallback",
            "error_type": error.error_type.value,
            "fallback_reason": "Minimal functionality due to error",
            "timestamp": time.time(),
        }


class IELErrorHandler:
    """Main IEL error handling system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.error_classifier = ErrorClassifier()
        self.recovery_engine = ErrorRecoveryEngine()

        # Configuration
        self.enable_auto_recovery = self.config.get("enable_auto_recovery", True)
        self.max_recovery_attempts = self.config.get("max_recovery_attempts", 3)
        self.enable_detailed_logging = self.config.get("enable_detailed_logging", True)

        # Error tracking
        self.error_history: List[IELError] = []
        self.recovery_statistics = {
            "total_errors": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "recovery_success_rate": 0.0,
        }

        # Error handlers by type
        self.error_handlers: Dict[IELErrorType, Callable] = {}

        self.logger.info("IEL error handler initialized")

    @contextmanager
    def error_context(self, context: ErrorContext, operation_name: str = ""):
        """Context manager for error handling"""

        operation_start = time.time()

        try:
            yield
        except Exception as e:
            # Classify and handle the error
            iel_error = self.error_classifier.classify_error(e, context)
            iel_error.error_data["operation_name"] = operation_name
            iel_error.error_data["operation_duration"] = time.time() - operation_start

            # Log error
            self._log_error(iel_error)

            # Add to history
            self.error_history.append(iel_error)
            self.recovery_statistics["total_errors"] += 1

            # Re-raise the original exception with IEL error attached
            e.iel_error = iel_error
            raise

    def handle_error_with_recovery(
        self,
        exception: Exception,
        operation_func: Callable,
        context: ErrorContext,
        *args,
        **kwargs,
    ) -> Any:
        """
        Handle error with automatic recovery attempt

        Args:
            exception: The exception that occurred
            operation_func: The function that failed
            context: The context where error occurred
            *args, **kwargs: Arguments to pass to operation_func for recovery

        Returns:
            Result of successful recovery or raises exception
        """

        # Classify error
        iel_error = self.error_classifier.classify_error(exception, context)

        # Log error
        self._log_error(iel_error)

        # Add to history
        self.error_history.append(iel_error)
        self.recovery_statistics["total_errors"] += 1

        # Attempt recovery if enabled and error is recoverable
        if self.enable_auto_recovery and iel_error.is_recoverable():
            recovery_result = self.recovery_engine.attempt_recovery(
                iel_error, operation_func, *args, **kwargs
            )

            if recovery_result.success:
                self.recovery_statistics["successful_recoveries"] += 1
                self._update_recovery_statistics()
                self.logger.info(
                    f"Successfully recovered from error: {recovery_result}"
                )
                return recovery_result.recovery_data
            else:
                self.recovery_statistics["failed_recoveries"] += 1
                self._update_recovery_statistics()
                self.logger.error(f"Recovery failed: {recovery_result}")

        # If no recovery or recovery failed, re-raise original exception
        exception.iel_error = iel_error
        raise exception

    def register_error_handler(self, error_type: IELErrorType, handler_func: Callable):
        """Register custom error handler for specific error type"""

        self.error_handlers[error_type] = handler_func
        self.logger.debug(f"Registered custom handler for {error_type.value}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""

        # Error type distribution
        error_type_counts = {}
        for error in self.error_history:
            error_type = error.error_type.value
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

        # Severity distribution
        severity_counts = {}
        for error in self.error_history:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Context distribution
        context_counts = {}
        for error in self.error_history:
            context = error.context.value
            context_counts[context] = context_counts.get(context, 0) + 1

        # Recent errors (last 10)
        recent_errors = [
            error.get_full_description() for error in self.error_history[-10:]
        ]

        return {
            "total_errors": len(self.error_history),
            "error_type_distribution": error_type_counts,
            "severity_distribution": severity_counts,
            "context_distribution": context_counts,
            "recovery_statistics": self.recovery_statistics,
            "recent_errors": recent_errors,
            "most_common_error_type": (
                max(error_type_counts.items(), key=lambda x: x[1])[0]
                if error_type_counts
                else None
            ),
            "average_recovery_time": self._calculate_average_recovery_time(),
        }

    def _log_error(self, error: IELError):
        """Log error with appropriate level"""

        error_msg = (
            error.get_full_description() if self.enable_detailed_logging else str(error)
        )

        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(error_msg)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(error_msg)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(error_msg)
        else:
            self.logger.info(error_msg)

    def _update_recovery_statistics(self):
        """Update recovery success rate"""

        total_recovery_attempts = (
            self.recovery_statistics["successful_recoveries"]
            + self.recovery_statistics["failed_recoveries"]
        )

        if total_recovery_attempts > 0:
            self.recovery_statistics["recovery_success_rate"] = (
                self.recovery_statistics["successful_recoveries"]
                / total_recovery_attempts
            )

    def _calculate_average_recovery_time(self) -> float:
        """Calculate average recovery time from error history"""

        recovery_times = []

        for error in self.error_history:
            if error.resolution_time and error.timestamp:
                recovery_time = error.resolution_time - error.timestamp
                recovery_times.append(recovery_time)

        return sum(recovery_times) / len(recovery_times) if recovery_times else 0.0


# Decorator for automatic error handling
def iel_error_handler(
    context: ErrorContext = ErrorContext.VALIDATION, enable_recovery: bool = True
):
    """Decorator for automatic IEL error handling"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if enable_recovery:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return iel_error_handler_instance.handle_error_with_recovery(
                        e, func, context, *args, **kwargs
                    )
            else:
                with iel_error_handler_instance.error_context(context, func.__name__):
                    return func(*args, **kwargs)

        return wrapper

    return decorator


# Global IEL error handler instance
iel_error_handler_instance = IELErrorHandler()


__all__ = [
    "IELErrorType",
    "ErrorSeverity",
    "ErrorRecoveryStrategy",
    "ErrorContext",
    "IELError",
    "ErrorRecoveryResult",
    "ErrorClassifier",
    "ErrorRecoveryEngine",
    "IELErrorHandler",
    "iel_error_handler",
    "iel_error_handler_instance",
]
