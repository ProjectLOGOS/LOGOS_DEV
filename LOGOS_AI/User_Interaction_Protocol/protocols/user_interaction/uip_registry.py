"""
UIP Registry - User Interaction Protocol Step Handler Management
================================================================

Central registry for managing UIP pipeline steps, routing functions, and step coordination.
Provides the backbone for deterministic, auditable user interaction processing.
"""

from protocols.shared.system_imports import *
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional
from enum import Enum


class UIPStep(Enum):
    """UIP Pipeline Steps"""
    STEP_0_PREPROCESSING = "preprocessing_ingress_routing"
    STEP_1_LINGUISTIC = "linguistic_analysis" 
    STEP_2_PXL_COMPLIANCE = "pxl_compliance_validation"
    STEP_3_IEL_OVERLAY = "iel_overlay_analysis"
    STEP_4_TRINITY_INVOCATION = "trinity_invocation"
    STEP_5_ADAPTIVE_INFERENCE = "adaptive_inference"
    STEP_6_RESPONSE_SYNTHESIS = "response_synthesis"
    STEP_7_COMPLIANCE_RECHECK = "compliance_recheck_audit"
    STEP_8_EGRESS_DELIVERY = "egress_delivery"


class UIPStatus(Enum):
    """UIP Processing Status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    DENIED = "denied"


@dataclass
class UIPContext:
    """Context object passed through UIP pipeline"""
    session_id: str
    correlation_id: str
    user_input: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[UIPStep, Any] = field(default_factory=dict)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    current_step: Optional[UIPStep] = None
    status: UIPStatus = UIPStatus.PENDING
    timestamp: float = field(default_factory=time.time)


@dataclass 
class StepHandler:
    """Handler configuration for a UIP step"""
    step: UIPStep
    handler_func: Callable[[UIPContext], UIPContext]
    dependencies: List[UIPStep] = field(default_factory=list)
    timeout_seconds: int = 30
    retry_count: int = 3
    is_critical: bool = True


class UIPRegistry:
    """Central registry for UIP pipeline management"""
    
    def __init__(self):
        """Initialize UIP registry"""
        self.handlers: Dict[UIPStep, StepHandler] = {}
        self.pipeline_order: List[UIPStep] = list(UIPStep)
        self.active_contexts: Dict[str, UIPContext] = {}
        self.logger = logging.getLogger(__name__)
        
    def register_handler(self, handler: StepHandler):
        """Register a step handler"""
        self.handlers[handler.step] = handler
        self.logger.info(f"Registered handler for {handler.step.value}")
    
    def create_context(self, session_id: str, user_input: str, metadata: Dict[str, Any] = None) -> UIPContext:
        """Create new UIP processing context"""
        correlation_id = str(uuid.uuid4())
        
        context = UIPContext(
            session_id=session_id,
            correlation_id=correlation_id,
            user_input=user_input,
            metadata=metadata or {}
        )
        
        self.active_contexts[correlation_id] = context
        
        # Initial audit entry
        context.audit_trail.append({
            "timestamp": time.time(),
            "step": "INITIALIZATION",
            "action": "context_created",
            "session_id": session_id,
            "correlation_id": correlation_id
        })
        
        return context
    
    async def process_pipeline(self, context: UIPContext) -> UIPContext:
        """Process complete UIP pipeline"""
        self.logger.info(f"Starting UIP pipeline for {context.correlation_id}")
        
        try:
            for step in self.pipeline_order:
                context = await self.process_step(context, step)
                
                if context.status == UIPStatus.FAILED:
                    self.logger.error(f"Pipeline failed at {step.value}")
                    break
                elif context.status == UIPStatus.DENIED:
                    self.logger.warning(f"Pipeline denied at {step.value}")
                    break
                    
            return context
            
        except Exception as e:
            self.logger.error(f"Pipeline error for {context.correlation_id}: {e}")
            context.status = UIPStatus.FAILED
            context.step_results["error"] = str(e)
            return context
        finally:
            # Cleanup active context
            if context.correlation_id in self.active_contexts:
                del self.active_contexts[context.correlation_id]
    
    async def process_step(self, context: UIPContext, step: UIPStep) -> UIPContext:
        """Process individual UIP step"""
        if step not in self.handlers:
            self.logger.error(f"No handler registered for {step.value}")
            context.status = UIPStatus.FAILED
            return context
        
        handler = self.handlers[step]
        context.current_step = step
        context.status = UIPStatus.IN_PROGRESS
        
        # Log step start
        context.audit_trail.append({
            "timestamp": time.time(),
            "step": step.value,
            "action": "step_started",
            "correlation_id": context.correlation_id
        })
        
        try:
            # Check dependencies
            for dep_step in handler.dependencies:
                if dep_step not in context.step_results:
                    raise ValueError(f"Dependency {dep_step.value} not satisfied for {step.value}")
            
            # Execute handler with timeout
            result = await asyncio.wait_for(
                handler.handler_func(context),
                timeout=handler.timeout_seconds
            )
            
            context.step_results[step] = result
            context.status = UIPStatus.COMPLETED
            
            # Log step completion
            context.audit_trail.append({
                "timestamp": time.time(),
                "step": step.value, 
                "action": "step_completed",
                "correlation_id": context.correlation_id,
                "result_summary": str(result)[:200] if result else None
            })
            
        except asyncio.TimeoutError:
            self.logger.error(f"Step {step.value} timed out")
            context.status = UIPStatus.FAILED
            context.audit_trail.append({
                "timestamp": time.time(),
                "step": step.value,
                "action": "step_timeout",
                "correlation_id": context.correlation_id
            })
            
        except Exception as e:
            self.logger.error(f"Step {step.value} failed: {e}")
            context.status = UIPStatus.FAILED
            context.audit_trail.append({
                "timestamp": time.time(),
                "step": step.value,
                "action": "step_failed",
                "error": str(e),
                "correlation_id": context.correlation_id
            })
        
        return context
    
    def get_context(self, correlation_id: str) -> Optional[UIPContext]:
        """Get active context by correlation ID"""
        return self.active_contexts.get(correlation_id)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status"""
        return {
            "registered_handlers": len(self.handlers),
            "active_contexts": len(self.active_contexts),
            "pipeline_steps": [step.value for step in self.pipeline_order],
            "handler_status": {
                step.value: step in self.handlers 
                for step in UIPStep
            }
        }


# Global registry instance
uip_registry = UIPRegistry()


def register_uip_handler(step: UIPStep, dependencies: List[UIPStep] = None, timeout: int = 30, critical: bool = True):
    """Decorator for registering UIP step handlers"""
    def decorator(func: Callable[[UIPContext], UIPContext]) -> Callable:
        handler = StepHandler(
            step=step,
            handler_func=func,
            dependencies=dependencies or [],
            timeout_seconds=timeout,
            is_critical=critical
        )
        uip_registry.register_handler(handler)
        return func
    return decorator


__all__ = [
    'UIPStep',
    'UIPStatus', 
    'UIPContext',
    'StepHandler',
    'UIPRegistry',
    'uip_registry',
    'register_uip_handler'
]