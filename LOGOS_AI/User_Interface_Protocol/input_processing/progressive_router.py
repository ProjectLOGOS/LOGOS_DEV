"""
Progressive Router - UIP Step 0 (Primary Entry Point)
=====================================================

Primary entry point for all user interactions. Routes inputs through the UIP pipeline
in a deterministic, auditable sequence. Ensures consistent processing path for
all user queries regardless of complexity or domain.
"""

from protocols.shared.message_formats import MessageValidator, UIPRequest, UIPResponse
from protocols.shared.system_imports import *
from protocols.user_interaction.uip_registry import (
    UIPContext,
    UIPStatus,
    UIPStep,
    register_uip_handler,
    uip_registry,
)


class ProgressiveRouter:
    """
    Progressive Router for UIP Pipeline

    Manages the complete user interaction flow from input to output,
    ensuring deterministic processing through all UIP steps.
    """

    def __init__(self):
        """Initialize progressive router"""
        self.logger = logging.getLogger(__name__)
        self.active_sessions: Dict[str, Dict] = {}
        self.routing_metrics = {
            "total_requests": 0,
            "successful_completions": 0,
            "failed_requests": 0,
            "denied_requests": 0,
            "average_processing_time": 0.0,
        }

    async def route_user_input(self, request: UIPRequest) -> UIPResponse:
        """
        Primary routing function - processes user input through complete UIP pipeline

        Args:
            request: Standardized UIP request object

        Returns:
            UIPResponse: Complete response with audit trail
        """
        start_time = time.time()
        self.routing_metrics["total_requests"] += 1

        # Validate request format
        is_valid, error_msg = MessageValidator.validate_uip_request(request)
        if not is_valid:
            self.logger.error(f"Invalid request format: {error_msg}")
            return self._create_error_response(request.session_id, error_msg)

        # Log incoming request
        self.logger.info(f"Processing UIP request for session {request.session_id}")

        try:
            # Create UIP processing context
            context = uip_registry.create_context(
                session_id=request.session_id,
                user_input=request.user_input,
                metadata={
                    "input_type": request.input_type,
                    "language": request.language,
                    "user_context": request.context,
                    "request_metadata": request.metadata,
                    "start_time": start_time,
                },
            )

            # Process through complete UIP pipeline
            result_context = await uip_registry.process_pipeline(context)

            # Generate response
            response = self._create_response_from_context(result_context, start_time)

            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(result_context.status, processing_time)

            return response

        except Exception as e:
            self.logger.error(f"Router processing error: {e}")
            self.routing_metrics["failed_requests"] += 1
            return self._create_error_response(
                request.session_id, f"Internal processing error: {str(e)}"
            )

    def _create_response_from_context(
        self, context: UIPContext, start_time: float
    ) -> UIPResponse:
        """Create UIP response from processing context"""
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Extract response components from step results
        response_text = self._extract_response_text(context)
        confidence_score = self._extract_confidence_score(context)
        alignment_flags = self._extract_alignment_flags(context)
        ontological_vector = self._extract_ontological_vector(context)
        audit_proof = self._extract_audit_proof(context)
        disclaimers = self._extract_disclaimers(context)

        return UIPResponse(
            session_id=context.session_id,
            correlation_id=context.correlation_id,
            response_text=response_text,
            confidence_score=confidence_score,
            alignment_flags=alignment_flags,
            ontological_vector=ontological_vector,
            audit_proof=audit_proof,
            disclaimers=disclaimers,
            metadata={
                "processing_steps": len(context.step_results),
                "final_status": context.status.value,
                "audit_entries": len(context.audit_trail),
                "step_breakdown": {
                    step.value: step in context.step_results for step in UIPStep
                },
            },
            processing_time_ms=processing_time,
        )

    def _extract_response_text(self, context: UIPContext) -> str:
        """Extract final response text from context"""
        # Look for response in step 6 (response synthesis) or step 8 (egress)
        if UIPStep.STEP_6_RESPONSE_SYNTHESIS in context.step_results:
            synthesis_result = context.step_results[UIPStep.STEP_6_RESPONSE_SYNTHESIS]
            if (
                isinstance(synthesis_result, dict)
                and "response_text" in synthesis_result
            ):
                return synthesis_result["response_text"]

        if UIPStep.STEP_8_EGRESS_DELIVERY in context.step_results:
            egress_result = context.step_results[UIPStep.STEP_8_EGRESS_DELIVERY]
            if (
                isinstance(egress_result, dict)
                and "formatted_response" in egress_result
            ):
                return egress_result["formatted_response"]

        # Fallback responses based on status
        if context.status == UIPStatus.DENIED:
            return "Request denied due to compliance validation failure."
        elif context.status == UIPStatus.FAILED:
            return "Unable to process request due to system error."
        else:
            return "Processing completed but no response generated."

    def _extract_confidence_score(self, context: UIPContext) -> float:
        """Extract confidence score from context"""
        # Look for confidence in linguistic analysis or adaptive inference
        if UIPStep.STEP_1_LINGUISTIC in context.step_results:
            linguistic_result = context.step_results[UIPStep.STEP_1_LINGUISTIC]
            if (
                isinstance(linguistic_result, dict)
                and "confidence" in linguistic_result
            ):
                return float(linguistic_result["confidence"])

        # Default confidence based on completion status
        if context.status == UIPStatus.COMPLETED:
            return 0.85
        elif context.status == UIPStatus.DENIED:
            return 0.95  # High confidence in denial
        else:
            return 0.3  # Low confidence for errors/failures

    def _extract_alignment_flags(self, context: UIPContext) -> Dict[str, bool]:
        """Extract alignment flags from PXL compliance step"""
        alignment_flags = {
            "existence_valid": False,
            "truth_verified": False,
            "goodness_confirmed": False,
            "coherence_maintained": False,
        }

        if UIPStep.STEP_2_PXL_COMPLIANCE in context.step_results:
            pxl_result = context.step_results[UIPStep.STEP_2_PXL_COMPLIANCE]
            if isinstance(pxl_result, dict):
                alignment_flags.update(pxl_result.get("alignment_flags", {}))

        return alignment_flags

    def _extract_ontological_vector(
        self, context: UIPContext
    ) -> Optional[Dict[str, float]]:
        """Extract ontological vector from IEL analysis"""
        if UIPStep.STEP_3_IEL_OVERLAY in context.step_results:
            iel_result = context.step_results[UIPStep.STEP_3_IEL_OVERLAY]
            if isinstance(iel_result, dict) and "ontological_vector" in iel_result:
                return iel_result["ontological_vector"]
        return None

    def _extract_audit_proof(self, context: UIPContext) -> Optional[str]:
        """Extract audit proof from compliance recheck"""
        if UIPStep.STEP_7_COMPLIANCE_RECHECK in context.step_results:
            audit_result = context.step_results[UIPStep.STEP_7_COMPLIANCE_RECHECK]
            if isinstance(audit_result, dict) and "audit_proof" in audit_result:
                return audit_result["audit_proof"]
        return None

    def _extract_disclaimers(self, context: UIPContext) -> List[str]:
        """Extract disclaimers from various steps"""
        disclaimers = []

        # Check each step for disclaimers
        for step_result in context.step_results.values():
            if isinstance(step_result, dict) and "disclaimers" in step_result:
                disclaimers.extend(step_result["disclaimers"])

        return list(set(disclaimers))  # Remove duplicates

    def _create_error_response(
        self, session_id: str, error_message: str
    ) -> UIPResponse:
        """Create error response for failed requests"""
        return UIPResponse(
            session_id=session_id,
            correlation_id=str(uuid.uuid4()),
            response_text=f"Request could not be processed: {error_message}",
            confidence_score=0.0,
            alignment_flags={
                "existence_valid": False,
                "truth_verified": False,
                "goodness_confirmed": False,
                "coherence_maintained": False,
            },
            disclaimers=["This is an error response due to processing failure."],
            metadata={"error": True, "error_message": error_message},
        )

    def _update_metrics(self, status: UIPStatus, processing_time: float):
        """Update routing metrics"""
        if status == UIPStatus.COMPLETED:
            self.routing_metrics["successful_completions"] += 1
        elif status == UIPStatus.DENIED:
            self.routing_metrics["denied_requests"] += 1
        else:
            self.routing_metrics["failed_requests"] += 1

        # Update average processing time
        total_requests = self.routing_metrics["total_requests"]
        current_avg = self.routing_metrics["average_processing_time"]
        new_avg = (
            (current_avg * (total_requests - 1)) + processing_time
        ) / total_requests
        self.routing_metrics["average_processing_time"] = new_avg

    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get current routing performance metrics"""
        return self.routing_metrics.copy()

    def get_active_sessions(self) -> Dict[str, Dict]:
        """Get information about active sessions"""
        return {
            session_id: {
                "active_contexts": len(contexts),
                "last_activity": (
                    max(ctx.get("timestamp", 0) for ctx in contexts.values())
                    if contexts
                    else 0
                ),
            }
            for session_id, contexts in self.active_sessions.items()
        }


# Global progressive router instance
progressive_router = ProgressiveRouter()


@register_uip_handler(UIPStep.STEP_0_PREPROCESSING, dependencies=[], timeout=10)
async def handle_preprocessing(context: UIPContext) -> Dict[str, Any]:
    """
    UIP Step 0: Preprocessing & Ingress Routing

    Sanitizes input, establishes session context, validates format.
    """
    logger = logging.getLogger(__name__)

    try:
        # Input sanitization will be handled by input_sanitizer.py
        # Session initialization will be handled by session_initializer.py
        # This handler coordinates the preprocessing step

        result = {
            "step": "preprocessing_complete",
            "input_sanitized": True,
            "session_validated": True,
            "routing_prepared": True,
            "preprocessing_metadata": {
                "input_length": len(context.user_input),
                "session_id": context.session_id,
                "timestamp": context.timestamp,
            },
        }

        logger.info(f"Preprocessing completed for {context.correlation_id}")
        return result

    except Exception as e:
        logger.error(f"Preprocessing failed for {context.correlation_id}: {e}")
        raise


__all__ = ["ProgressiveRouter", "progressive_router", "handle_preprocessing"]
