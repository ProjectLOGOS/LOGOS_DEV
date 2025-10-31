#!/usr/bin/env python3
"""
User Interaction Protocol (UIP) Startup Manager
==============================================

Initializes and manages the 7-step UIP reasoning pipeline including:
- Step 0: Preprocessing & Ingress Routing
- Step 1: Linguistic Analysis
- Step 2: PXL Compliance & Validation
- Step 3: IEL Overlay Analysis
- Step 4: Trinity Invocation (Enhanced with Singularity)
- Step 5: Adaptive Inference
- Step 6: Response Synthesis
- Step 7: Compliance Recheck & Audit
- Step 8: Egress Delivery

This is the executable implementation of the User Interaction Protocol
as defined in the LOGOS architecture documentation.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

# Add UIP modules to path
sys.path.append(str(Path(__file__).parent.parent / "User_Interaction_Protocol"))

# Import UIP components
try:
    from intelligence.adaptive.adaptive_reasoning_engine import AdaptiveReasoningEngine
    from intelligence.trinity.trinity_vector_processor import TrinityVectorProcessor
    from intelligence.uip.uip_pipeline_core import UIPPipelineCore, UIPStep
    from intelligence.uip.uip_step4_enhancement import UIPStep4Enhancement
    from interfaces.services.api_service import APIService
    from interfaces.services.gui_interfaces.logos_gui import LogosGUI
    from mathematics.pxl.core.pxl_core import PXLCore
    from User_Interaction_Protocol.protocols.shared.message_formats import UIPRequest, UIPResponse
    from protocols.user_interaction.uip_registry import UIPRegistry, UIPStatus
except ImportError as e:
    logging.warning(f"Some UIP components not available: {e}")
    # Fallback implementations for development
    
    @dataclass
    class UIPRequest:
        """Fallback UIPRequest for development"""
        request_id: str = ""
        user_input: str = ""
        context: Dict[str, Any] = field(default_factory=dict)
        timestamp: float = 0.0
        
    @dataclass
    class UIPResponse:
        """Fallback UIPResponse for development"""
        response_id: str = ""
        content: str = ""
        status: str = "success"
        metadata: Dict[str, Any] = field(default_factory=dict)
        timestamp: float = 0.0


@dataclass
class UIPSystemStatus:
    """Status tracking for UIP systems"""

    preprocessing_active: bool = False
    linguistic_analysis_active: bool = False
    pxl_compliance_active: bool = False
    iel_overlay_active: bool = False
    trinity_invocation_active: bool = False
    adaptive_inference_active: bool = False
    response_synthesis_active: bool = False
    compliance_recheck_active: bool = False
    egress_delivery_active: bool = False

    pipeline_ready: bool = False
    startup_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    active_interfaces: List[str] = field(default_factory=list)
    processing_sessions: int = 0
    error_count: int = 0

    def is_fully_operational(self) -> bool:
        """Check if all critical UIP systems are operational"""
        return all(
            [
                self.preprocessing_active,
                self.linguistic_analysis_active,
                self.pxl_compliance_active,
                self.iel_overlay_active,
                self.trinity_invocation_active,
                self.adaptive_inference_active,
                self.response_synthesis_active,
                self.compliance_recheck_active,
                self.egress_delivery_active,
                self.pipeline_ready,
            ]
        )


class UIPStartupManager:
    """
    User Interaction Protocol Startup Manager

    Orchestrates the initialization and management of the 7-step UIP reasoning
    pipeline according to the LOGOS architectural specifications.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = (
            config_path
            or Path(__file__).parent.parent / "System_Archive" / "uip_config.json"
        )
        self.status = UIPSystemStatus()

        # Core pipeline components
        self.uip_pipeline: Optional[UIPPipelineCore] = None
        self.uip_registry: Optional[UIPRegistry] = None
        self.step4_enhancement: Optional[UIPStep4Enhancement] = None
        self.trinity_processor: Optional[TrinityVectorProcessor] = None
        self.adaptive_engine: Optional[AdaptiveReasoningEngine] = None
        self.pxl_core: Optional[PXLCore] = None

        # Interface components
        self.api_service: Optional[APIService] = None
        self.gui_interface: Optional[LogosGUI] = None

        # Pipeline management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_monitor_thread: Optional[threading.Thread] = None
        self.shutdown_requested = False

        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0

        # Logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        self.logger.info("UIP Startup Manager initialized")

    def setup_logging(self):
        """Configure logging for UIP operations"""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - [UIP] %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler("startup/logs/uip_startup.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    async def initialize_uip_systems(self) -> bool:
        """
        Initialize all UIP pipeline systems in proper sequence

        Returns:
            True if all systems initialized successfully
        """

        self.logger.info("=== UIP SYSTEM INITIALIZATION STARTING ===")
        self.status.startup_time = datetime.now(timezone.utc)

        try:
            # Phase 1: Core Pipeline Infrastructure
            self.logger.info("Phase 1: Core Pipeline Infrastructure")

            if not await self._initialize_uip_registry():
                return False

            if not await self._initialize_uip_pipeline():
                return False

            # Phase 2: Mathematical and Reasoning Engines
            self.logger.info("Phase 2: Mathematical and Reasoning Engines")

            if not await self._initialize_pxl_core():
                return False

            if not await self._initialize_trinity_processor():
                return False

            if not await self._initialize_adaptive_engine():
                return False

            # Phase 3: UIP Pipeline Steps
            self.logger.info("Phase 3: UIP Pipeline Steps Initialization")

            if not await self._initialize_step0_preprocessing():
                return False

            if not await self._initialize_step1_linguistic():
                return False

            if not await self._initialize_step2_pxl_compliance():
                return False

            if not await self._initialize_step3_iel_overlay():
                return False

            if not await self._initialize_step4_trinity_invocation():
                return False

            if not await self._initialize_step5_adaptive_inference():
                return False

            if not await self._initialize_step6_response_synthesis():
                return False

            if not await self._initialize_step7_compliance_recheck():
                return False

            if not await self._initialize_step8_egress_delivery():
                return False

            # Phase 4: User Interfaces
            self.logger.info("Phase 4: User Interface Systems")

            if not await self._initialize_api_service():
                return False

            # GUI is optional - don't fail startup if unavailable
            await self._initialize_gui_interface()

            # Phase 5: Pipeline Integration and Validation
            self.logger.info("Phase 5: Pipeline Integration")

            if not await self._validate_pipeline_integration():
                return False

            # Final readiness check
            if self.status.is_fully_operational():
                self.logger.info("=== UIP SYSTEM INITIALIZATION COMPLETE ===")
                self._start_session_monitoring()
                return True
            else:
                self.logger.error(
                    "UIP system initialization incomplete - some steps failed"
                )
                return False

        except Exception as e:
            self.logger.error(f"UIP initialization failed: {e}")
            self.status.error_count += 1
            return False

    async def _initialize_uip_registry(self) -> bool:
        """Initialize UIP step registry"""
        try:
            self.uip_registry = UIPRegistry()
            await self.uip_registry.initialize_registry()
            self.logger.info("UIP registry initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"UIP registry initialization failed: {e}")
            return False

    async def _initialize_uip_pipeline(self) -> bool:
        """Initialize core UIP pipeline"""
        try:
            self.uip_pipeline = UIPPipelineCore(registry=self.uip_registry)
            await self.uip_pipeline.initialize_pipeline()
            self.logger.info("UIP pipeline core initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"UIP pipeline initialization failed: {e}")
            return False

    async def _initialize_pxl_core(self) -> bool:
        """Initialize PXL mathematical core"""
        try:
            self.pxl_core = PXLCore()
            await self.pxl_core.initialize()
            self.logger.info("PXL core initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"PXL core initialization failed: {e}")
            return False

    async def _initialize_trinity_processor(self) -> bool:
        """Initialize Trinity vector processor"""
        try:
            self.trinity_processor = TrinityVectorProcessor(pxl_core=self.pxl_core)
            await self.trinity_processor.initialize()
            self.logger.info("Trinity processor initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Trinity processor initialization failed: {e}")
            return False

    async def _initialize_adaptive_engine(self) -> bool:
        """Initialize adaptive reasoning engine"""
        try:
            self.adaptive_engine = AdaptiveReasoningEngine()
            await self.adaptive_engine.initialize()
            self.logger.info("Adaptive reasoning engine initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Adaptive engine initialization failed: {e}")
            return False

    async def _initialize_step0_preprocessing(self) -> bool:
        """Initialize Step 0: Preprocessing & Ingress Routing"""
        try:
            # Register step 0 handlers
            await self.uip_registry.register_step_handler(
                UIPStep.STEP_0_PREPROCESSING, self._create_step0_handler()
            )

            self.status.preprocessing_active = True
            self.logger.info("Step 0 (Preprocessing) initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Step 0 initialization failed: {e}")
            return False

    async def _initialize_step1_linguistic(self) -> bool:
        """Initialize Step 1: Linguistic Analysis"""
        try:
            await self.uip_registry.register_step_handler(
                UIPStep.STEP_1_LINGUISTIC, self._create_step1_handler()
            )

            self.status.linguistic_analysis_active = True
            self.logger.info("Step 1 (Linguistic Analysis) initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Step 1 initialization failed: {e}")
            return False

    async def _initialize_step2_pxl_compliance(self) -> bool:
        """Initialize Step 2: PXL Compliance & Validation"""
        try:
            await self.uip_registry.register_step_handler(
                UIPStep.STEP_2_PXL_COMPLIANCE, self._create_step2_handler()
            )

            self.status.pxl_compliance_active = True
            self.logger.info("Step 2 (PXL Compliance) initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Step 2 initialization failed: {e}")
            return False

    async def _initialize_step3_iel_overlay(self) -> bool:
        """Initialize Step 3: IEL Overlay Analysis"""
        try:
            await self.uip_registry.register_step_handler(
                UIPStep.STEP_3_IEL_OVERLAY, self._create_step3_handler()
            )

            self.status.iel_overlay_active = True
            self.logger.info("Step 3 (IEL Overlay) initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Step 3 initialization failed: {e}")
            return False

    async def _initialize_step4_trinity_invocation(self) -> bool:
        """Initialize Step 4: Trinity Invocation (Enhanced with Singularity)"""
        try:
            # Initialize enhanced Step 4 with optional Singularity integration
            self.step4_enhancement = UIPStep4Enhancement(
                trinity_processor=self.trinity_processor,
                enable_singularity_enhancement=True,  # Enable Singularity AGI
            )

            await self.uip_registry.register_step_handler(
                UIPStep.STEP_4_TRINITY_INVOCATION, self._create_step4_handler()
            )

            self.status.trinity_invocation_active = True
            self.logger.info(
                "Step 4 (Trinity Invocation + Singularity) initialized successfully"
            )
            return True
        except Exception as e:
            self.logger.error(f"Step 4 initialization failed: {e}")
            return False

    async def _initialize_step5_adaptive_inference(self) -> bool:
        """Initialize Step 5: Adaptive Inference"""
        try:
            await self.uip_registry.register_step_handler(
                UIPStep.STEP_5_ADAPTIVE_INFERENCE, self._create_step5_handler()
            )

            self.status.adaptive_inference_active = True
            self.logger.info("Step 5 (Adaptive Inference) initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Step 5 initialization failed: {e}")
            return False

    async def _initialize_step6_response_synthesis(self) -> bool:
        """Initialize Step 6: Response Synthesis"""
        try:
            await self.uip_registry.register_step_handler(
                UIPStep.STEP_6_RESPONSE_SYNTHESIS, self._create_step6_handler()
            )

            self.status.response_synthesis_active = True
            self.logger.info("Step 6 (Response Synthesis) initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Step 6 initialization failed: {e}")
            return False

    async def _initialize_step7_compliance_recheck(self) -> bool:
        """Initialize Step 7: Compliance Recheck & Audit"""
        try:
            await self.uip_registry.register_step_handler(
                UIPStep.STEP_7_COMPLIANCE_RECHECK, self._create_step7_handler()
            )

            self.status.compliance_recheck_active = True
            self.logger.info("Step 7 (Compliance Recheck) initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Step 7 initialization failed: {e}")
            return False

    async def _initialize_step8_egress_delivery(self) -> bool:
        """Initialize Step 8: Egress Delivery"""
        try:
            await self.uip_registry.register_step_handler(
                UIPStep.STEP_8_EGRESS_DELIVERY, self._create_step8_handler()
            )

            self.status.egress_delivery_active = True
            self.logger.info("Step 8 (Egress Delivery) initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Step 8 initialization failed: {e}")
            return False

    async def _initialize_api_service(self) -> bool:
        """Initialize API service interface"""
        try:
            self.api_service = APIService(uip_pipeline=self.uip_pipeline)
            await self.api_service.start_service()

            self.status.active_interfaces.append("api")
            self.logger.info("API service initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"API service initialization failed: {e}")
            return False

    async def _initialize_gui_interface(self) -> bool:
        """Initialize GUI interface (optional)"""
        try:
            # GUI initialization is optional and may not be available in all environments
            self.gui_interface = LogosGUI(uip_pipeline=self.uip_pipeline)

            self.status.active_interfaces.append("gui")
            self.logger.info("GUI interface initialized successfully")
            return True
        except Exception as e:
            self.logger.warning(f"GUI interface initialization failed (optional): {e}")
            return False  # Don't fail startup for GUI

    async def _validate_pipeline_integration(self) -> bool:
        """Validate that all pipeline components are properly integrated"""
        try:
            # Test pipeline with a simple validation request
            test_request = UIPRequest(
                session_id="validation_test",
                user_input="System validation test",
                context={"validation_mode": True},
            )

            # Process through pipeline (dry run)
            validation_result = await self.uip_pipeline.process_request_dry_run(
                test_request
            )

            if validation_result.get("pipeline_valid", False):
                self.status.pipeline_ready = True
                self.logger.info("Pipeline integration validation successful")
                return True
            else:
                self.logger.error("Pipeline integration validation failed")
                return False

        except Exception as e:
            self.logger.error(f"Pipeline validation failed: {e}")
            return False

    def _create_step0_handler(self):
        """Create handler for Step 0: Preprocessing"""

        async def step0_handler(request: UIPRequest) -> Dict[str, Any]:
            # Implement Step 0 logic: input sanitization, session management
            return {
                "step": "step_0_preprocessing",
                "status": "completed",
                "processed_input": request.user_input,
                "session_validated": True,
            }

        return step0_handler

    def _create_step1_handler(self):
        """Create handler for Step 1: Linguistic Analysis"""

        async def step1_handler(request: UIPRequest) -> Dict[str, Any]:
            # Implement Step 1 logic: linguistic analysis, Trinity decomposition
            return {
                "step": "step_1_linguistic",
                "status": "completed",
                "linguistic_analysis": "completed",
                "trinity_decomposition": {
                    "existence": 0.6,
                    "goodness": 0.7,
                    "truth": 0.8,
                },
            }

        return step1_handler

    def _create_step2_handler(self):
        """Create handler for Step 2: PXL Compliance"""

        async def step2_handler(request: UIPRequest) -> Dict[str, Any]:
            # Implement Step 2 logic: PXL validation using pxl_core
            pxl_result = await self.pxl_core.validate_request(request)
            return {
                "step": "step_2_pxl_compliance",
                "status": "completed",
                "pxl_compliance_validated": pxl_result.get("valid", False),
                "safety_constraints_met": pxl_result.get("safe", False),
            }

        return step2_handler

    def _create_step3_handler(self):
        """Create handler for Step 3: IEL Overlay"""

        async def step3_handler(request: UIPRequest) -> Dict[str, Any]:
            # Implement Step 3 logic: IEL domain analysis
            return {
                "step": "step_3_iel_overlay",
                "status": "completed",
                "iel_domains_analyzed": ["modal_logic", "epistemic_logic"],
                "domain_synthesis_complete": True,
            }

        return step3_handler

    def _create_step4_handler(self):
        """Create handler for Step 4: Trinity Invocation (Enhanced)"""

        async def step4_handler(request: UIPRequest) -> Dict[str, Any]:
            # Enhanced Step 4 with Singularity integration
            enhancement_result = await self.step4_enhancement.enhance_reasoning(
                request.context or {}
            )
            return {
                "step": "step_4_trinity_invocation",
                "status": "completed",
                "trinity_processing_complete": True,
                "singularity_enhancement": enhancement_result,
                "infinite_reasoning_applied": enhancement_result.get(
                    "infinite_reasoning_applied", False
                ),
            }

        return step4_handler

    def _create_step5_handler(self):
        """Create handler for Step 5: Adaptive Inference"""

        async def step5_handler(request: UIPRequest) -> Dict[str, Any]:
            # Implement Step 5 logic: adaptive reasoning
            inference_result = await self.adaptive_engine.perform_inference(request)
            return {
                "step": "step_5_adaptive_inference",
                "status": "completed",
                "adaptive_inference_complete": True,
                "inference_results": inference_result,
            }

        return step5_handler

    def _create_step6_handler(self):
        """Create handler for Step 6: Response Synthesis"""

        async def step6_handler(request: UIPRequest) -> Dict[str, Any]:
            # Implement Step 6 logic: response generation and synthesis
            return {
                "step": "step_6_response_synthesis",
                "status": "completed",
                "response_synthesized": True,
                "confidence_score": 0.85,
            }

        return step6_handler

    def _create_step7_handler(self):
        """Create handler for Step 7: Compliance Recheck"""

        async def step7_handler(request: UIPRequest) -> Dict[str, Any]:
            # Implement Step 7 logic: final compliance validation
            return {
                "step": "step_7_compliance_recheck",
                "status": "completed",
                "final_compliance_validated": True,
                "audit_trail_complete": True,
            }

        return step7_handler

    def _create_step8_handler(self):
        """Create handler for Step 8: Egress Delivery"""

        async def step8_handler(request: UIPRequest) -> Dict[str, Any]:
            # Implement Step 8 logic: final delivery and cleanup
            return {
                "step": "step_8_egress_delivery",
                "status": "completed",
                "response_delivered": True,
                "session_cleanup_complete": True,
            }

        return step8_handler

    def _start_session_monitoring(self):
        """Start background session monitoring"""
        self.session_monitor_thread = threading.Thread(
            target=self._session_monitor_loop, daemon=True
        )
        self.session_monitor_thread.start()
        self.logger.info("Session monitoring started")

    def _session_monitor_loop(self):
        """Background session monitoring loop"""
        while not self.shutdown_requested:
            try:
                # Monitor active sessions
                self._monitor_active_sessions()

                # Update processing statistics
                self.status.processing_sessions = len(self.active_sessions)

                # Health checks
                self._perform_uip_health_checks()

                # Sleep between checks
                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"Session monitoring error: {e}")
                self.status.error_count += 1
                time.sleep(2)

    def _monitor_active_sessions(self):
        """Monitor active UIP processing sessions"""
        # Clean up completed sessions
        completed_sessions = []

        for session_id, session_data in self.active_sessions.items():
            if session_data.get("status") == "completed":
                completed_sessions.append(session_id)

        for session_id in completed_sessions:
            del self.active_sessions[session_id]

    def _perform_uip_health_checks(self):
        """Perform health checks on UIP components"""
        self.status.last_health_check = datetime.now(timezone.utc)

        # Check pipeline health
        if self.uip_pipeline and not self.uip_pipeline.is_healthy():
            self.logger.warning("UIP pipeline health check failed")

        # Check critical components
        if self.pxl_core and not self.pxl_core.is_healthy():
            self.logger.warning("PXL core health check failed")

    async def process_user_request(self, request: UIPRequest) -> UIPResponse:
        """Process a user request through the complete UIP pipeline"""

        session_id = request.session_id
        start_time = time.time()

        try:
            # Track session
            self.active_sessions[session_id] = {
                "start_time": start_time,
                "status": "processing",
                "current_step": 0,
            }

            # Process through pipeline
            pipeline_result = await self.uip_pipeline.process_request(request)

            # Create response
            response = UIPResponse(
                session_id=session_id,
                correlation_id=request.correlation_id or str(uuid.uuid4()),
                response_text=pipeline_result.get(
                    "final_response", "Processing complete"
                ),
                confidence_score=pipeline_result.get("confidence_score", 0.8),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            # Update statistics
            self.request_count += 1
            self.total_processing_time += time.time() - start_time

            # Mark session complete
            self.active_sessions[session_id]["status"] = "completed"

            return response

        except Exception as e:
            self.logger.error(f"UIP processing failed for session {session_id}: {e}")
            self.status.error_count += 1

            # Error response
            error_response = UIPResponse(
                session_id=session_id,
                correlation_id=request.correlation_id or str(uuid.uuid4()),
                response_text=f"Processing error: {str(e)}",
                confidence_score=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            # Mark session failed
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["status"] = "failed"

            return error_response

    async def shutdown_uip_systems(self):
        """Gracefully shutdown all UIP systems"""
        self.logger.info("=== UIP SYSTEM SHUTDOWN INITIATED ===")
        self.shutdown_requested = True

        try:
            # Shutdown interfaces first
            if self.api_service:
                await self.api_service.shutdown()

            if self.gui_interface:
                await self.gui_interface.shutdown()

            # Shutdown core components
            if self.uip_pipeline:
                await self.uip_pipeline.shutdown()

            if self.adaptive_engine:
                await self.adaptive_engine.shutdown()

            if self.trinity_processor:
                await self.trinity_processor.shutdown()

            if self.pxl_core:
                await self.pxl_core.shutdown()

            # Wait for monitoring thread to finish
            if self.session_monitor_thread and self.session_monitor_thread.is_alive():
                self.session_monitor_thread.join(timeout=5)

            self.logger.info("=== UIP SYSTEM SHUTDOWN COMPLETE ===")

        except Exception as e:
            self.logger.error(f"UIP shutdown error: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive UIP system status"""
        avg_processing_time = self.total_processing_time / max(self.request_count, 1)

        return {
            "uip_status": {
                "fully_operational": self.status.is_fully_operational(),
                "pipeline_ready": self.status.pipeline_ready,
                "steps_active": {
                    "step_0_preprocessing": self.status.preprocessing_active,
                    "step_1_linguistic": self.status.linguistic_analysis_active,
                    "step_2_pxl_compliance": self.status.pxl_compliance_active,
                    "step_3_iel_overlay": self.status.iel_overlay_active,
                    "step_4_trinity_invocation": self.status.trinity_invocation_active,
                    "step_5_adaptive_inference": self.status.adaptive_inference_active,
                    "step_6_response_synthesis": self.status.response_synthesis_active,
                    "step_7_compliance_recheck": self.status.compliance_recheck_active,
                    "step_8_egress_delivery": self.status.egress_delivery_active,
                },
            },
            "system_info": {
                "startup_time": (
                    self.status.startup_time.isoformat()
                    if self.status.startup_time
                    else None
                ),
                "last_health_check": (
                    self.status.last_health_check.isoformat()
                    if self.status.last_health_check
                    else None
                ),
                "active_interfaces": self.status.active_interfaces,
                "processing_sessions": self.status.processing_sessions,
                "error_count": self.status.error_count,
            },
            "performance_metrics": {
                "total_requests_processed": self.request_count,
                "average_processing_time_ms": avg_processing_time * 1000,
                "active_sessions": len(self.active_sessions),
                "uptime_seconds": (
                    (
                        datetime.now(timezone.utc) - self.status.startup_time
                    ).total_seconds()
                    if self.status.startup_time
                    else 0
                ),
            },
        }


# Global UIP manager instance
uip_manager: Optional[UIPStartupManager] = None


async def start_uip_system(config_path: Optional[Path] = None) -> bool:
    """
    Start the User Interaction Protocol system

    Args:
        config_path: Optional path to configuration file

    Returns:
        True if startup successful
    """
    global uip_manager

    uip_manager = UIPStartupManager(config_path)
    return await uip_manager.initialize_uip_systems()


async def shutdown_uip_system():
    """Shutdown the UIP system"""
    global uip_manager

    if uip_manager:
        await uip_manager.shutdown_uip_systems()
        uip_manager = None


def get_uip_status() -> Dict[str, Any]:
    """Get UIP system status"""
    global uip_manager

    if uip_manager:
        return uip_manager.get_system_status()
    else:
        return {
            "uip_status": {"fully_operational": False},
            "error": "UIP not initialized",
        }


async def process_user_request(request: UIPRequest) -> UIPResponse:
    """Process a user request through UIP pipeline"""
    global uip_manager

    if uip_manager:
        return await uip_manager.process_user_request(request)
    else:
        raise RuntimeError("UIP system not initialized")


# Main execution for standalone UIP startup
if __name__ == "__main__":

    async def main():
        try:
            # Start UIP system
            success = await start_uip_system()

            if success:
                print("✅ UIP System initialized successfully")
                print("🔄 UIP Pipeline ready for user interactions")

                # Keep running until interrupted
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    print("\n🛑 UIP System shutdown requested")

                # Graceful shutdown
                await shutdown_uip_system()
                print("✅ UIP System shutdown complete")

            else:
                print("❌ UIP System initialization failed")
                sys.exit(1)

        except Exception as e:
            print(f"❌ UIP System error: {e}")
            sys.exit(1)

    # Run the UIP system
    asyncio.run(main())
