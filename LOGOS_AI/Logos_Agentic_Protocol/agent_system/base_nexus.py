#!/usr/bin/env python3
"""
Base Nexus Infrastructure
========================

Provides the foundation for all protocol nexus layers in LOGOS.
Each protocol (SOP, AGP, UIP) will have its own nexus that inherits
from this base class and implements protocol-specific functionality.

Key Features:
- Agent authentication and type classification
- Standardized request/response handling
- Security boundary enforcement
- Performance monitoring and logging
- Error handling and recovery
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Classification of agents interacting with LOGOS"""
    SYSTEM_AGENT = "system_internal"      # LOGOS internal SystemAgent
    EXTERIOR_AGENT = "exterior_external"  # External users, APIs, services


class ProtocolType(Enum):
    """Available LOGOS protocols"""
    SOP = "system_operations_protocol"
    AGP = "advanced_general_protocol"
    UIP = "user_interaction_protocol"


class NexusLifecycle(Enum):
    """Nexus operational states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DORMANT = "dormant"
    TESTING = "testing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class AgentRequest:
    """Standardized agent request format"""
    
    def __init__(
        self,
        agent_id: str,
        operation: str,
        payload: Dict[str, Any],
        agent_type: Optional[AgentType] = None,
        authentication: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.request_id = str(uuid.uuid4())
        self.agent_id = agent_id
        self.operation = operation
        self.payload = payload
        self.agent_type = agent_type
        self.authentication = authentication or {}
        self.context = context or {}
        self.timestamp = datetime.now(timezone.utc)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "operation": self.operation,
            "payload": self.payload,
            "agent_type": self.agent_type.value if self.agent_type else None,
            "authentication": self.authentication,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentRequest':
        """Create from dictionary format"""
        agent_type = None
        if data.get("agent_type"):
            agent_type = AgentType(data["agent_type"])
            
        return cls(
            agent_id=data["agent_id"],
            operation=data["operation"],
            payload=data["payload"],
            agent_type=agent_type,
            authentication=data.get("authentication", {}),
            context=data.get("context", {})
        )


class NexusResponse:
    """Standardized nexus response format"""
    
    def __init__(
        self,
        request_id: str,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.response_id = str(uuid.uuid4())
        self.request_id = request_id
        self.success = success
        self.data = data or {}
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "response_id": self.response_id,
            "request_id": self.request_id,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class BaseNexus(ABC):
    """
    Abstract base class for all protocol nexus layers
    
    Each protocol nexus inherits from this class and implements
    protocol-specific functionality while maintaining consistent
    agent communication patterns.
    """
    
    def __init__(
        self,
        protocol_type: ProtocolType,
        nexus_name: str,
        always_active: bool = True
    ):
        self.protocol_type = protocol_type
        self.nexus_name = nexus_name
        self.always_active = always_active
        self.lifecycle_state = NexusLifecycle.INITIALIZING
        
        # Agent management
        self.authenticated_agents: Dict[str, AgentType] = {}
        self.active_requests: Dict[str, AgentRequest] = {}
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.performance_metrics = {}
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{nexus_name}")
        
    async def initialize(self) -> bool:
        """
        Initialize the nexus layer
        
        This handles common initialization tasks and calls
        protocol-specific initialization.
        """
        try:
            self.logger.info(f"🚀 Initializing {self.nexus_name}")
            
            # Common initialization
            self.lifecycle_state = NexusLifecycle.INITIALIZING
            
            # Protocol-specific initialization
            success = await self._protocol_specific_initialization()
            
            if success:
                if self.always_active:
                    self.lifecycle_state = NexusLifecycle.ACTIVE
                    self.logger.info(f"✅ {self.nexus_name} - Active (Always-On)")
                else:
                    self.lifecycle_state = NexusLifecycle.DORMANT
                    self.logger.info(f"✅ {self.nexus_name} - Dormant (On-Demand)")
                    
                return True
            else:
                self.lifecycle_state = NexusLifecycle.ERROR
                return False
                
        except Exception as e:
            self.logger.error(f"❌ {self.nexus_name} initialization failed: {e}")
            self.lifecycle_state = NexusLifecycle.ERROR
            return False
    
    @abstractmethod
    async def _protocol_specific_initialization(self) -> bool:
        """Protocol-specific initialization logic"""
        pass
    
    async def process_agent_request(self, request: AgentRequest) -> NexusResponse:
        """
        Main entry point for agent requests
        
        This method handles the complete request lifecycle:
        1. Agent authentication and type classification
        2. Security validation
        3. Request routing to protocol core
        4. Response formatting and delivery
        """
        try:
            self.logger.info(f"📨 Processing request from agent: {request.agent_id}")
            
            # Track request
            self.active_requests[request.request_id] = request
            self.request_count += 1
            
            # Step 1: Agent Authentication and Classification
            agent_type = await self._authenticate_and_classify_agent(request)
            
            if not agent_type:
                return NexusResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Agent authentication failed"
                )
            
            # Update request with classified agent type
            request.agent_type = agent_type
            
            # Step 2: Security Validation
            security_check = await self._validate_security_boundaries(request)
            
            if not security_check.get("valid", False):
                return NexusResponse(
                    request_id=request.request_id,
                    success=False,
                    error=f"Security validation failed: {security_check.get('reason', 'Unknown')}"
                )
            
            # Step 3: Lifecycle Management (for on-demand protocols)
            if not self.always_active:
                await self._activate_protocol_if_needed()
            
            # Step 4: Protocol-Specific Processing
            core_response = await self._route_to_protocol_core(request)
            
            # Step 5: Response Formatting
            response = NexusResponse(
                request_id=request.request_id,
                success=core_response.get("success", False),
                data=core_response.get("data", {}),
                error=core_response.get("error"),
                metadata={
                    "agent_type": agent_type.value,
                    "protocol": self.protocol_type.value,
                    "nexus": self.nexus_name,
                    "processing_time_ms": self._calculate_processing_time(request)
                }
            )
            
            # Step 6: Cleanup
            if not self.always_active:
                await self._deactivate_protocol_if_needed()
            
            # Remove from active tracking
            self.active_requests.pop(request.request_id, None)
            
            return response
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"❌ Request processing error: {e}")
            
            return NexusResponse(
                request_id=request.request_id,
                success=False,
                error=f"Internal processing error: {str(e)}"
            )
    
    async def _authenticate_and_classify_agent(self, request: AgentRequest) -> Optional[AgentType]:
        """
        Authenticate agent and classify as System or Exterior
        
        This is the critical function that determines agent type
        and establishes security context.
        """
        
        # Check if agent is already authenticated
        if request.agent_id in self.authenticated_agents:
            return self.authenticated_agents[request.agent_id]
        
        # Perform agent classification
        agent_type = await self._classify_agent_type(request)
        
        if agent_type:
            # Cache authentication
            self.authenticated_agents[request.agent_id] = agent_type
            self.logger.info(f"🔐 Agent {request.agent_id} classified as: {agent_type.value}")
            
        return agent_type
    
    async def _classify_agent_type(self, request: AgentRequest) -> Optional[AgentType]:
        """
        Classify agent as System Agent or Exterior Agent
        
        This is protocol-specific and will be overridden by
        protocol nexus implementations.
        """
        
        # Default classification logic
        if request.agent_id.startswith("SYSTEM_AGENT_"):
            return AgentType.SYSTEM_AGENT
        else:
            return AgentType.EXTERIOR_AGENT
    
    async def _validate_security_boundaries(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Validate security boundaries based on agent type
        
        Returns validation result with success/failure and reason.
        """
        
        if not request.agent_type:
            return {"valid": False, "reason": "Agent type not classified"}
        
        # Protocol-specific security validation
        return await self._protocol_specific_security_validation(request)
    
    @abstractmethod
    async def _protocol_specific_security_validation(self, request: AgentRequest) -> Dict[str, Any]:
        """Protocol-specific security validation logic"""
        pass
    
    async def _activate_protocol_if_needed(self) -> None:
        """Activate protocol core if in dormant state"""
        if self.lifecycle_state == NexusLifecycle.DORMANT:
            self.logger.info(f"🔄 Activating {self.nexus_name} protocol core")
            self.lifecycle_state = NexusLifecycle.ACTIVE
            await self._protocol_specific_activation()
    
    async def _deactivate_protocol_if_needed(self) -> None:
        """Return protocol core to dormant state if no active requests"""
        if (not self.always_active and 
            len(self.active_requests) == 0 and 
            self.lifecycle_state == NexusLifecycle.ACTIVE):
            
            self.logger.info(f"💤 Returning {self.nexus_name} to dormant state")
            self.lifecycle_state = NexusLifecycle.DORMANT
            await self._protocol_specific_deactivation()
    
    @abstractmethod
    async def _protocol_specific_activation(self) -> None:
        """Protocol-specific activation logic"""
        pass
    
    @abstractmethod
    async def _protocol_specific_deactivation(self) -> None:
        """Protocol-specific deactivation logic"""
        pass
    
    @abstractmethod
    async def _route_to_protocol_core(self, request: AgentRequest) -> Dict[str, Any]:
        """Route request to protocol core for processing"""
        pass
    
    def _calculate_processing_time(self, request: AgentRequest) -> float:
        """Calculate request processing time in milliseconds"""
        processing_time = (datetime.now(timezone.utc) - request.timestamp).total_seconds() * 1000
        return round(processing_time, 2)
    
    async def get_nexus_status(self) -> Dict[str, Any]:
        """Get current nexus status and metrics"""
        return {
            "nexus_name": self.nexus_name,
            "protocol_type": self.protocol_type.value,
            "lifecycle_state": self.lifecycle_state.value,
            "always_active": self.always_active,
            "authenticated_agents": len(self.authenticated_agents),
            "active_requests": len(self.active_requests),
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "performance_metrics": self.performance_metrics
        }
    
    async def run_smoke_test(self) -> Dict[str, Any]:
        """
        Run comprehensive smoke test
        
        This is particularly important for UIP nexus which needs
        to validate agent distinction capabilities.
        """
        self.logger.info(f"🧪 Running smoke test for {self.nexus_name}")
        
        try:
            self.lifecycle_state = NexusLifecycle.TESTING
            
            # Basic functionality test
            basic_test = await self._test_basic_functionality()
            
            # Agent classification test
            agent_test = await self._test_agent_classification()
            
            # Security boundary test
            security_test = await self._test_security_boundaries()
            
            # Protocol-specific tests
            protocol_test = await self._protocol_specific_smoke_test()
            
            # Aggregate results
            all_tests_passed = all([
                basic_test.get("passed", False),
                agent_test.get("passed", False),
                security_test.get("passed", False),
                protocol_test.get("passed", False)
            ])
            
            smoke_test_result = {
                "nexus": self.nexus_name,
                "overall_success": all_tests_passed,
                "tests": {
                    "basic_functionality": basic_test,
                    "agent_classification": agent_test,
                    "security_boundaries": security_test,
                    "protocol_specific": protocol_test
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Return to appropriate lifecycle state
            if self.always_active:
                self.lifecycle_state = NexusLifecycle.ACTIVE
            else:
                self.lifecycle_state = NexusLifecycle.DORMANT
            
            if all_tests_passed:
                self.logger.info(f"✅ {self.nexus_name} smoke test: PASSED")
            else:
                self.logger.warning(f"⚠️ {self.nexus_name} smoke test: FAILED")
            
            return smoke_test_result
            
        except Exception as e:
            self.logger.error(f"❌ Smoke test error: {e}")
            self.lifecycle_state = NexusLifecycle.ERROR
            
            return {
                "nexus": self.nexus_name,
                "overall_success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic nexus functionality"""
        try:
            # Test request processing pipeline
            test_request = AgentRequest(
                agent_id="TEST_AGENT",
                operation="health_check",
                payload={"test": True}
            )
            
            # This should go through the full pipeline
            response = await self.process_agent_request(test_request)
            
            return {
                "passed": response.success,
                "details": "Basic request processing pipeline"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    async def _test_agent_classification(self) -> Dict[str, Any]:
        """Test agent type classification"""
        try:
            # Test System Agent classification
            system_request = AgentRequest(
                agent_id="SYSTEM_AGENT_TEST",
                operation="classification_test",
                payload={"agent_type_test": "system"}
            )
            
            system_type = await self._classify_agent_type(system_request)
            
            # Test Exterior Agent classification
            exterior_request = AgentRequest(
                agent_id="USER_AGENT_TEST",
                operation="classification_test", 
                payload={"agent_type_test": "exterior"}
            )
            
            exterior_type = await self._classify_agent_type(exterior_request)
            
            system_correct = system_type == AgentType.SYSTEM_AGENT
            exterior_correct = exterior_type == AgentType.EXTERIOR_AGENT
            
            return {
                "passed": system_correct and exterior_correct,
                "details": {
                    "system_agent_classification": system_correct,
                    "exterior_agent_classification": exterior_correct
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    async def _test_security_boundaries(self) -> Dict[str, Any]:
        """Test security boundary enforcement"""
        try:
            # This will be overridden by protocol-specific implementations
            return {
                "passed": True,
                "details": "Base security boundary test"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    @abstractmethod
    async def _protocol_specific_smoke_test(self) -> Dict[str, Any]:
        """Protocol-specific smoke test implementation"""
        pass


# Utility functions for nexus system

def create_agent_request(
    agent_id: str,
    operation: str,
    payload: Dict[str, Any],
    **kwargs
) -> AgentRequest:
    """Utility function to create standardized agent requests"""
    return AgentRequest(
        agent_id=agent_id,
        operation=operation,
        payload=payload,
        **kwargs
    )


async def validate_nexus_system(nexus_list: List[BaseNexus]) -> Dict[str, Any]:
    """Validate complete nexus system functionality"""
    
    logger.info("🔍 Validating complete nexus system")
    
    validation_results = {}
    overall_success = True
    
    for nexus in nexus_list:
        nexus_result = await nexus.run_smoke_test()
        validation_results[nexus.nexus_name] = nexus_result
        
        if not nexus_result.get("overall_success", False):
            overall_success = False
    
    return {
        "system_validation_success": overall_success,
        "nexus_results": validation_results,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }