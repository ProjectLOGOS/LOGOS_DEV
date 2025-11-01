"""
Protocol Message Formats
=========================

Standardized message formats for UIP and SOP protocol communication.
Ensures consistent data exchange across all protocol layers.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import uuid
import time
import logging
import json


@dataclass
class ProtocolMessage:
    """Base protocol message format"""

    message_id: str
    protocol_type: str  # 'UIP' or 'SOP'
    timestamp: float
    source: str
    destination: Optional[str] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UIPMessage(ProtocolMessage):
    """User Interaction Protocol message"""

    step: Optional[str] = None
    user_input: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.protocol_type = "UIP"


@dataclass
class SOPMessage(ProtocolMessage):
    """System Operations Protocol message"""

    operation: Optional[str] = None
    subsystem: Optional[str] = None
    priority: int = 1  # 1=low, 5=critical

    def __post_init__(self):
        self.protocol_type = "SOP"


@dataclass
class UIPRequest:
    """Standardized UIP request format"""

    session_id: str
    user_input: str
    input_type: str = "text"  # text, voice, image, etc.
    language: str = "en"
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class UIPResponse:
    """Standardized UIP response format"""

    session_id: str
    correlation_id: str
    response_text: str
    confidence_score: float
    alignment_flags: Dict[str, bool] = field(default_factory=dict)
    ontological_vector: Optional[Dict[str, float]] = None
    audit_proof: Optional[str] = None
    disclaimers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    processing_time_ms: Optional[float] = None


@dataclass
class SOPRequest:
    """Standardized SOP request format"""

    operation_id: str
    operation_type: str
    target_subsystem: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout_seconds: int = 30
    correlation_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SOPResponse:
    """Standardized SOP response format"""

    operation_id: str
    status: str  # success, failure, timeout, denied
    result: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0
    subsystem_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MessageValidator:
    """Validates protocol message formats"""

    @staticmethod
    def validate_uip_request(request: UIPRequest) -> Tuple[bool, Optional[str]]:
        """Validate UIP request format"""
        if not request.session_id:
            return False, "Missing session_id"
        if not request.user_input:
            return False, "Missing user_input"
        return True, None

    @staticmethod
    def validate_uip_response(response: UIPResponse) -> Tuple[bool, Optional[str]]:
        """Validate UIP response format"""
        if not response.session_id:
            return False, "Missing session_id"
        if not response.correlation_id:
            return False, "Missing correlation_id"
        if not response.response_text:
            return False, "Missing response_text"
        if not 0 <= response.confidence_score <= 1:
            return False, "Invalid confidence_score range"
        return True, None

    @staticmethod
    def validate_sop_request(request: SOPRequest) -> Tuple[bool, Optional[str]]:
        """Validate SOP request format"""
        if not request.operation_id:
            return False, "Missing operation_id"
        if not request.operation_type:
            return False, "Missing operation_type"
        if not request.target_subsystem:
            return False, "Missing target_subsystem"
        return True, None

    @staticmethod
    def validate_sop_response(response: SOPResponse) -> Tuple[bool, Optional[str]]:
        """Validate SOP response format"""
        if not response.operation_id:
            return False, "Missing operation_id"
        if response.status not in ["success", "failure", "timeout", "denied"]:
            return False, "Invalid status value"
        return True, None


class MessageSerializer:
    """Serializes protocol messages for transmission"""

    @staticmethod
    def serialize(
        message: Union[UIPRequest, UIPResponse, SOPRequest, SOPResponse],
    ) -> str:
        """Serialize message to JSON string"""
        try:
            # Convert dataclass to dict
            if hasattr(message, "__dict__"):
                data = message.__dict__
            else:
                data = message

            return json.dumps(data, default=str, sort_keys=True)
        except Exception as e:
            raise ValueError(f"Failed to serialize message: {e}")

    @staticmethod
    def deserialize(json_str: str, message_type: type) -> Any:
        """Deserialize JSON string to message object"""
        try:
            data = json.loads(json_str)
            return message_type(**data)
        except Exception as e:
            raise ValueError(f"Failed to deserialize message: {e}")


class MessageRouter:
    """Routes protocol messages to appropriate handlers"""

    def __init__(self):
        self.uip_handlers: Dict[str, Callable] = {}
        self.sop_handlers: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)

    def register_uip_handler(self, step: str, handler: Callable):
        """Register UIP step handler"""
        self.uip_handlers[step] = handler
        self.logger.info(f"Registered UIP handler for step: {step}")

    def register_sop_handler(self, operation: str, handler: Callable):
        """Register SOP operation handler"""
        self.sop_handlers[operation] = handler
        self.logger.info(f"Registered SOP handler for operation: {operation}")

    async def route_uip_message(self, message: UIPMessage) -> Optional[Any]:
        """Route UIP message to appropriate handler"""
        if message.step and message.step in self.uip_handlers:
            handler = self.uip_handlers[message.step]
            return await handler(message)
        else:
            self.logger.warning(f"No handler found for UIP step: {message.step}")
            return None

    async def route_sop_message(self, message: SOPMessage) -> Optional[Any]:
        """Route SOP message to appropriate handler"""
        if message.operation and message.operation in self.sop_handlers:
            handler = self.sop_handlers[message.operation]
            return await handler(message)
        else:
            self.logger.warning(
                f"No handler found for SOP operation: {message.operation}"
            )
            return None


# Global message router
message_router = MessageRouter()


__all__ = [
    "ProtocolMessage",
    "UIPMessage",
    "SOPMessage",
    "UIPRequest",
    "UIPResponse",
    "SOPRequest",
    "SOPResponse",
    "MessageValidator",
    "MessageSerializer",
    "MessageRouter",
    "message_router",
]
