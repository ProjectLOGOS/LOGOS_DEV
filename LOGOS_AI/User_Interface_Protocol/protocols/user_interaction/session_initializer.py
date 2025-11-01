"""
Session Initializer - UIP Step 0 Component
===========================================

Session initialization and management for UIP pipeline.
Creates and manages user session contexts with proper isolation and tracking.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from protocols.shared.message_formats import UIPRequest, UIPResponse
from protocols.shared.system_imports import *
from protocols.user_interaction.uip_registry import UIPContext, UIPStatus


@dataclass
class SessionMetadata:
    """Session metadata and tracking information"""

    session_id: str
    user_id: Optional[str] = None
    created_timestamp: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    request_count: int = 0
    total_processing_time: float = 0.0
    session_type: str = "user_interaction"
    client_info: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    rate_limit_tokens: int = 100  # Default rate limit tokens

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
        self.request_count += 1


@dataclass
class SessionConfiguration:
    """Session configuration parameters"""

    max_session_duration: int = 3600  # 1 hour default
    max_requests_per_session: int = 1000
    enable_rate_limiting: bool = True
    rate_limit_window: int = 60  # seconds
    rate_limit_max_requests: int = 60
    enable_audit_logging: bool = True
    session_isolation_level: str = "standard"  # standard, strict, minimal


class SessionStore:
    """Thread-safe session storage and management"""

    def __init__(self):
        self._sessions: Dict[str, SessionMetadata] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

    def create_session(
        self, session_config: SessionConfiguration, **kwargs
    ) -> SessionMetadata:
        """Create new session with configuration"""
        with self._lock:
            session_id = str(uuid.uuid4())

            # Avoid ID collisions (unlikely but safe)
            while session_id in self._sessions:
                session_id = str(uuid.uuid4())

            metadata = SessionMetadata(
                session_id=session_id,
                user_id=kwargs.get("user_id"),
                client_info=kwargs.get("client_info", {}),
                feature_flags=kwargs.get("feature_flags", {}),
                session_type=kwargs.get("session_type", "user_interaction"),
            )

            self._sessions[session_id] = metadata

            self.logger.info(
                f"Created session {session_id} for user {metadata.user_id}"
            )
            return metadata

    def get_session(self, session_id: str) -> Optional[SessionMetadata]:
        """Retrieve session metadata"""
        with self._lock:
            return self._sessions.get(session_id)

    def update_session(self, session_id: str, **updates) -> bool:
        """Update session metadata"""
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.update_activity()

                # Apply updates
                for key, value in updates.items():
                    if hasattr(session, key):
                        setattr(session, key, value)

                return True
            return False

    def remove_session(self, session_id: str) -> bool:
        """Remove session from store"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self.logger.info(f"Removed session {session_id}")
                return True
            return False

    def cleanup_expired_sessions(self, session_config: SessionConfiguration):
        """Remove expired sessions"""
        current_time = time.time()
        expired_sessions = []

        with self._lock:
            for session_id, metadata in self._sessions.items():
                age = current_time - metadata.created_timestamp
                inactive_time = current_time - metadata.last_activity

                if (
                    age > session_config.max_session_duration
                    or inactive_time > session_config.max_session_duration
                    or metadata.request_count > session_config.max_requests_per_session
                ):
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                del self._sessions[session_id]

        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        return len(expired_sessions)

    def get_session_count(self) -> int:
        """Get total active session count"""
        with self._lock:
            return len(self._sessions)


class SessionInitializer:
    """Session initialization and lifecycle management"""

    def __init__(self, config: Optional[SessionConfiguration] = None):
        self.config = config or SessionConfiguration()
        self.session_store = SessionStore()
        self.logger = logging.getLogger(__name__)

        # Start cleanup background task
        self._start_cleanup_task()

    def initialize_session(self, request: UIPRequest) -> SessionMetadata:
        """
        Initialize new session from UIP request

        Args:
            request: UIP request containing session initialization data

        Returns:
            SessionMetadata: Created session metadata
        """
        try:
            # Extract session parameters from request
            client_info = {
                "user_agent": request.metadata.get("user_agent", "unknown"),
                "ip_address": request.metadata.get("ip_address", "unknown"),
                "platform": request.metadata.get("platform", "unknown"),
                "source": request.metadata.get("source", "api"),
            }

            # Create session
            session = self.session_store.create_session(
                self.config,
                user_id=request.metadata.get("user_id"),
                client_info=client_info,
                feature_flags=request.metadata.get("feature_flags", {}),
                session_type=request.metadata.get("session_type", "user_interaction"),
            )

            self.logger.info(f"Session {session.session_id} initialized successfully")
            return session

        except Exception as e:
            self.logger.error(f"Session initialization failed: {e}")
            raise

    def create_uip_context(
        self, request: UIPRequest, session: SessionMetadata
    ) -> UIPContext:
        """
        Create UIP processing context from request and session

        Args:
            request: UIP request to process
            session: Session metadata

        Returns:
            UIPContext: Ready for pipeline processing
        """
        try:
            # Update session activity
            self.session_store.update_session(session.session_id)

            # Create UIP context
            context = UIPContext(
                session_id=session.session_id,
                correlation_id=str(uuid.uuid4()),
                user_input=request.user_input,
                metadata={
                    "session_metadata": session,
                    "request_metadata": request.metadata,
                    "client_info": session.client_info,
                    "feature_flags": session.feature_flags,
                    "processing_start_time": time.time(),
                },
            )

            # Add session tracking to audit trail
            context.audit_trail.append(
                {
                    "timestamp": time.time(),
                    "action": "session_context_created",
                    "session_id": session.session_id,
                    "correlation_id": context.correlation_id,
                    "request_count": session.request_count,
                }
            )

            return context

        except Exception as e:
            self.logger.error(f"UIP context creation failed: {e}")
            raise

    def validate_session_limits(self, session: SessionMetadata) -> Dict[str, Any]:
        """
        Validate session against configured limits

        Returns:
            Dict with validation results
        """
        issues = []

        # Check session age
        age = time.time() - session.created_timestamp
        if age > self.config.max_session_duration:
            issues.append(f"Session expired (age: {age:.1f}s)")

        # Check request count
        if session.request_count > self.config.max_requests_per_session:
            issues.append(f"Request limit exceeded ({session.request_count})")

        # Check rate limiting
        if self.config.enable_rate_limiting:
            # Simple rate limiting check (would be more sophisticated in production)
            recent_activity = time.time() - session.last_activity
            if (
                recent_activity < 1.0
                and session.request_count > self.config.rate_limit_max_requests
            ):
                issues.append("Rate limit exceeded")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "session_age": age,
            "request_count": session.request_count,
        }

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session status"""
        session = self.session_store.get_session(session_id)

        if not session:
            return {"exists": False}

        validation = self.validate_session_limits(session)

        return {
            "exists": True,
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created": session.created_timestamp,
            "last_activity": session.last_activity,
            "request_count": session.request_count,
            "session_type": session.session_type,
            "valid": validation["valid"],
            "issues": validation["issues"],
        }

    def cleanup_session(self, session_id: str) -> bool:
        """Clean up specific session"""
        return self.session_store.remove_session(session_id)

    def _start_cleanup_task(self):
        """Start background cleanup task"""

        def cleanup_worker():
            while True:
                try:
                    time.sleep(300)  # Run every 5 minutes
                    self.session_store.cleanup_expired_sessions(self.config)
                except Exception as e:
                    self.logger.error(f"Session cleanup error: {e}")

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        self.logger.info("Session cleanup task started")


# Global session initializer instance
session_initializer = SessionInitializer()


__all__ = [
    "SessionMetadata",
    "SessionConfiguration",
    "SessionStore",
    "SessionInitializer",
    "session_initializer",
]
