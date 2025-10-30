"""
Audit Logger - System Event Auditing

This module provides comprehensive audit logging capabilities for the LOGOS
adaptive inference system, ensuring all critical operations are tracked
for security, debugging, and compliance purposes.

⧟ Identity constraint: Event integrity via immutable logging
⇌ Balance constraint: Detailed logging vs performance impact
⟹ Causal constraint: Event causality chains preserved
⩪ Equivalence constraint: Consistent event format across components
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
from enum import Enum

class AuditLevel(Enum):
    """Audit event severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"

@dataclass
class AuditEvent:
    """Structured audit event."""
    event_type: str
    timestamp: str
    level: AuditLevel
    component: str
    payload: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None

class AuditLogger:
    """Thread-safe audit logger with file and structured output."""
    
    def __init__(self, 
                 log_path: Optional[Path] = None,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        self.log_path = log_path or Path("audit/step5.log")
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self._lock = threading.Lock()
        self._ensure_log_directory()
        
        # Configure Python logger
        self.logger = logging.getLogger(f"audit.{__name__}")
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_path)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _ensure_log_directory(self):
        """Ensure audit log directory exists."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_event(self, event_type: str, 
                  payload: Optional[Union[Dict[str, Any], str]] = None,
                  level: AuditLevel = AuditLevel.INFO,
                  component: str = "adaptive_inference") -> bool:
        """
        Log an audit event.
        
        Args:
            event_type: Type/name of the event
            payload: Event payload data (dict or string)
            level: Audit severity level
            component: Component generating the event
            
        Returns:
            Boolean indicating logging success
        """
        try:
            with self._lock:
                # Convert string payload to dict
                if isinstance(payload, str):
                    payload = {"message": payload}
                elif payload is None:
                    payload = {}
                
                # Create structured audit event
                audit_event = AuditEvent(
                    event_type=event_type,
                    timestamp=datetime.now().isoformat(),
                    level=level,
                    component=component,
                    payload=payload,
                    correlation_id=self._generate_correlation_id()
                )
                
                # Log to structured file
                self._write_structured_event(audit_event)
                
                # Log to Python logger  
                log_message = f"{event_type}: {json.dumps(payload, default=str)[:200]}"
                if level == AuditLevel.DEBUG:
                    self.logger.debug(log_message)
                elif level == AuditLevel.INFO:
                    self.logger.info(log_message)
                elif level == AuditLevel.WARNING:
                    self.logger.warning(log_message)
                elif level in [AuditLevel.ERROR, AuditLevel.CRITICAL, AuditLevel.SECURITY]:
                    self.logger.error(log_message)
                
                return True
                
        except Exception as e:
            # Fallback logging to stderr
            print(f"Audit logging failed: {e}", file=__import__('sys').stderr)
            return False
    
    def _write_structured_event(self, event: AuditEvent):
        """Write structured event to audit log file."""
        try:
            # Check file size and rotate if necessary
            if self.log_path.exists() and self.log_path.stat().st_size > self.max_file_size:
                self._rotate_log_file()
            
            # Write event as JSON line
            event_json = json.dumps(asdict(event), default=str, ensure_ascii=False)
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(event_json + '\n')
                
        except Exception as e:
            print(f"Structured audit write failed: {e}", file=__import__('sys').stderr)
    
    def _rotate_log_file(self):
        """Rotate log files when size limit is reached."""
        try:
            # Rotate existing backups
            for i in range(self.backup_count - 1, 0, -1):
                old_file = self.log_path.with_suffix(f'.{i}')
                new_file = self.log_path.with_suffix(f'.{i + 1}')
                if old_file.exists():
                    old_file.replace(new_file)
            
            # Move current log to .1
            if self.log_path.exists():
                backup_file = self.log_path.with_suffix('.1')
                self.log_path.replace(backup_file)
                
        except Exception as e:
            print(f"Log rotation failed: {e}", file=__import__('sys').stderr)
    
    def _generate_correlation_id(self) -> str:
        """Generate correlation ID for event tracing."""
        return f"step5_{int(time.time() * 1000000) % 1000000:06d}"

# Global audit logger instance
_global_audit_logger = None

def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger instance."""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger()
    return _global_audit_logger

def log_event(event_type: str, payload: Optional[Union[Dict[str, Any], str]] = None) -> bool:
    """
    Convenience function for logging audit events.
    
    Args:
        event_type: Type/name of the event
        payload: Event payload data
        
    Returns:
        Boolean indicating logging success
    """
    try:
        audit_logger = get_audit_logger()
        return audit_logger.log_event(event_type, payload)
    except Exception as e:
        # Fallback logging
        print(f"Audit event logging failed: {event_type}, {payload}, {e}", 
              file=__import__('sys').stderr)
        return False

def log_step5_event(phase: str, data: Optional[Dict[str, Any]] = None) -> bool:
    """
    Specialized logging function for UIP Step 5 events.
    
    Args:
        phase: Step 5 phase name
        data: Phase-specific data
        
    Returns:
        Boolean indicating logging success
    """
    return log_event(f"STEP5_{phase.upper()}", data)

def get_audit_status() -> Dict[str, Any]:
    """Get audit system status and statistics."""
    try:
        audit_logger = get_audit_logger()
        
        status = {
            "log_file": str(audit_logger.log_path),
            "log_file_exists": audit_logger.log_path.exists(),
            "log_file_size": audit_logger.log_path.stat().st_size if audit_logger.log_path.exists() else 0,
            "max_file_size": audit_logger.max_file_size,
            "backup_count": audit_logger.backup_count
        }
        
        # Count backup files
        backup_files = list(audit_logger.log_path.parent.glob(f"{audit_logger.log_path.stem}.*"))
        status["backup_files_count"] = len([f for f in backup_files if f.suffix.lstrip('.').isdigit()])
        
        # Read recent events count
        if audit_logger.log_path.exists():
            try:
                with open(audit_logger.log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    status["recent_events_count"] = len(lines)
                    status["last_event_sample"] = lines[-1].strip() if lines else None
            except Exception:
                status["recent_events_count"] = "unavailable"
        
        return status
        
    except Exception as e:
        return {"error": str(e), "status": "unavailable"}

__all__ = [
    "log_event",
    "log_step5_event", 
    "AuditLogger",
    "AuditLevel",
    "AuditEvent",
    "get_audit_logger",
    "get_audit_status"
]