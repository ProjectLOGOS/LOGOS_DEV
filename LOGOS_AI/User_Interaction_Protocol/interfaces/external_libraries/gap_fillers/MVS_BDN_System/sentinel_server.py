#!/usr/bin/env python3
"""
LOGOS Sentinel Server
=====================

System-level monitoring, heartbeat management, and compliance enforcement
server for the LOGOS AGI Trinity-grounded architecture.

Trinity Foundation: Continuous validation of Trinity mathematical coherence
across all system components with real-time monitoring and alert management.

Key Responsibilities:
- System heartbeat monitoring and health status tracking
- Real-time compliance validation and enforcement
- Performance metrics collection and analysis
- Alert generation and notification management
- System recovery and failover coordination
- Trinity coherence continuous validation

Author: LOGOS Development Team
Version: 2.0.0
License: Proprietary - Trinity Foundation
"""

import os
import sys
import json
import time
import logging
import threading
import asyncio
import signal
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from threading import Lock, Event, Thread
import socket
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
import uuid
import hashlib

# Core imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schemas import LOGOSValidationOrchestrator, ValidationResult
from init.system_init import SystemInitializer, SubsystemState


class SentinelStatus(Enum):
    """Sentinel server status states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    MONITORING = "monitoring"
    ALERT = "alert"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    SHUTDOWN = "shutdown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class HeartbeatRecord:
    """Record of a system component heartbeat."""
    component_id: str
    component_type: str
    timestamp: datetime
    status: str
    performance_data: Dict[str, Any] = field(default_factory=dict)
    trinity_coherence: float = 0.0
    validation_token: Optional[str] = None
    error_count: int = 0
    
    def is_healthy(self, timeout_seconds: int = 30) -> bool:
        """Check if heartbeat indicates healthy component."""
        time_diff = datetime.now(timezone.utc) - self.timestamp
        return (time_diff.total_seconds() < timeout_seconds and 
                self.status in ["healthy", "operational", "active"] and
                self.trinity_coherence >= 0.7 and
                self.error_count < 5)


@dataclass
class SystemAlert:
    """System alert record."""
    alert_id: str
    severity: AlertSeverity
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    escalation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_io: float = 0.0
    disk_io: float = 0.0
    response_time_avg: float = 0.0
    error_rate: float = 0.0
    trinity_coherence_avg: float = 0.0
    validation_success_rate: float = 0.0
    active_components: int = 0
    total_requests: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SentinelHTTPHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for sentinel server API endpoints."""
    
    def __init__(self, sentinel_server, *args, **kwargs):
        self.sentinel_server = sentinel_server
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        try:
            path = urlparse(self.path).path
            query = parse_qs(urlparse(self.path).query)
            
            if path == "/status":
                self._handle_status_request()
            elif path == "/health":
                self._handle_health_request()
            elif path == "/metrics":
                self._handle_metrics_request()
            elif path == "/alerts":
                self._handle_alerts_request(query)
            elif path == "/heartbeats":
                self._handle_heartbeats_request(query)
            else:
                self._send_error(404, "Endpoint not found")
                
        except Exception as e:
            self._send_error(500, f"Internal server error: {str(e)}")
    
    def do_POST(self):
        """Handle POST requests."""
        try:
            path = urlparse(self.path).path
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            if path == "/heartbeat":
                self._handle_heartbeat_post(data)
            elif path == "/alert":
                self._handle_alert_post(data)
            elif path == "/command":
                self._handle_command_post(data)
            else:
                self._send_error(404, "Endpoint not found")
                
        except Exception as e:
            self._send_error(500, f"Internal server error: {str(e)}")
    
    def _handle_status_request(self):
        """Handle status request."""
        status = self.sentinel_server.get_system_status()
        self._send_json_response(status)
    
    def _handle_health_request(self):
        """Handle health check request."""
        health = self.sentinel_server.get_health_summary()
        self._send_json_response(health)
    
    def _handle_metrics_request(self):
        """Handle metrics request."""
        metrics = self.sentinel_server.get_performance_metrics()
        self._send_json_response(metrics.__dict__)
    
    def _handle_alerts_request(self, query: Dict[str, List[str]]):
        """Handle alerts request."""
        severity = query.get('severity', [None])[0]
        resolved = query.get('resolved', ['false'])[0].lower() == 'true'
        alerts = self.sentinel_server.get_alerts(severity, resolved)
        self._send_json_response([alert.__dict__ for alert in alerts])
    
    def _handle_heartbeats_request(self, query: Dict[str, List[str]]):
        """Handle heartbeats request."""
        component = query.get('component', [None])[0]
        heartbeats = self.sentinel_server.get_heartbeats(component)
        self._send_json_response([hb.__dict__ for hb in heartbeats])
    
    def _handle_heartbeat_post(self, data: Dict[str, Any]):
        """Handle heartbeat POST."""
        result = self.sentinel_server.process_heartbeat(data)
        self._send_json_response(result)
    
    def _handle_alert_post(self, data: Dict[str, Any]):
        """Handle alert POST."""
        result = self.sentinel_server.create_alert(data)
        self._send_json_response(result)
    
    def _handle_command_post(self, data: Dict[str, Any]):
        """Handle command POST."""
        result = self.sentinel_server.execute_command(data)
        self._send_json_response(result)
    
    def _send_json_response(self, data: Any):
        """Send JSON response."""
        response = json.dumps(data, default=str, indent=2)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-length', str(len(response)))
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))
    
    def _send_error(self, code: int, message: str):
        """Send error response."""
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        error_response = json.dumps({"error": message, "code": code})
        self.wfile.write(error_response.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override log message to use sentinel logger."""
        if hasattr(self, 'sentinel_server'):
            self.sentinel_server.logger.info(f"HTTP: {format % args}")


class LogosSentinelServer:
    """
    Main LOGOS Sentinel Server for system monitoring and compliance enforcement.
    """
    
    def __init__(self, port: int = 8080, config_path: str = "sentinel_config.json"):
        self.port = port
        self.config_path = config_path
        
        # Core components
        self.schemas_validator = LOGOSValidationOrchestrator()
        self.system_initializer: Optional[SystemInitializer] = None
        
        # Server state
        self.status = SentinelStatus.INITIALIZING
        self.startup_time = datetime.now(timezone.utc)
        self.shutdown_event = Event()
        
        # Monitoring data
        self.heartbeats: Dict[str, HeartbeatRecord] = {}
        self.alerts: List[SystemAlert] = []
        self.performance_metrics = PerformanceMetrics()
        self.metrics_history = deque(maxlen=1000)
        
        # Threading
        self.monitoring_thread: Optional[Thread] = None
        self.http_server: Optional[socketserver.TCPServer] = None
        self.data_lock = Lock()
        
        # Configuration
        self.config = self._load_configuration()
        self.heartbeat_timeout = self.config.get("heartbeat_timeout", 30)
        self.monitoring_interval = self.config.get("monitoring_interval", 5)
        self.alert_escalation_threshold = self.config.get("alert_escalation_threshold", 3)
        self.trinity_coherence_threshold = self.config.get("trinity_coherence_threshold", 0.7)
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        self.health_callbacks: List[Callable] = []
        
        # Logging
        self.logger = self._setup_logging()
        
    def _load_configuration(self) -> Dict[str, Any]:
        """Load sentinel configuration."""
        default_config = {
            "heartbeat_timeout": 30,
            "monitoring_interval": 5,
            "alert_escalation_threshold": 3,
            "trinity_coherence_threshold": 0.7,
            "max_alerts": 1000,
            "max_heartbeats": 500,
            "performance_history_size": 1000,
            "log_level": "INFO",
            "enable_http_api": True,
            "enable_real_time_monitoring": True
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Set up sentinel server logging."""
        logger = logging.getLogger('SENTINEL_SERVER')
        logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
        
        # File handler
        file_handler = logging.FileHandler('logs/sentinel_server.log')
        file_formatter = logging.Formatter(
            '%(asctime)s - SENTINEL - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - SENTINEL - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def initialize(self, system_initializer: SystemInitializer = None) -> bool:
        """Initialize the sentinel server."""
        self.logger.info("🔧 Initializing LOGOS Sentinel Server...")
        
        try:
            # Store system initializer reference
            self.system_initializer = system_initializer
            
            # Validate schemas system
            if not self._validate_schemas_system():
                return False
            
            # Start HTTP server if enabled
            if self.config.get("enable_http_api", True):
                self._start_http_server()
            
            # Start monitoring thread if enabled
            if self.config.get("enable_real_time_monitoring", True):
                self._start_monitoring_thread()
            
            # Set active status
            self.status = SentinelStatus.ACTIVE
            
            self.logger.info("✅ LOGOS Sentinel Server initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"💥 Sentinel server initialization failed: {e}")
            return False
    
    def _validate_schemas_system(self) -> bool:
        """Validate that schemas system is operational."""
        try:
            # Test schemas validation
            test_request = {
                "request_id": "sentinel_validation_test",
                "proposition_or_plan": {
                    "sentinel_test": True,
                    "ontological_grounding": 0.95
                }
            }
            
            result = self.schemas_validator.validate_complete_request(test_request)
            
            if result.get("decision") in ["locked", "quarantine", "reject"]:
                self.logger.info("✅ Schemas validation system operational")
                return True
            else:
                self.logger.error("❌ Schemas validation system not responding properly")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Schemas validation test failed: {e}")
            return False
    
    def _start_http_server(self):
        """Start HTTP API server."""
        try:
            def handler_factory(*args, **kwargs):
                return SentinelHTTPHandler(self, *args, **kwargs)
            
            self.http_server = socketserver.TCPServer(("", self.port), handler_factory)
            
            # Start server in separate thread
            server_thread = Thread(
                target=self.http_server.serve_forever,
                name="SentinelHTTPServer",
                daemon=True
            )
            server_thread.start()
            
            self.logger.info(f"✅ HTTP API server started on port {self.port}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start HTTP server: {e}")
            raise
    
    def _start_monitoring_thread(self):
        """Start real-time monitoring thread."""
        self.monitoring_thread = Thread(
            target=self._monitoring_loop,
            name="SentinelMonitoring",
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("✅ Real-time monitoring thread started")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        self.logger.info("🔄 Sentinel monitoring loop started")
        
        while not self.shutdown_event.is_set():
            try:
                # Update system status
                self._update_system_status()
                
                # Process heartbeats
                self._process_heartbeat_timeouts()
                
                # Validate Trinity coherence
                self._validate_trinity_coherence()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Process alerts
                self._process_alert_escalations()
                
                # Check system health
                self._check_system_health()
                
                # Sleep until next cycle
                self.shutdown_event.wait(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"💥 Monitoring loop error: {e}")
                time.sleep(1)  # Brief pause on error
    
    def process_heartbeat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming heartbeat from system component."""
        try:
            component_id = data.get("component_id")
            component_type = data.get("component_type", "unknown")
            
            if not component_id:
                return {"status": "error", "message": "Missing component_id"}
            
            # Create heartbeat record
            heartbeat = HeartbeatRecord(
                component_id=component_id,
                component_type=component_type,
                timestamp=datetime.now(timezone.utc),
                status=data.get("status", "unknown"),
                performance_data=data.get("performance_data", {}),
                trinity_coherence=data.get("trinity_coherence", 0.0),
                validation_token=data.get("validation_token"),
                error_count=data.get("error_count", 0)
            )
            
            # Store heartbeat
            with self.data_lock:
                self.heartbeats[component_id] = heartbeat
            
            # Validate Trinity coherence
            if heartbeat.trinity_coherence < self.trinity_coherence_threshold:
                self._create_trinity_coherence_alert(component_id, heartbeat.trinity_coherence)
            
            # Validate component health
            if not heartbeat.is_healthy(self.heartbeat_timeout):
                self._create_health_alert(component_id, heartbeat)
            
            self.logger.debug(f"📡 Heartbeat received from {component_id}: {heartbeat.status}")
            
            return {
                "status": "accepted",
                "component_id": component_id,
                "timestamp": heartbeat.timestamp.isoformat(),
                "next_heartbeat_due": (heartbeat.timestamp + timedelta(seconds=self.heartbeat_timeout)).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"💥 Error processing heartbeat: {e}")
            return {"status": "error", "message": str(e)}
    
    def create_alert(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new system alert."""
        try:
            alert_id = str(uuid.uuid4())
            
            alert = SystemAlert(
                alert_id=alert_id,
                severity=AlertSeverity(data.get("severity", "info")),
                component=data.get("component", "unknown"),
                message=data.get("message", ""),
                timestamp=datetime.now(timezone.utc),
                metadata=data.get("metadata", {})
            )
            
            with self.data_lock:
                self.alerts.append(alert)
                
                # Trim alerts if too many
                max_alerts = self.config.get("max_alerts", 1000)
                if len(self.alerts) > max_alerts:
                    self.alerts = self.alerts[-max_alerts:]
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback error: {e}")
            
            self.logger.warning(f"🚨 Alert created: {alert.severity.value} - {alert.message}")
            
            return {"status": "created", "alert_id": alert_id}
            
        except Exception as e:
            self.logger.error(f"💥 Error creating alert: {e}")
            return {"status": "error", "message": str(e)}
    
    def execute_command(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sentinel command."""
        try:
            command = data.get("command")
            
            if command == "shutdown":
                return self._command_shutdown()
            elif command == "reset_metrics":
                return self._command_reset_metrics()
            elif command == "clear_alerts":
                return self._command_clear_alerts()
            elif command == "force_validation":
                return self._command_force_validation(data.get("component"))
            else:
                return {"status": "error", "message": f"Unknown command: {command}"}
                
        except Exception as e:
            self.logger.error(f"💥 Error executing command: {e}")
            return {"status": "error", "message": str(e)}
    
    def _process_heartbeat_timeouts(self):
        """Process heartbeat timeouts and create alerts."""
        current_time = datetime.now(timezone.utc)
        
        with self.data_lock:
            timed_out_components = []
            
            for component_id, heartbeat in self.heartbeats.items():
                time_diff = current_time - heartbeat.timestamp
                
                if time_diff.total_seconds() > self.heartbeat_timeout:
                    timed_out_components.append(component_id)
            
            # Create timeout alerts
            for component_id in timed_out_components:
                self._create_timeout_alert(component_id)
    
    def _validate_trinity_coherence(self):
        """Validate Trinity coherence across all components."""
        with self.data_lock:
            if not self.heartbeats:
                return
            
            coherence_values = [hb.trinity_coherence for hb in self.heartbeats.values()]
            avg_coherence = sum(coherence_values) / len(coherence_values)
            
            self.performance_metrics.trinity_coherence_avg = avg_coherence
            
            if avg_coherence < self.trinity_coherence_threshold:
                self._create_system_coherence_alert(avg_coherence)
    
    def _update_performance_metrics(self):
        """Update system performance metrics."""
        with self.data_lock:
            # Update basic counts
            self.performance_metrics.active_components = len([
                hb for hb in self.heartbeats.values() 
                if hb.is_healthy(self.heartbeat_timeout)
            ])
            
            # Calculate error rate
            total_errors = sum(hb.error_count for hb in self.heartbeats.values())
            total_components = len(self.heartbeats)
            self.performance_metrics.error_rate = (total_errors / total_components 
                                                  if total_components > 0 else 0.0)
            
            # Update validation success rate
            validation_metrics = self.schemas_validator.get_validation_metrics()
            self.performance_metrics.validation_success_rate = validation_metrics.get("success_rate_percent", 0.0) / 100.0
            
            # Update timestamp
            self.performance_metrics.last_updated = datetime.now(timezone.utc)
            
            # Add to history
            self.metrics_history.append({
                'timestamp': self.performance_metrics.last_updated,
                'active_components': self.performance_metrics.active_components,
                'error_rate': self.performance_metrics.error_rate,
                'trinity_coherence_avg': self.performance_metrics.trinity_coherence_avg,
                'validation_success_rate': self.performance_metrics.validation_success_rate
            })
    
    def _process_alert_escalations(self):
        """Process alert escalations based on thresholds."""
        with self.data_lock:
            for alert in self.alerts:
                if (not alert.resolved and 
                    alert.escalation_count < self.alert_escalation_threshold and
                    alert.severity == AlertSeverity.WARNING):
                    
                    # Check if alert should be escalated
                    time_since_creation = datetime.now(timezone.utc) - alert.timestamp
                    if time_since_creation.total_seconds() > 300:  # 5 minutes
                        alert.severity = AlertSeverity.ERROR
                        alert.escalation_count += 1
                        self.logger.error(f"🔥 Alert escalated: {alert.alert_id}")
    
    def _check_system_health(self):
        """Check overall system health and update status."""
        with self.data_lock:
            healthy_components = len([
                hb for hb in self.heartbeats.values() 
                if hb.is_healthy(self.heartbeat_timeout)
            ])
            total_components = len(self.heartbeats)
            
            # Count critical alerts
            critical_alerts = len([
                alert for alert in self.alerts 
                if not alert.resolved and alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
            ])
            
            # Determine system status
            if total_components == 0:
                new_status = SentinelStatus.INITIALIZING
            elif critical_alerts > 0:
                new_status = SentinelStatus.CRITICAL
            elif healthy_components == 0:
                new_status = SentinelStatus.CRITICAL
            elif healthy_components < total_components // 2:
                new_status = SentinelStatus.DEGRADED
            elif len([alert for alert in self.alerts if not alert.resolved]) > 0:
                new_status = SentinelStatus.ALERT
            else:
                new_status = SentinelStatus.MONITORING
            
            # Update status if changed
            if new_status != self.status:
                old_status = self.status
                self.status = new_status
                self.logger.info(f"📊 System status changed: {old_status.value} → {new_status.value}")
                
                # Trigger health callbacks
                for callback in self.health_callbacks:
                    try:
                        callback(old_status, new_status)
                    except Exception as e:
                        self.logger.error(f"Health callback error: {e}")
    
    def _update_system_status(self):
        """Update overall system status based on components."""
        if self.system_initializer:
            # Get subsystem status from system initializer
            subsystem_count = len(self.system_initializer.subsystems)
            authorized_count = len([
                auth for auth in self.system_initializer.subsystems.values()
                if auth.state == SubsystemState.AUTHORIZED
            ])
            
            self.performance_metrics.total_requests = subsystem_count
            self.logger.debug(f"📊 Subsystems: {authorized_count}/{subsystem_count} authorized")
    
    def _create_trinity_coherence_alert(self, component_id: str, coherence: float):
        """Create alert for Trinity coherence violation."""
        alert_data = {
            "severity": "warning",
            "component": component_id,
            "message": f"Trinity coherence below threshold: {coherence:.3f} < {self.trinity_coherence_threshold}",
            "metadata": {
                "type": "trinity_coherence",
                "measured_coherence": coherence,
                "threshold": self.trinity_coherence_threshold
            }
        }
        self.create_alert(alert_data)
    
    def _create_health_alert(self, component_id: str, heartbeat: HeartbeatRecord):
        """Create alert for component health issues."""
        alert_data = {
            "severity": "error" if heartbeat.error_count > 3 else "warning",
            "component": component_id,
            "message": f"Component health degraded: {heartbeat.status}, errors: {heartbeat.error_count}",
            "metadata": {
                "type": "component_health",
                "status": heartbeat.status,
                "error_count": heartbeat.error_count,
                "trinity_coherence": heartbeat.trinity_coherence
            }
        }
        self.create_alert(alert_data)
    
    def _create_timeout_alert(self, component_id: str):
        """Create alert for component timeout."""
        alert_data = {
            "severity": "error",
            "component": component_id,
            "message": f"Component heartbeat timeout (>{self.heartbeat_timeout}s)",
            "metadata": {
                "type": "heartbeat_timeout",
                "timeout_seconds": self.heartbeat_timeout
            }
        }
        self.create_alert(alert_data)
    
    def _create_system_coherence_alert(self, avg_coherence: float):
        """Create alert for system-wide Trinity coherence issues."""
        alert_data = {
            "severity": "critical",
            "component": "SYSTEM",
            "message": f"System Trinity coherence degraded: {avg_coherence:.3f}",
            "metadata": {
                "type": "system_coherence",
                "average_coherence": avg_coherence,
                "threshold": self.trinity_coherence_threshold
            }
        }
        self.create_alert(alert_data)
    
    def _command_shutdown(self) -> Dict[str, Any]:
        """Execute shutdown command."""
        self.logger.info("🛑 Shutdown command received")
        self.shutdown()
        return {"status": "shutdown_initiated"}
    
    def _command_reset_metrics(self) -> Dict[str, Any]:
        """Execute reset metrics command."""
        with self.data_lock:
            self.performance_metrics = PerformanceMetrics()
            self.metrics_history.clear()
            self.schemas_validator.reset_metrics()
        
        self.logger.info("📊 Metrics reset")
        return {"status": "metrics_reset"}
    
    def _command_clear_alerts(self) -> Dict[str, Any]:
        """Execute clear alerts command."""
        with self.data_lock:
            cleared_count = len(self.alerts)
            self.alerts.clear()
        
        self.logger.info(f"🧹 Cleared {cleared_count} alerts")
        return {"status": "alerts_cleared", "count": cleared_count}
    
    def _command_force_validation(self, component: Optional[str]) -> Dict[str, Any]:
        """Execute force validation command."""
        if component and component in self.heartbeats:
            # Force validation for specific component
            heartbeat = self.heartbeats[component]
            
            # Create validation request
            validation_request = {
                "request_id": f"forced_validation_{component}_{datetime.now().timestamp()}",
                "proposition_or_plan": {
                    "component": component,
                    "forced_validation": True,
                    "ontological_grounding": 0.9
                }
            }
            
            result = self.schemas_validator.validate_complete_request(validation_request)
            return {"status": "validation_completed", "component": component, "result": result}
        else:
            return {"status": "error", "message": "Component not found or not specified"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self.data_lock:
            return {
                "sentinel_server": {
                    "status": self.status.value,
                    "startup_time": self.startup_time.isoformat(),
                    "uptime_seconds": (datetime.now(timezone.utc) - self.startup_time).total_seconds(),
                    "version": "2.0.0",
                    "port": self.port
                },
                "monitoring": {
                    "heartbeat_timeout": self.heartbeat_timeout,
                    "monitoring_interval": self.monitoring_interval,
                    "trinity_coherence_threshold": self.trinity_coherence_threshold
                },
                "components": {
                    "total": len(self.heartbeats),
                    "healthy": len([hb for hb in self.heartbeats.values() if hb.is_healthy(self.heartbeat_timeout)]),
                    "unhealthy": len([hb for hb in self.heartbeats.values() if not hb.is_healthy(self.heartbeat_timeout)])
                },
                "alerts": {
                    "total": len(self.alerts),
                    "unresolved": len([alert for alert in self.alerts if not alert.resolved]),
                    "critical": len([alert for alert in self.alerts if not alert.resolved and alert.severity == AlertSeverity.CRITICAL])
                },
                "performance": self.performance_metrics.__dict__,
                "schemas_validation": self.schemas_validator.get_validation_metrics()
            }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get simple health summary."""
        with self.data_lock:
            healthy_components = len([hb for hb in self.heartbeats.values() if hb.is_healthy(self.heartbeat_timeout)])
            total_components = len(self.heartbeats)
            unresolved_alerts = len([alert for alert in self.alerts if not alert.resolved])
            
            overall_health = "healthy"
            if self.status == SentinelStatus.CRITICAL:
                overall_health = "critical"
            elif self.status == SentinelStatus.DEGRADED:
                overall_health = "degraded"
            elif unresolved_alerts > 0:
                overall_health = "warning"
            
            return {
                "overall_health": overall_health,
                "components_healthy": f"{healthy_components}/{total_components}",
                "unresolved_alerts": unresolved_alerts,
                "trinity_coherence_avg": self.performance_metrics.trinity_coherence_avg,
                "uptime_seconds": (datetime.now(timezone.utc) - self.startup_time).total_seconds(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.performance_metrics
    
    def get_alerts(self, severity: Optional[str] = None, resolved: bool = False) -> List[SystemAlert]:
        """Get alerts with optional filtering."""
        with self.data_lock:
            filtered_alerts = []
            
            for alert in self.alerts:
                if resolved != alert.resolved:
                    continue
                if severity and alert.severity.value != severity:
                    continue
                filtered_alerts.append(alert)
            
            return filtered_alerts[-50:]  # Return last 50 matching alerts
    
    def get_heartbeats(self, component: Optional[str] = None) -> List[HeartbeatRecord]:
        """Get heartbeats with optional filtering."""
        with self.data_lock:
            if component:
                return [self.heartbeats[component]] if component in self.heartbeats else []
            else:
                return list(self.heartbeats.values())
    
    def add_alert_callback(self, callback: Callable[[SystemAlert], None]):
        """Add callback for alert events."""
        self.alert_callbacks.append(callback)
    
    def add_health_callback(self, callback: Callable[[SentinelStatus, SentinelStatus], None]):
        """Add callback for health status changes."""
        self.health_callbacks.append(callback)
    
    def shutdown(self):
        """Gracefully shutdown the sentinel server."""
        self.logger.info("🛑 Shutting down LOGOS Sentinel Server...")
        
        # Set shutdown flag
        self.shutdown_event.set()
        self.status = SentinelStatus.SHUTDOWN
        
        # Stop HTTP server
        if self.http_server:
            self.http_server.shutdown()
            self.http_server.server_close()
        
        # Wait for monitoring thread to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        self.logger.info("🛑 LOGOS Sentinel Server shutdown complete")


def create_sentinel_server(port: int = 8080, config_path: str = "sentinel_config.json") -> LogosSentinelServer:
    """Factory function to create sentinel server."""
    return LogosSentinelServer(port, config_path)


def run_sentinel_server(port: int = 8080, system_initializer: SystemInitializer = None):
    """Run sentinel server with proper signal handling."""
    
    # Create sentinel server
    sentinel = create_sentinel_server(port)
    
    # Signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}")
        sentinel.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize and run
    if sentinel.initialize(system_initializer):
        print(f"✅ LOGOS Sentinel Server running on port {port}")
        print("🔍 Monitoring system components...")
        print("🌐 HTTP API available at http://localhost:{port}")
        print("🛑 Press Ctrl+C to shutdown")
        
        try:
            # Keep main thread alive
            while not sentinel.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested")
        finally:
            sentinel.shutdown()
    else:
        print("❌ Failed to initialize LOGOS Sentinel Server")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LOGOS Sentinel Server")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port")
    parser.add_argument("--config", default="sentinel_config.json", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Run sentinel server
    run_sentinel_server(args.port)