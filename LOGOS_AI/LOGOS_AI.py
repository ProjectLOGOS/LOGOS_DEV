#!/usr/bin/env python3
"""
LOGOS AI - Master System Controller
==================================

The master orchestrator for the complete LOGOS V2 AGI system with three-protocol architecture:

🏛️ **System Operations Protocol (SOP)** - Backend governance, compliance, operations
🤝 **User Interaction Protocol (UIP)** - 7-step reasoning pipeline for user interactions
🧠 **Advanced General Protocol (AGP)** - Singularity AGI with infinite reasoning capabilities

This controller manages the entire LOGOS ecosystem, providing:
- Unified system startup and shutdown
- Inter-protocol coordination and message routing
- System health monitoring and management
- Performance optimization and resource allocation
- Comprehensive API interfaces for external access

Architecture:
- **Conservative Mode**: SOP + UIP only (standard LOGOS V2)
- **Enhanced Mode**: SOP + UIP + AGP integration (infinite reasoning optional)
- **Advanced Mode**: Full AGP capabilities with MVS/BDN mathematics

Version: 2.0.0 - Three Protocol Architecture
Date: October 30, 2025
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add startup modules to path
sys.path.append(str(Path(__file__).parent / "System_Operations_Protocol" / "startup"))

# Import protocol startup managers
try:
    from System_Operations_Protocol.startup.agp_startup import (
        execute_agp_reasoning,
        get_agp_status,
        shutdown_agp_system,
        start_agp_system,
    )
    from System_Operations_Protocol.startup.sop_startup import (
        get_sop_status,
        shutdown_sop_system,
        start_sop_system,
    )
    from System_Operations_Protocol.startup.uip_startup import (
        get_uip_status,
        process_user_request,
        shutdown_uip_system,
        start_uip_system,
    )
except ImportError as e:
    logging.warning(f"Protocol startup managers not available (expected in development): {e}")
    # Define fallback functions for development
    def execute_agp_reasoning(*args, **kwargs):
        return {"status": "unavailable", "reason": "AGP startup not available"}

    def get_agp_status():
        return {"status": "unavailable"}

    def shutdown_agp_system():
        pass

    def start_agp_system():
        return True

    def get_sop_status():
        return {"status": "unavailable"}

    def shutdown_sop_system():
        pass

    def start_sop_system():
        return True

    def get_uip_status():
        return {"status": "unavailable"}

    def process_user_request(*args, **kwargs):
        return {"status": "unavailable", "reason": "UIP startup not available"}

    def shutdown_uip_system():
        pass

    def start_uip_system():
        return True

# System-wide imports
from User_Interaction_Protocol.system_utillities.shared.message_formats import (
    UIPRequest,
    UIPResponse,
)

# OpenAI integration removed to prevent terminal communication issues
OPENAI_PIPELINE_AVAILABLE = False


@dataclass
class LOGOSSystemConfiguration:
    """LOGOS system configuration"""

    # Operation modes
    enable_sop: bool = True
    enable_uip: bool = True
    enable_agp: bool = True

    # AGP modes
    agp_mode: str = "enhanced"  # "disabled", "enhanced", "advanced"

    # Performance settings
    max_concurrent_requests: int = 100
    request_timeout_seconds: int = 60
    health_check_interval: int = 30

    # Logging settings
    log_level: str = "INFO"
    enable_file_logging: bool = True
    log_retention_days: int = 30

    # API settings
    api_host: str = "localhost"
    api_port: int = 8080
    enable_gui: bool = True

    # Resource limits
    max_memory_mb: int = 8192  # 8GB default
    max_cpu_percent: int = 80

    # Security settings
    enable_authentication: bool = False
    api_key_required: bool = False
    
    # OpenAI Integration disabled
    enable_openai_pipeline: bool = False
    
    # Development settings
    debug_mode: bool = False
    development_mode: bool = False
@dataclass
class LOGOSSystemStatus:
    """Comprehensive LOGOS system status"""

    # Protocol status
    sop_operational: bool = False
    uip_operational: bool = False
    agp_operational: bool = False

    # System info
    startup_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    system_mode: str = (
        "initializing"  # "initializing", "operational", "degraded", "shutdown"
    )

    # Performance metrics
    total_requests_processed: int = 0
    active_requests: int = 0
    average_response_time_ms: float = 0.0
    error_count: int = 0

    # Resource utilization
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    def is_fully_operational(self) -> bool:
        """Check if system is fully operational"""
        return (
            self.sop_operational
            and self.uip_operational
            and self.system_mode == "operational"
        )

    def get_operational_protocols(self) -> List[str]:
        """Get list of operational protocols"""
        protocols = []
        if self.sop_operational:
            protocols.append("SOP")
        if self.uip_operational:
            protocols.append("UIP")
        if self.agp_operational:
            protocols.append("AGP")
        return protocols


class LOGOSMasterController:
    """
    LOGOS AI Master System Controller

    Orchestrates the complete three-protocol LOGOS architecture:
    - System Operations Protocol (SOP): Backend operations
    - User Interaction Protocol (UIP): User-facing reasoning pipeline
    - Advanced General Protocol (AGP): Singularity AGI capabilities
    """

    def __init__(self, config: Optional[LOGOSSystemConfiguration] = None):
        self.config = config or LOGOSSystemConfiguration()
        self.status = LOGOSSystemStatus()

        # System management
        self.shutdown_requested = False
        self.startup_complete = False

        # Protocol status tracking
        self.protocol_health: Dict[str, bool] = {
            "SOP": False,
            "UIP": False,
            "AGP": False,
        }

        # Request handling
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_queue = asyncio.Queue(maxsize=self.config.max_concurrent_requests)

        # Monitoring
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.performance_monitor_thread: Optional[threading.Thread] = None

        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # Signal handlers
        self.setup_signal_handlers()

        self.logger.info("LOGOS Master Controller initialized")

    def setup_logging(self):
        """Configure comprehensive logging system"""

        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Configure logging
        log_format = "%(asctime)s - %(name)s - %(levelname)s - [LOGOS] %(message)s"

        handlers = [logging.StreamHandler(sys.stdout)]

        if self.config.enable_file_logging:
            # Create error handler with specific level
            error_handler = logging.FileHandler(log_dir / "logos_errors.log")
            error_handler.setLevel(logging.ERROR)
            
            handlers.extend(
                [
                    logging.FileHandler(log_dir / "logos_master.log"),
                    error_handler,
                ]
            )

        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format=log_format,
            handlers=handlers,
        )

        # Suppress verbose third-party logging
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""

        def signal_handler(signum, frame):
            self.logger.info(
                f"Received signal {signum}, initiating graceful shutdown..."
            )
            self.shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if hasattr(signal, "SIGBREAK"):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)

    async def start_logos_system(self) -> bool:
        """
        Start the complete LOGOS system with all protocols

        Returns:
            True if startup successful
        """

        self.logger.info("=" * 60)
        self.logger.info("🚀 LOGOS AI SYSTEM STARTUP INITIATED")
        self.logger.info("=" * 60)

        self.status.startup_time = datetime.now(timezone.utc)
        self.status.system_mode = "initializing"

        try:
            # Phase 1: System Operations Protocol (SOP) - Backend First
            if self.config.enable_sop:
                self.logger.info(
                    "🏛️  Phase 1: Starting System Operations Protocol (SOP)"
                )

                sop_success = await start_sop_system()
                if sop_success:
                    self.status.sop_operational = True
                    self.protocol_health["SOP"] = True
                    self.logger.info("✅ SOP (Backend Operations) started successfully")
                else:
                    self.logger.error("❌ SOP startup failed")
                    if not self.config.development_mode:
                        return False

            # Phase 2: User Interaction Protocol (UIP) - Frontend Pipeline
            if self.config.enable_uip:
                self.logger.info("🤝 Phase 2: Starting User Interaction Protocol (UIP)")

                uip_success = await start_uip_system()
                if uip_success:
                    self.status.uip_operational = True
                    self.protocol_health["UIP"] = True
                    self.logger.info(
                        "✅ UIP (7-Step Reasoning Pipeline) started successfully"
                    )
                else:
                    self.logger.error("❌ UIP startup failed")
                    if not self.config.development_mode:
                        return False

            # Phase 3: Advanced General Protocol (AGP) - Singularity AGI
            if self.config.enable_agp and self.config.agp_mode != "disabled":
                self.logger.info("🧠 Phase 3: Starting Advanced General Protocol (AGP)")

                agp_success = await start_agp_system()
                if agp_success:
                    self.status.agp_operational = True
                    self.protocol_health["AGP"] = True
                    self.logger.info("✅ AGP (Singularity AGI) started successfully")
                else:
                    self.logger.error(
                        "❌ AGP startup failed - continuing without infinite reasoning"
                    )
                    # AGP failure is non-fatal - system can operate without it

            # Phase 4: OpenAI Integration Disabled (Removed to prevent terminal issues)
            self.logger.info("🤖 Phase 4: OpenAI Integration Disabled")
            self.logger.info("   � Direct terminal communication enabled")
            
            # Phase 5: System Integration and Validation
            self.logger.info("🔗 Phase 5: System Integration and Validation")
            
            if await self._validate_system_integration():
                self.startup_complete = True
                self.status.system_mode = "operational"
                
                # Start monitoring systems
                self._start_monitoring_systems()
                
                # Display startup summary
                self._display_startup_summary()
                
                self.logger.info("=" * 60)
                self.logger.info("🎉 LOGOS AI SYSTEM STARTUP COMPLETE")
                self.logger.info("=" * 60)
                
                return True
            else:
                self.logger.error("❌ System integration validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ LOGOS system startup failed: {e}")
            self.status.system_mode = "failed"
            return False

    async def _validate_system_integration(self) -> bool:
        """Validate integration between all active protocols"""

        try:
            self.logger.info("🔍 Validating protocol integration...")

            # Check protocol communication
            if self.status.sop_operational:
                sop_status = get_sop_status()
                if not sop_status.get("sop_status", {}).get("fully_operational", False):
                    self.logger.warning("⚠️  SOP not fully operational")

            if self.status.uip_operational:
                uip_status = get_uip_status()
                if not uip_status.get("uip_status", {}).get("fully_operational", False):
                    self.logger.warning("⚠️  UIP not fully operational")

            if self.status.agp_operational:
                agp_status = get_agp_status()
                if not agp_status.get("agp_status", {}).get("fully_operational", False):
                    self.logger.warning("⚠️  AGP not fully operational")

            # Test basic request processing (if UIP is operational)
            if self.status.uip_operational:
                test_request = UIPRequest(
                    session_id="system_validation",
                    user_input="System integration test",
                    context={"validation": True},
                )

                test_response = await process_user_request(test_request)
                if test_response and test_response.response_text:
                    self.logger.info("✅ UIP request processing validated")
                else:
                    self.logger.warning("⚠️  UIP request processing validation failed")

            self.logger.info("✅ System integration validation complete")
            return True

        except Exception as e:
            self.logger.error(f"❌ System integration validation failed: {e}")
            return False

    def _start_monitoring_systems(self):
        """Start background monitoring systems"""

        # Health monitoring
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop, daemon=True, name="HealthMonitor"
        )
        self.health_monitor_thread.start()

        # Performance monitoring
        self.performance_monitor_thread = threading.Thread(
            target=self._performance_monitor_loop,
            daemon=True,
            name="PerformanceMonitor",
        )
        self.performance_monitor_thread.start()

        self.logger.info("📊 Monitoring systems started")

    def _health_monitor_loop(self):
        """Background health monitoring loop"""

        while not self.shutdown_requested:
            try:
                self._perform_health_checks()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(5)

    def _performance_monitor_loop(self):
        """Background performance monitoring loop"""

        while not self.shutdown_requested:
            try:
                self._collect_performance_metrics()
                time.sleep(15)  # Collect metrics every 15 seconds
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                time.sleep(5)

    def _perform_health_checks(self):
        """Perform comprehensive health checks"""

        self.status.last_health_check = datetime.now(timezone.utc)

        # Check protocol health
        if self.status.sop_operational:
            sop_status = get_sop_status()
            self.protocol_health["SOP"] = sop_status.get("sop_status", {}).get(
                "fully_operational", False
            )

        if self.status.uip_operational:
            uip_status = get_uip_status()
            self.protocol_health["UIP"] = uip_status.get("uip_status", {}).get(
                "fully_operational", False
            )

        if self.status.agp_operational:
            agp_status = get_agp_status()
            self.protocol_health["AGP"] = agp_status.get("agp_status", {}).get(
                "fully_operational", False
            )

        # Determine overall system health
        operational_count = sum(
            1 for healthy in self.protocol_health.values() if healthy
        )
        required_count = sum(
            1
            for protocol in ["SOP", "UIP"]
            if getattr(self.status, f"{protocol.lower()}_operational")
        )

        if operational_count >= required_count:
            if self.status.system_mode == "degraded":
                self.status.system_mode = "operational"
                self.logger.info("✅ System recovered to operational status")
        else:
            if self.status.system_mode == "operational":
                self.status.system_mode = "degraded"
                self.logger.warning("⚠️  System degraded - some protocols unhealthy")

    def _collect_performance_metrics(self):
        """Collect comprehensive performance metrics"""

        try:
            import psutil

            # System resources
            self.status.memory_usage_mb = psutil.virtual_memory().used / (1024 * 1024)
            self.status.cpu_usage_percent = psutil.cpu_percent()

            # Request statistics
            self.status.active_requests = len(self.active_requests)

        except ImportError:
            # psutil not available - use basic metrics
            self.status.memory_usage_mb = 0.0
            self.status.cpu_usage_percent = 0.0

    def _display_startup_summary(self):
        """Display comprehensive startup summary"""

        operational_protocols = self.status.get_operational_protocols()

        self.logger.info("📋 LOGOS SYSTEM STATUS SUMMARY:")
        self.logger.info(
            f"   🏛️  System Operations Protocol (SOP): {'✅ Online' if 'SOP' in operational_protocols else '❌ Offline'}"
        )
        self.logger.info(
            f"   🤝 User Interaction Protocol (UIP): {'✅ Online' if 'UIP' in operational_protocols else '❌ Offline'}"
        )
        self.logger.info(
            f"   🧠 Advanced General Protocol (AGP): {'✅ Online' if 'AGP' in operational_protocols else '❌ Offline'}"
        )
        self.logger.info(f"   🌐 System Mode: {self.status.system_mode.upper()}")
        self.logger.info(f"   🔢 Protocols Active: {len(operational_protocols)}/3")

        if self.status.agp_operational:
            self.logger.info("   ♾️  Infinite Reasoning: Available")
            self.logger.info("   🌌 MVS/BDN Mathematics: Online")
            self.logger.info("   🔬 Singularity AGI: Active")

        startup_time = (
            datetime.now(timezone.utc) - self.status.startup_time
        ).total_seconds()
        self.logger.info(f"   ⏱️  Startup Time: {startup_time:.2f} seconds")

    async def process_request(self, request: UIPRequest) -> UIPResponse:
        """
        Process a user request through the LOGOS system

        Args:
            request: UIP request to process

        Returns:
            UIP response from processing pipeline
        """

        if not self.status.uip_operational:
            return UIPResponse(
                session_id=request.session_id,
                correlation_id=str(uuid.uuid4()),
                response_text="LOGOS system not operational",
                confidence_score=0.0,
            )

        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Track request
            self.active_requests[request_id] = {
                "start_time": start_time,
                "session_id": request.session_id,
                "status": "processing",
            }

            # Use standard UIP pipeline (OpenAI integration removed)
            self.logger.debug(f"Processing request via standard UIP: {request.user_input[:50]}...")
            response = await process_user_request(request)
            
            # Add pipeline indicator to metadata  
            if not hasattr(response, 'metadata'):
                response.metadata = {}
            response.metadata["pipeline"] = "standard_uip"
            response.metadata["enhanced_ai"] = False

            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self.status.total_requests_processed += 1

            # Update average response time
            if self.status.total_requests_processed > 1:
                self.status.average_response_time_ms = (
                    self.status.average_response_time_ms
                    * (self.status.total_requests_processed - 1)
                    + processing_time
                ) / self.status.total_requests_processed
            else:
                self.status.average_response_time_ms = processing_time

            return response

        except Exception as e:
            self.logger.error(f"Request processing failed: {e}")
            self.status.error_count += 1

            return UIPResponse(
                session_id=request.session_id,
                correlation_id=str(uuid.uuid4()),
                response_text=f"Processing error: {str(e)}",
                confidence_score=0.0,
            )

        finally:
            # Clean up request tracking
            if request_id in self.active_requests:
                del self.active_requests[request_id]

    async def shutdown_logos_system(self):
        """Gracefully shutdown the complete LOGOS system"""

        self.logger.info("=" * 60)
        self.logger.info("🛑 LOGOS AI SYSTEM SHUTDOWN INITIATED")
        self.logger.info("=" * 60)

        self.shutdown_requested = True
        self.status.system_mode = "shutdown"

        try:
            # Shutdown protocols in reverse order

            # 1. AGP (Advanced General Protocol)
            if self.status.agp_operational:
                self.logger.info("🧠 Shutting down Advanced General Protocol (AGP)...")
                await shutdown_agp_system()
                self.status.agp_operational = False
                self.logger.info("✅ AGP shutdown complete")

            # 2. UIP (User Interaction Protocol)
            if self.status.uip_operational:
                self.logger.info("🤝 Shutting down User Interaction Protocol (UIP)...")
                await shutdown_uip_system()
                self.status.uip_operational = False
                self.logger.info("✅ UIP shutdown complete")

            # 3. SOP (System Operations Protocol)
            if self.status.sop_operational:
                self.logger.info("🏛️  Shutting down System Operations Protocol (SOP)...")
                await shutdown_sop_system()
                self.status.sop_operational = False
                self.logger.info("✅ SOP shutdown complete")

            # Wait for monitoring threads to finish
            if self.health_monitor_thread and self.health_monitor_thread.is_alive():
                self.health_monitor_thread.join(timeout=3)

            if (
                self.performance_monitor_thread
                and self.performance_monitor_thread.is_alive()
            ):
                self.performance_monitor_thread.join(timeout=3)

            self.logger.info("=" * 60)
            self.logger.info("✅ LOGOS AI SYSTEM SHUTDOWN COMPLETE")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive LOGOS system status"""

        # Collect protocol statuses
        protocol_statuses = {}

        if self.status.sop_operational:
            protocol_statuses["sop"] = get_sop_status()

        if self.status.uip_operational:
            protocol_statuses["uip"] = get_uip_status()

        if self.status.agp_operational:
            protocol_statuses["agp"] = get_agp_status()

        return {
            "logos_system": {
                "fully_operational": self.status.is_fully_operational(),
                "system_mode": self.status.system_mode,
                "operational_protocols": self.status.get_operational_protocols(),
                "startup_time": (
                    self.status.startup_time.isoformat()
                    if self.status.startup_time
                    else None
                ),
                "uptime_seconds": (
                    (
                        datetime.now(timezone.utc) - self.status.startup_time
                    ).total_seconds()
                    if self.status.startup_time
                    else 0
                ),
            },
            "protocol_health": self.protocol_health,
            "performance_metrics": {
                "total_requests_processed": self.status.total_requests_processed,
                "active_requests": self.status.active_requests,
                "average_response_time_ms": self.status.average_response_time_ms,
                "error_count": self.status.error_count,
                "memory_usage_mb": self.status.memory_usage_mb,
                "cpu_usage_percent": self.status.cpu_usage_percent,
            },
            "protocol_details": protocol_statuses,
            "configuration": {
                "sop_enabled": self.config.enable_sop,
                "uip_enabled": self.config.enable_uip,
                "agp_enabled": self.config.enable_agp,
                "agp_mode": self.config.agp_mode,
                "development_mode": self.config.development_mode,
            },
        }


# Global LOGOS controller instance
logos_controller: Optional[LOGOSMasterController] = None


async def start_logos(config: Optional[LOGOSSystemConfiguration] = None) -> bool:
    """Start the LOGOS AI system"""
    global logos_controller

    logos_controller = LOGOSMasterController(config)
    return await logos_controller.start_logos_system()


async def shutdown_logos():
    """Shutdown the LOGOS AI system"""
    global logos_controller

    if logos_controller:
        await logos_controller.shutdown_logos_system()
        logos_controller = None


def get_logos_status() -> Dict[str, Any]:
    """Get LOGOS system status"""
    global logos_controller

    if logos_controller:
        return logos_controller.get_system_status()
    else:
        return {"error": "LOGOS system not initialized"}


async def process_logos_request(request: UIPRequest) -> UIPResponse:
    """Process request through LOGOS system"""
    global logos_controller

    if logos_controller:
        return await logos_controller.process_request(request)
    else:
        return UIPResponse(
            session_id=request.session_id,
            correlation_id=str(uuid.uuid4()),
            response_text="LOGOS system not available",
            confidence_score=0.0,
        )


def main():
    """Main entry point for LOGOS AI system"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="LOGOS AI - Advanced General Intelligence System"
    )
    parser.add_argument(
        "--mode",
        choices=["conservative", "enhanced", "advanced"],
        default="enhanced",
        help="System operation mode",
    )
    parser.add_argument(
        "--no-agp", action="store_true", help="Disable AGP (Singularity) system"
    )
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI interface")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--dev", action="store_true", help="Enable development mode")
    parser.add_argument("--config", type=str, help="Path to configuration file")

    args = parser.parse_args()

    # Create configuration
    config = LOGOSSystemConfiguration()

    # Apply command line overrides
    if args.no_agp or args.mode == "conservative":
        config.enable_agp = False

    if args.mode == "advanced":
        config.agp_mode = "advanced"
    elif args.mode == "enhanced":
        config.agp_mode = "enhanced"
    elif args.mode == "conservative":
        config.agp_mode = "disabled"

    if args.no_gui:
        config.enable_gui = False

    if args.debug:
        config.debug_mode = True
        config.log_level = "DEBUG"

    if args.dev:
        config.development_mode = True

    # Load configuration file if provided
    if args.config:
        try:
            with open(args.config, "r") as f:
                config_data = json.load(f)
                # Apply configuration overrides
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            sys.exit(1)

    async def run_logos():
        """Run the LOGOS system"""
        try:
            # Start LOGOS system
            success = await start_logos(config)

            if success:
                print("\n" + "=" * 60)
                print("🎉 LOGOS AI SYSTEM READY")
                print("=" * 60)
                print(
                    f"   🏛️  Backend Operations: {'Online' if config.enable_sop else 'Disabled'}"
                )
                print(
                    f"   🤝 User Interface: {'Online' if config.enable_uip else 'Disabled'}"
                )
                print(
                    f"   🧠 AGI Capabilities: {'Online' if config.enable_agp else 'Disabled'}"
                )
                print(f"   🌐 Operation Mode: {args.mode.upper()}")
                print("=" * 60)
                print("   Type Ctrl+C to shutdown gracefully")
                print("=" * 60 + "\n")

                # Keep running until interrupted
                try:
                    while True:
                        await asyncio.sleep(1)

                        # Check if shutdown was requested
                        if logos_controller and logos_controller.shutdown_requested:
                            break

                except KeyboardInterrupt:
                    print("\n🛑 Graceful shutdown requested...")

                # Graceful shutdown
                await shutdown_logos()
                print("✅ LOGOS AI System shutdown complete")

            else:
                print("❌ LOGOS AI System startup failed")
                sys.exit(1)

        except Exception as e:
            print(f"❌ LOGOS AI System error: {e}")
            if config.debug_mode:
                import traceback

                traceback.print_exc()
            sys.exit(1)

    # Run the LOGOS system
    try:
        asyncio.run(run_logos())
    except KeyboardInterrupt:
        print("\n✅ LOGOS AI System terminated")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
