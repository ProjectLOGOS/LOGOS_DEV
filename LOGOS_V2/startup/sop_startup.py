#!/usr/bin/env python3
"""
System Operations Protocol (SOP) Startup Manager
================================================

Initializes and manages all backend system operations including:
- Governance and compliance systems
- Operational monitoring and management
- Deployment and configuration management
- Validation and audit systems
- State management and persistence

This is the executable implementation of the System Operations Protocol
as defined in the LOGOS architecture documentation.
"""

import asyncio
import logging
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import json
import threading
from concurrent.futures import ThreadPoolExecutor

# Add SOP modules to path
sys.path.append(str(Path(__file__).parent.parent / "System_Operations_Protocol"))

# Import SOP components
try:
    from governance.core.logos_core import LogosNexus, ReferenceMonitor
    from operations.monitoring import SystemMonitor, PerformanceTracker
    from deployment.deploy_core_services import LogosCoreDeployment
    from configuration.system_config import SystemConfiguration
    from validation.compliance_validator import ComplianceValidator
    from audit.audit_manager import AuditManager
    from boot.system_bootstrap import SystemBootstrap
    from state.state_manager import StateManager
    from persistence.data_manager import DataManager
except ImportError as e:
    logging.warning(f"Some SOP components not available: {e}")
    # Fallback implementations for development


@dataclass
class SOPSystemStatus:
    """Status tracking for SOP systems"""
    governance_active: bool = False
    operations_active: bool = False
    deployment_active: bool = False
    validation_active: bool = False
    audit_active: bool = False
    persistence_active: bool = False
    
    startup_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    active_services: List[str] = field(default_factory=list)
    error_count: int = 0
    
    def is_fully_operational(self) -> bool:
        """Check if all critical SOP systems are operational"""
        return all([
            self.governance_active,
            self.operations_active,
            self.validation_active,
            self.audit_active,
            self.persistence_active
        ])


class SOPStartupManager:
    """
    System Operations Protocol Startup Manager
    
    Orchestrates the initialization and management of all SOP backend systems
    according to the LOGOS architectural specifications.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent.parent / "System_Archive" / "sop_config.json"
        self.status = SOPSystemStatus()
        
        # Core components
        self.system_config: Optional[SystemConfiguration] = None
        self.logos_nexus: Optional[LogosNexus] = None
        self.reference_monitor: Optional[ReferenceMonitor] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.audit_manager: Optional[AuditManager] = None
        self.state_manager: Optional[StateManager] = None
        
        # Service management
        self.running_services: Dict[str, Any] = {}
        self.service_monitor_thread: Optional[threading.Thread] = None
        self.shutdown_requested = False
        
        # Logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("SOP Startup Manager initialized")
    
    def setup_logging(self):
        """Configure logging for SOP operations"""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - [SOP] %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler("startup/logs/sop_startup.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    async def initialize_sop_systems(self) -> bool:
        """
        Initialize all SOP backend systems in proper sequence
        
        Returns:
            True if all systems initialized successfully
        """
        
        self.logger.info("=== SOP SYSTEM INITIALIZATION STARTING ===")
        self.status.startup_time = datetime.now(timezone.utc)
        
        try:
            # Phase 1: Configuration and Bootstrap
            self.logger.info("Phase 1: System Configuration and Bootstrap")
            
            if not await self._initialize_configuration():
                return False
                
            if not await self._initialize_bootstrap():
                return False
            
            # Phase 2: Core Governance Systems
            self.logger.info("Phase 2: Core Governance Systems")
            
            if not await self._initialize_governance():
                return False
                
            if not await self._initialize_reference_monitor():
                return False
            
            # Phase 3: Operations and Monitoring
            self.logger.info("Phase 3: Operations and Monitoring")
            
            if not await self._initialize_operations():
                return False
                
            if not await self._initialize_monitoring():
                return False
            
            # Phase 4: Validation and Audit
            self.logger.info("Phase 4: Validation and Audit")
            
            if not await self._initialize_validation():
                return False
                
            if not await self._initialize_audit():
                return False
            
            # Phase 5: State and Persistence
            self.logger.info("Phase 5: State and Persistence")
            
            if not await self._initialize_state_management():
                return False
                
            if not await self._initialize_persistence():
                return False
            
            # Phase 6: Service Deployment
            self.logger.info("Phase 6: Service Deployment")
            
            if not await self._initialize_deployment():
                return False
            
            # Final health check
            if self.status.is_fully_operational():
                self.logger.info("=== SOP SYSTEM INITIALIZATION COMPLETE ===")
                self._start_service_monitoring()
                return True
            else:
                self.logger.error("SOP system initialization incomplete - some systems failed")
                return False
                
        except Exception as e:
            self.logger.error(f"SOP initialization failed: {e}")
            self.status.error_count += 1
            return False
    
    async def _initialize_configuration(self) -> bool:
        """Initialize system configuration"""
        try:
            self.system_config = SystemConfiguration(self.config_path)
            await self.system_config.load_configuration()
            self.logger.info("System configuration loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Configuration initialization failed: {e}")
            return False
    
    async def _initialize_bootstrap(self) -> bool:
        """Initialize system bootstrap"""
        try:
            bootstrap = SystemBootstrap(self.system_config)
            await bootstrap.execute_bootstrap_sequence()
            self.logger.info("System bootstrap completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Bootstrap initialization failed: {e}")
            return False
    
    async def _initialize_governance(self) -> bool:
        """Initialize governance systems"""
        try:
            self.logos_nexus = LogosNexus()
            await self.logos_nexus.initialize()
            
            self.status.governance_active = True
            self.status.active_services.append("governance")
            self.logger.info("Governance systems initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Governance initialization failed: {e}")
            return False
    
    async def _initialize_reference_monitor(self) -> bool:
        """Initialize reference monitor"""
        try:
            self.reference_monitor = ReferenceMonitor()
            await self.reference_monitor.start_monitoring()
            
            self.logger.info("Reference monitor initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Reference monitor initialization failed: {e}")
            return False
    
    async def _initialize_operations(self) -> bool:
        """Initialize operations management"""
        try:
            # Initialize operations components
            self.status.operations_active = True
            self.status.active_services.append("operations")
            self.logger.info("Operations management initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Operations initialization failed: {e}")
            return False
    
    async def _initialize_monitoring(self) -> bool:
        """Initialize system monitoring"""
        try:
            self.system_monitor = SystemMonitor()
            await self.system_monitor.start_monitoring()
            
            self.logger.info("System monitoring initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Monitoring initialization failed: {e}")
            return False
    
    async def _initialize_validation(self) -> bool:
        """Initialize validation systems"""
        try:
            validator = ComplianceValidator()
            await validator.initialize_validation_rules()
            
            self.status.validation_active = True
            self.status.active_services.append("validation")
            self.logger.info("Validation systems initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Validation initialization failed: {e}")
            return False
    
    async def _initialize_audit(self) -> bool:
        """Initialize audit systems"""
        try:
            self.audit_manager = AuditManager()
            await self.audit_manager.start_audit_logging()
            
            self.status.audit_active = True
            self.status.active_services.append("audit")
            self.logger.info("Audit systems initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Audit initialization failed: {e}")
            return False
    
    async def _initialize_state_management(self) -> bool:
        """Initialize state management"""
        try:
            self.state_manager = StateManager()
            await self.state_manager.initialize_state_tracking()
            
            self.logger.info("State management initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"State management initialization failed: {e}")
            return False
    
    async def _initialize_persistence(self) -> bool:
        """Initialize persistence systems"""
        try:
            data_manager = DataManager()
            await data_manager.initialize_persistence_layer()
            
            self.status.persistence_active = True
            self.status.active_services.append("persistence")
            self.logger.info("Persistence systems initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Persistence initialization failed: {e}")
            return False
    
    async def _initialize_deployment(self) -> bool:
        """Initialize deployment systems"""
        try:
            deployment = LogosCoreDeployment()
            await deployment.deploy_core_services()
            
            self.status.deployment_active = True
            self.status.active_services.append("deployment")
            self.logger.info("Deployment systems initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Deployment initialization failed: {e}")
            return False
    
    def _start_service_monitoring(self):
        """Start background service monitoring"""
        self.service_monitor_thread = threading.Thread(
            target=self._service_monitor_loop,
            daemon=True
        )
        self.service_monitor_thread.start()
        self.logger.info("Service monitoring started")
    
    def _service_monitor_loop(self):
        """Background service monitoring loop"""
        while not self.shutdown_requested:
            try:
                # Perform health checks
                self._perform_health_checks()
                
                # Sleep between checks
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Service monitoring error: {e}")
                self.status.error_count += 1
                time.sleep(5)  # Brief pause before retry
    
    def _perform_health_checks(self):
        """Perform health checks on all SOP systems"""
        self.status.last_health_check = datetime.now(timezone.utc)
        
        # Check each system component
        if self.logos_nexus and not self.logos_nexus.is_healthy():
            self.logger.warning("Governance system health check failed")
            
        if self.system_monitor and not self.system_monitor.is_healthy():
            self.logger.warning("System monitor health check failed")
            
        # Add additional health checks as needed
    
    async def shutdown_sop_systems(self):
        """Gracefully shutdown all SOP systems"""
        self.logger.info("=== SOP SYSTEM SHUTDOWN INITIATED ===")
        self.shutdown_requested = True
        
        try:
            # Shutdown in reverse order
            if self.audit_manager:
                await self.audit_manager.shutdown()
                
            if self.system_monitor:
                await self.system_monitor.shutdown()
                
            if self.reference_monitor:
                await self.reference_monitor.shutdown()
                
            if self.logos_nexus:
                await self.logos_nexus.shutdown()
            
            # Wait for monitoring thread to finish
            if self.service_monitor_thread and self.service_monitor_thread.is_alive():
                self.service_monitor_thread.join(timeout=5)
            
            self.logger.info("=== SOP SYSTEM SHUTDOWN COMPLETE ===")
            
        except Exception as e:
            self.logger.error(f"SOP shutdown error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive SOP system status"""
        return {
            "sop_status": {
                "fully_operational": self.status.is_fully_operational(),
                "governance_active": self.status.governance_active,
                "operations_active": self.status.operations_active,
                "deployment_active": self.status.deployment_active,
                "validation_active": self.status.validation_active,
                "audit_active": self.status.audit_active,
                "persistence_active": self.status.persistence_active
            },
            "system_info": {
                "startup_time": self.status.startup_time.isoformat() if self.status.startup_time else None,
                "last_health_check": self.status.last_health_check.isoformat() if self.status.last_health_check else None,
                "active_services": self.status.active_services,
                "error_count": self.status.error_count,
                "uptime_seconds": (datetime.now(timezone.utc) - self.status.startup_time).total_seconds() if self.status.startup_time else 0
            },
            "service_details": {
                service: "active" for service in self.status.active_services
            }
        }


# Global SOP manager instance
sop_manager: Optional[SOPStartupManager] = None


async def start_sop_system(config_path: Optional[Path] = None) -> bool:
    """
    Start the System Operations Protocol system
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        True if startup successful
    """
    global sop_manager
    
    sop_manager = SOPStartupManager(config_path)
    return await sop_manager.initialize_sop_systems()


async def shutdown_sop_system():
    """Shutdown the SOP system"""
    global sop_manager
    
    if sop_manager:
        await sop_manager.shutdown_sop_systems()
        sop_manager = None


def get_sop_status() -> Dict[str, Any]:
    """Get SOP system status"""
    global sop_manager
    
    if sop_manager:
        return sop_manager.get_system_status()
    else:
        return {"sop_status": {"fully_operational": False}, "error": "SOP not initialized"}


# Main execution for standalone SOP startup
if __name__ == "__main__":
    async def main():
        try:
            # Start SOP system
            success = await start_sop_system()
            
            if success:
                print("✅ SOP System initialized successfully")
                
                # Keep running until interrupted
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    print("\n🛑 SOP System shutdown requested")
                    
                # Graceful shutdown
                await shutdown_sop_system()
                print("✅ SOP System shutdown complete")
                
            else:
                print("❌ SOP System initialization failed")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ SOP System error: {e}")
            sys.exit(1)
    
    # Run the SOP system
    asyncio.run(main())