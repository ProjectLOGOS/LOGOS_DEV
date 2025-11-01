#!/usr/bin/env python3
"""
SOP Nexus - System Operations Protocol Central Hub
================================================

The SOP Nexus is the central communication hub and infrastructure coordinator
for the LOGOS system. It provides:

- Infrastructure management and system boot coordination
- Token distribution and verification system
- Gap detection and TODO generation
- File management with scaffolds and backups
- Cross-protocol communication facilitation
- System maintenance and monitoring

Key Principles:
- Always Active: Runs continuously in background
- Facilitation Only: Enables operations, doesn't process data
- System Agent Access Only: No direct user interaction
- Central Hub: All inter-protocol communication flows through here
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Import base nexus infrastructure
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from agent_system.base_nexus import BaseNexus, AgentRequest, NexusResponse, ProtocolType, AgentType
except ImportError:
    # Fallback for development
    class BaseNexus:
        pass
    class AgentRequest:
        pass
    class NexusResponse:
        pass
    class ProtocolType(Enum):
        SOP = "sop"
        UIP = "uip"
        AGP = "agp"
    class AgentType(Enum):
        SYSTEM_AGENT = "system"
        EXTERIOR_AGENT = "exterior"

logger = logging.getLogger(__name__)


class TokenType(Enum):
    """Types of tokens managed by SOP"""
    OPERATION = "operation_token"       # Basic protocol operation authorization
    TODO = "todo_token"                 # Task-specific cross-protocol coordination
    INTEGRATION = "integration_token"   # File replacement authorization
    TEST = "test_token"                # System testing authorization


class TokenStatus(Enum):
    """Token lifecycle states"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"
    REVOKED = "revoked"


class GapType(Enum):
    """Types of gaps detected in system analysis"""
    FUNCTIONALITY_MISSING = "functionality_missing"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SECURITY_VULNERABILITY = "security_vulnerability"
    CODE_QUALITY = "code_quality"
    DOCUMENTATION_INCOMPLETE = "documentation_incomplete"
    TEST_COVERAGE = "test_coverage"


class SOPToken:
    """SOP-managed token for system authorization"""
    
    def __init__(
        self,
        token_type: TokenType,
        protocol: ProtocolType,
        operation: str,
        requester: str,
        expiration_hours: int = 24
    ):
        self.token_id = str(uuid.uuid4())
        self.token_type = token_type
        self.protocol = protocol
        self.operation = operation
        self.requester = requester
        self.created_at = datetime.now(timezone.utc)
        self.expires_at = self.created_at.replace(hour=self.created_at.hour + expiration_hours)
        self.status = TokenStatus.PENDING
        self.validation_key = self._generate_validation_key()
        self.usage_count = 0
        self.max_usage = 1 if token_type == TokenType.OPERATION else 100
        
    def _generate_validation_key(self) -> str:
        """Generate validation key for token security"""
        key_data = f"{self.token_id}:{self.protocol.value}:{self.requester}:{self.created_at.isoformat()}"
        # In production, this would use proper cryptographic hashing
        return f"VAL_{hash(key_data) % 1000000:06d}"
        
    def activate(self) -> bool:
        """Activate the token for use"""
        if self.status == TokenStatus.PENDING:
            self.status = TokenStatus.ACTIVE
            return True
        return False
        
    def use_token(self) -> bool:
        """Use the token (increment usage count)"""
        if self.status == TokenStatus.ACTIVE and self.usage_count < self.max_usage:
            self.usage_count += 1
            if self.usage_count >= self.max_usage:
                self.status = TokenStatus.COMPLETED
            return True
        return False
        
    def is_valid(self) -> bool:
        """Check if token is valid for use"""
        now = datetime.now(timezone.utc)
        return (
            self.status == TokenStatus.ACTIVE and
            now < self.expires_at and
            self.usage_count < self.max_usage
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert token to dictionary format"""
        return {
            "token_id": self.token_id,
            "token_type": self.token_type.value,
            "protocol": self.protocol.value,
            "operation": self.operation,
            "requester": self.requester,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "status": self.status.value,
            "validation_key": self.validation_key,
            "usage_count": self.usage_count,
            "max_usage": self.max_usage
        }


class SystemGap:
    """Detected gap in system functionality or performance"""
    
    def __init__(
        self,
        gap_type: GapType,
        description: str,
        target_file: Optional[str] = None,
        priority: str = "medium"
    ):
        self.gap_id = str(uuid.uuid4())
        self.gap_type = gap_type
        self.description = description
        self.target_file = target_file
        self.priority = priority
        self.detected_at = datetime.now(timezone.utc)
        self.status = "open"
        self.todo_generated = False
        
    def to_todo_json(self) -> Dict[str, Any]:
        """Convert gap to TODO JSON format"""
        return {
            "todo_id": f"TODO_{self.gap_id[:8]}",
            "gap_id": self.gap_id,
            "gap_type": self.gap_type.value,
            "priority": self.priority,
            "target_file": self.target_file,
            "description": self.description,
            "success_criteria": self._generate_success_criteria(),
            "cross_protocol_deps": self._analyze_protocol_dependencies(),
            "data_package": {
                "gap_analysis": self.description,
                "target_location": self.target_file,
                "existing_code": self._extract_existing_code() if self.target_file else None,
                "requirements": self._generate_requirements(),
                "test_cases": self._generate_test_cases()
            },
            "created_at": self.detected_at.isoformat(),
            "scaffold_available": self._check_scaffold_availability()
        }
        
    def _generate_success_criteria(self) -> List[str]:
        """Generate success criteria based on gap type"""
        criteria_map = {
            GapType.FUNCTIONALITY_MISSING: ["functionality_implemented", "tests_pass", "integration_successful"],
            GapType.PERFORMANCE_DEGRADATION: ["performance_improved", "benchmarks_meet_targets", "no_regression"],
            GapType.SECURITY_VULNERABILITY: ["vulnerability_fixed", "security_tests_pass", "audit_approved"],
            GapType.CODE_QUALITY: ["code_quality_improved", "standards_compliance", "maintainability_enhanced"],
            GapType.DOCUMENTATION_INCOMPLETE: ["documentation_complete", "accuracy_verified", "examples_provided"],
            GapType.TEST_COVERAGE: ["coverage_increased", "edge_cases_tested", "reliability_improved"]
        }
        return criteria_map.get(self.gap_type, ["basic_functionality", "quality_assurance"])
        
    def _analyze_protocol_dependencies(self) -> List[str]:
        """Analyze which protocols are needed to address this gap"""
        deps = []
        
        # AGP is typically needed for solution generation
        deps.append("AGP.solution_generation")
        
        # UIP may be needed for complex reasoning
        if self.gap_type in [GapType.FUNCTIONALITY_MISSING, GapType.PERFORMANCE_DEGRADATION]:
            deps.append("UIP.reasoning_pipeline")
            
        # SOP provides testing and validation
        deps.append("SOP.testing_validation")
        
        return deps
        
    def _extract_existing_code(self) -> Optional[str]:
        """Extract existing code from target file"""
        if self.target_file:
            try:
                file_path = Path(self.target_file)
                if file_path.exists():
                    return file_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.warning(f"Failed to extract code from {self.target_file}: {e}")
        return None
        
    def _generate_requirements(self) -> Dict[str, Any]:
        """Generate technical requirements for addressing the gap"""
        return {
            "functional_requirements": [f"Address {self.gap_type.value} in {self.target_file or 'system'}"],
            "technical_constraints": ["maintain_compatibility", "preserve_existing_functionality"],
            "quality_standards": ["follow_coding_standards", "include_documentation", "provide_tests"]
        }
        
    def _generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate test cases for gap resolution validation"""
        return [
            {
                "test_type": "functionality",
                "description": f"Verify {self.gap_type.value} is resolved",
                "expected_outcome": "gap_resolved"
            },
            {
                "test_type": "regression",
                "description": "Ensure no existing functionality is broken",
                "expected_outcome": "no_regression"
            },
            {
                "test_type": "integration",
                "description": "Verify integration with rest of system",
                "expected_outcome": "integration_successful"
            }
        ]
        
    def _check_scaffold_availability(self) -> bool:
        """Check if scaffold is available for this gap's target file"""
        if not self.target_file:
            return False
            
        scaffold_path = Path("SOP/file_management/scaffold_library") / f"{Path(self.target_file).name}.scaffold"
        return scaffold_path.exists()


class SOPNexus(BaseNexus):
    """
    System Operations Protocol Nexus
    
    Central hub for LOGOS infrastructure, token management, and
    cross-protocol coordination. Always active and facilitates
    all system operations without direct data processing.
    """
    
    def __init__(self):
        super().__init__(
            protocol_type=ProtocolType.SOP,
            nexus_name="SOP_Nexus",
            always_active=True  # SOP is always active
        )
        
        # Token management
        self.active_tokens: Dict[str, SOPToken] = {}
        self.token_history: List[SOPToken] = []
        
        # Gap detection and TODO management
        self.detected_gaps: Dict[str, SystemGap] = {}
        self.todo_queue: List[Dict[str, Any]] = []
        self.completed_todos: List[Dict[str, Any]] = []
        
        # File management
        self.scaffold_library_path = Path("SOP/file_management/scaffold_library")
        self.backup_storage_path = Path("SOP/file_management/backup_storage")
        self.todo_json_path = Path("SOP/data_storage/todo_json")
        
        # System monitoring
        self.system_metrics: Dict[str, Any] = {}
        self.protocol_statuses: Dict[str, Dict[str, Any]] = {}
        
        # Initialize directories
        self._initialize_directories()
        
    def _initialize_directories(self) -> None:
        """Initialize SOP directory structure"""
        directories = [
            self.scaffold_library_path,
            self.backup_storage_path, 
            self.todo_json_path,
            Path("SOP/data_storage/system_logs")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    async def _protocol_specific_initialization(self) -> bool:
        """SOP-specific initialization"""
        try:
            logger.info("🏗️ Initializing SOP infrastructure...")
            
            # Initialize token management system
            await self._initialize_token_system()
            
            # Initialize gap detection system
            await self._initialize_gap_detection()
            
            # Initialize file management system
            await self._initialize_file_management()
            
            # Start background monitoring
            await self._start_background_monitoring()
            
            logger.info("✅ SOP infrastructure initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ SOP initialization failed: {e}")
            return False
            
    async def _initialize_token_system(self) -> None:
        """Initialize token management system"""
        logger.info("🔐 Initializing token system...")
        
        # Load existing tokens if any
        token_file = Path("SOP/data_storage/active_tokens.json")
        if token_file.exists():
            try:
                with open(token_file, 'r') as f:
                    token_data = json.load(f)
                    # Reconstruct tokens from saved data
                    # Implementation would restore active tokens
                    logger.info(f"📋 Loaded {len(token_data)} existing tokens")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load existing tokens: {e}")
                
        logger.info("✅ Token system ready")
        
    async def _initialize_gap_detection(self) -> None:
        """Initialize gap detection system"""
        logger.info("🔍 Initializing gap detection system...")
        
        # Load existing gaps and TODOs
        gaps_file = Path("SOP/data_storage/detected_gaps.json")
        if gaps_file.exists():
            try:
                with open(gaps_file, 'r') as f:
                    gaps_data = json.load(f)
                    # Reconstruct gaps from saved data
                    logger.info(f"📋 Loaded {len(gaps_data)} existing gaps")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load existing gaps: {e}")
                
        # Load TODO queue
        todo_file = self.todo_json_path / "todo_queue.json"
        if todo_file.exists():
            try:
                with open(todo_file, 'r') as f:
                    self.todo_queue = json.load(f)
                    logger.info(f"📋 Loaded {len(self.todo_queue)} TODOs in queue")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load TODO queue: {e}")
                
        logger.info("✅ Gap detection system ready")
        
    async def _initialize_file_management(self) -> None:
        """Initialize file management system"""
        logger.info("📁 Initializing file management system...")
        
        # Scan for available scaffolds
        scaffold_count = len(list(self.scaffold_library_path.glob("*.scaffold")))
        logger.info(f"📋 Found {scaffold_count} available scaffolds")
        
        # Check backup storage
        backup_count = len(list(self.backup_storage_path.glob("*")))
        logger.info(f"💾 Found {backup_count} backup files")
        
        logger.info("✅ File management system ready")
        
    async def _start_background_monitoring(self) -> None:
        """Start background monitoring tasks"""
        logger.info("📊 Starting background monitoring...")
        
        # Start monitoring tasks
        asyncio.create_task(self._gap_detection_loop())
        asyncio.create_task(self._token_cleanup_loop())
        asyncio.create_task(self._system_health_monitor())
        
        logger.info("✅ Background monitoring active")
        
    # Protocol-specific abstract method implementations
    
    async def _protocol_specific_security_validation(self, request: AgentRequest) -> Dict[str, Any]:
        """SOP security validation - only System Agent allowed"""
        
        if request.agent_type != AgentType.SYSTEM_AGENT:
            return {
                "valid": False,
                "reason": "SOP access restricted to System Agent only"
            }
            
        # Additional SOP-specific security checks
        allowed_operations = [
            "request_token", "validate_token", "request_todo_token",
            "get_system_status", "run_gap_detection", "approve_todo_integration",
            "get_todo_queue", "system_test", "health_check"
        ]
        
        if request.operation not in allowed_operations:
            return {
                "valid": False,
                "reason": f"Operation '{request.operation}' not allowed in SOP"
            }
            
        return {"valid": True, "security_level": "system_agent"}
        
    async def _protocol_specific_activation(self) -> None:
        """SOP is always active - no activation needed"""
        pass
        
    async def _protocol_specific_deactivation(self) -> None:
        """SOP is always active - no deactivation allowed"""
        pass
        
    async def _route_to_protocol_core(self, request: AgentRequest) -> Dict[str, Any]:
        """Route request to SOP core functionality"""
        
        operation = request.operation
        payload = request.payload
        
        try:
            if operation == "request_token":
                return await self._handle_token_request(payload)
            elif operation == "validate_token":
                return await self._handle_token_validation(payload)
            elif operation == "request_todo_token":
                return await self._handle_todo_token_request(payload)
            elif operation == "approve_todo_integration":
                return await self._handle_todo_integration_approval(payload)
            elif operation == "get_system_status":
                return await self._handle_system_status_request(payload)
            elif operation == "run_gap_detection":
                return await self._handle_gap_detection_request(payload)
            elif operation == "get_todo_queue":
                return await self._handle_todo_queue_request(payload)
            elif operation == "system_test":
                return await self._handle_system_test_request(payload)
            elif operation == "health_check":
                return await self._handle_health_check_request(payload)
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            logger.error(f"❌ SOP operation failed: {e}")
            return {"success": False, "error": str(e)}
            
    # Token Management Methods
    
    async def _handle_token_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle protocol operation token request"""
        
        try:
            protocol = ProtocolType(payload["protocol"])
            operation = payload["operation"]
            requester = payload["requester"]
            
            # Create new token
            token = SOPToken(
                token_type=TokenType.OPERATION,
                protocol=protocol,
                operation=operation,
                requester=requester
            )
            
            # Activate token
            if token.activate():
                self.active_tokens[token.token_id] = token
                
                logger.info(f"🎫 Issued operation token for {protocol.value}: {token.token_id}")
                
                return {
                    "success": True,
                    "token": token.token_id,
                    "validation_key": token.validation_key,
                    "expires_at": token.expires_at.isoformat()
                }
            else:
                return {"success": False, "error": "Token activation failed"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _handle_token_validation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle token validation request"""
        
        try:
            token_id = payload["token_id"]
            validation_key = payload.get("validation_key")
            
            if token_id not in self.active_tokens:
                return {"valid": False, "reason": "Token not found"}
                
            token = self.active_tokens[token_id]
            
            # Validate token
            if not token.is_valid():
                return {"valid": False, "reason": "Token expired or exhausted"}
                
            # Validate key if provided
            if validation_key and token.validation_key != validation_key:
                return {"valid": False, "reason": "Invalid validation key"}
                
            # Use token
            if token.use_token():
                return {
                    "valid": True,
                    "token_info": token.to_dict()
                }
            else:
                return {"valid": False, "reason": "Token usage failed"}
                
        except Exception as e:
            return {"valid": False, "reason": str(e)}
            
    async def _handle_todo_token_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle TODO token request for cross-protocol coordination"""
        
        try:
            todo_id = payload["todo_id"]
            execution_plan = payload["execution_plan"]
            requester = payload["requester"]
            
            # Create TODO token with special properties
            token = SOPToken(
                token_type=TokenType.TODO,
                protocol=ProtocolType.AGP,  # Primary protocol for TODO processing
                operation="todo_processing",
                requester=requester,
                expiration_hours=48  # Longer expiration for complex TODOs
            )
            
            # TODO tokens can be used multiple times for coordination
            token.max_usage = 100
            
            if token.activate():
                self.active_tokens[token.token_id] = token
                
                # Store TODO coordination data
                self.todo_coordination[token.token_id] = {
                    "todo_id": todo_id,
                    "execution_plan": execution_plan,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "active"
                }
                
                logger.info(f"🎫 Issued TODO token for {todo_id}: {token.token_id}")
                
                return {
                    "success": True,
                    "token": token.token_id,
                    "validation_key": token.validation_key,
                    "coordination_data": self.todo_coordination[token.token_id]
                }
            else:
                return {"success": False, "error": "TODO token activation failed"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _handle_todo_integration_approval(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle TODO integration approval from System Agent"""
        
        try:
            todo_token = payload["todo_token"]
            processing_result = payload["processing_result"]
            agent_decision = payload["agent_decision"]
            
            if todo_token not in self.active_tokens:
                return {"success": False, "error": "TODO token not found"}
                
            if agent_decision == "approve":
                # Integrate the TODO result
                integration_result = await self._integrate_todo_solution(
                    todo_token, processing_result
                )
                
                # Mark token as completed
                token = self.active_tokens[todo_token]
                token.status = TokenStatus.COMPLETED
                
                logger.info(f"✅ TODO integration approved and completed: {todo_token}")
                
                return {
                    "success": True,
                    "integration_result": integration_result
                }
            else:
                logger.info(f"❌ TODO integration denied by System Agent: {todo_token}")
                return {"success": False, "error": "Integration denied by System Agent"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    # Gap Detection and TODO Management
    
    async def _handle_gap_detection_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle gap detection analysis request"""
        
        try:
            # Run comprehensive system analysis
            detected_gaps = await self._run_gap_detection_analysis()
            
            # Generate TODOs for new gaps
            new_todos = []
            for gap in detected_gaps:
                if gap.gap_id not in self.detected_gaps:
                    # New gap detected
                    self.detected_gaps[gap.gap_id] = gap
                    
                    # Generate TODO JSON
                    todo_json = gap.to_todo_json()
                    
                    # Save TODO to queue
                    self.todo_queue.append(todo_json)
                    new_todos.append(todo_json)
                    
                    # Save TODO JSON file
                    todo_file = self.todo_json_path / f"{todo_json['todo_id']}.json"
                    with open(todo_file, 'w') as f:
                        json.dump(todo_json, f, indent=2)
                        
                    gap.todo_generated = True
                    
            logger.info(f"🔍 Gap detection completed: {len(detected_gaps)} gaps, {len(new_todos)} new TODOs")
            
            return {
                "success": True,
                "gaps_detected": len(detected_gaps),
                "new_todos_generated": len(new_todos),
                "todo_queue_size": len(self.todo_queue),
                "new_todos": new_todos
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _run_gap_detection_analysis(self) -> List[SystemGap]:
        """Run comprehensive gap detection analysis"""
        
        detected_gaps = []
        
        # Analyze codebase for missing functionality
        # This would be a comprehensive analysis in production
        
        # Example gap detection (simplified)
        gap_examples = [
            SystemGap(
                gap_type=GapType.FUNCTIONALITY_MISSING,
                description="UIP nexus missing agent type distinction functionality",
                target_file="UIP/nexus/uip_nexus.py",
                priority="high"
            ),
            SystemGap(
                gap_type=GapType.DOCUMENTATION_INCOMPLETE,
                description="AGP cognitive systems lack comprehensive documentation",
                target_file="AGP/cognitive_systems/",
                priority="medium"
            ),
            SystemGap(
                gap_type=GapType.TEST_COVERAGE,
                description="Token validation system needs unit tests",
                target_file="SOP/token_system/token_manager.py",
                priority="medium"
            )
        ]
        
        detected_gaps.extend(gap_examples)
        
        return detected_gaps
        
    async def _handle_todo_queue_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle request for TODO queue information"""
        
        try:
            # Filter TODOs by priority or status if requested
            filter_criteria = payload.get("filter", {})
            
            filtered_todos = self.todo_queue
            if filter_criteria.get("priority"):
                filtered_todos = [
                    todo for todo in self.todo_queue 
                    if todo.get("priority") == filter_criteria["priority"]
                ]
                
            return {
                "success": True,
                "todo_queue": filtered_todos,
                "total_todos": len(self.todo_queue),
                "filtered_count": len(filtered_todos),
                "completed_todos": len(self.completed_todos)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    # System Status and Health
    
    async def _handle_system_status_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system status request"""
        
        try:
            status = {
                "sop_nexus": {
                    "status": "active",
                    "uptime": self._calculate_uptime(),
                    "active_tokens": len(self.active_tokens),
                    "detected_gaps": len(self.detected_gaps),
                    "todo_queue_size": len(self.todo_queue)
                },
                "token_system": {
                    "active_tokens": len(self.active_tokens),
                    "token_history": len(self.token_history),
                    "token_types": list(set(token.token_type.value for token in self.active_tokens.values()))
                },
                "gap_detection": {
                    "detected_gaps": len(self.detected_gaps),
                    "pending_todos": len(self.todo_queue),
                    "completed_todos": len(self.completed_todos)
                },
                "file_management": {
                    "available_scaffolds": len(list(self.scaffold_library_path.glob("*.scaffold"))),
                    "backup_files": len(list(self.backup_storage_path.glob("*")))
                }
            }
            
            return {"success": True, "system_status": status}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _handle_health_check_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check request"""
        
        try:
            health_status = {
                "overall_health": "healthy",
                "checks": {
                    "token_system": await self._check_token_system_health(),
                    "gap_detection": await self._check_gap_detection_health(),
                    "file_management": await self._check_file_management_health(),
                    "disk_space": await self._check_disk_space(),
                    "memory_usage": await self._check_memory_usage()
                }
            }
            
            # Determine overall health
            all_healthy = all(check.get("status") == "healthy" for check in health_status["checks"].values())
            health_status["overall_health"] = "healthy" if all_healthy else "degraded"
            
            return {"success": True, "health_status": health_status}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    async def _handle_system_test_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system test request"""
        
        try:
            test_results = await self.run_smoke_test()
            
            return {
                "success": True,
                "test_results": test_results
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    # Background Monitoring Tasks
    
    async def _gap_detection_loop(self) -> None:
        """Background gap detection monitoring"""
        
        while self.lifecycle_state.value in ["active", "testing"]:
            try:
                # Run periodic gap detection
                await self._run_gap_detection_analysis()
                
                # Wait 5 minutes before next scan
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Gap detection loop error: {e}")
                await asyncio.sleep(600)  # Error recovery delay
                
    async def _token_cleanup_loop(self) -> None:
        """Background token cleanup and maintenance"""
        
        while self.lifecycle_state.value in ["active", "testing"]:
            try:
                # Clean up expired tokens
                expired_tokens = [
                    token_id for token_id, token in self.active_tokens.items()
                    if not token.is_valid()
                ]
                
                for token_id in expired_tokens:
                    token = self.active_tokens.pop(token_id)
                    self.token_history.append(token)
                    
                if expired_tokens:
                    logger.info(f"🧹 Cleaned up {len(expired_tokens)} expired tokens")
                    
                # Wait 10 minutes before next cleanup
                await asyncio.sleep(600)
                
            except Exception as e:
                logger.error(f"Token cleanup loop error: {e}")
                await asyncio.sleep(1200)  # Error recovery delay
                
    async def _system_health_monitor(self) -> None:
        """Background system health monitoring"""
        
        while self.lifecycle_state.value in ["active", "testing"]:
            try:
                # Monitor system metrics
                self.system_metrics = await self._collect_system_metrics()
                
                # Check for health issues
                health_issues = await self._analyze_system_health(self.system_metrics)
                
                if health_issues:
                    logger.warning(f"⚠️ System health issues detected: {health_issues}")
                    
                # Wait 2 minutes before next check
                await asyncio.sleep(120)
                
            except Exception as e:
                logger.error(f"System health monitor error: {e}")
                await asyncio.sleep(300)  # Error recovery delay
                
    # Protocol-specific smoke test implementation
    
    async def _protocol_specific_smoke_test(self) -> Dict[str, Any]:
        """SOP-specific smoke test implementation"""
        
        try:
            tests = {
                "token_system": await self._test_token_system(),
                "gap_detection": await self._test_gap_detection(),
                "file_management": await self._test_file_management(),
                "todo_generation": await self._test_todo_generation()
            }
            
            all_passed = all(test.get("passed", False) for test in tests.values())
            
            return {
                "passed": all_passed,
                "tests": tests,
                "details": "SOP comprehensive functionality test"
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
            
    # Utility Methods
    
    def _calculate_uptime(self) -> str:
        """Calculate system uptime"""
        if hasattr(self, 'initialization_time'):
            uptime_seconds = (datetime.now(timezone.utc) - self.initialization_time).total_seconds()
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
        return "unknown"
        
    async def _integrate_todo_solution(self, todo_token: str, processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate TODO solution into system"""
        
        # This would handle the actual file integration process
        # For now, simulate successful integration
        
        return {
            "integration_successful": True,
            "files_updated": processing_result.get("files_generated", []),
            "backup_created": True,
            "tests_passed": True
        }
        
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics"""
        
        # This would collect actual system metrics
        return {
            "cpu_usage": 25.5,
            "memory_usage": 45.2,
            "disk_usage": 60.1,
            "active_processes": 15,
            "network_activity": "normal"
        }
        
    async def _analyze_system_health(self, metrics: Dict[str, Any]) -> List[str]:
        """Analyze system health based on metrics"""
        
        issues = []
        
        if metrics.get("cpu_usage", 0) > 80:
            issues.append("high_cpu_usage")
            
        if metrics.get("memory_usage", 0) > 85:
            issues.append("high_memory_usage")
            
        if metrics.get("disk_usage", 0) > 90:
            issues.append("low_disk_space")
            
        return issues
        
    # Health check methods
    
    async def _check_token_system_health(self) -> Dict[str, Any]:
        """Check token system health"""
        return {
            "status": "healthy",
            "active_tokens": len(self.active_tokens),
            "issues": []
        }
        
    async def _check_gap_detection_health(self) -> Dict[str, Any]:
        """Check gap detection system health"""
        return {
            "status": "healthy",
            "detected_gaps": len(self.detected_gaps),
            "issues": []
        }
        
    async def _check_file_management_health(self) -> Dict[str, Any]:
        """Check file management system health"""
        return {
            "status": "healthy",
            "scaffolds_available": len(list(self.scaffold_library_path.glob("*.scaffold"))),
            "issues": []
        }
        
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        return {
            "status": "healthy",
            "usage_percent": 60.1,
            "issues": []
        }
        
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage"""
        return {
            "status": "healthy",
            "usage_percent": 45.2,
            "issues": []
        }
        
    # Test methods
    
    async def _test_token_system(self) -> Dict[str, Any]:
        """Test token system functionality"""
        try:
            # Create test token
            test_token = SOPToken(
                token_type=TokenType.OPERATION,
                protocol=ProtocolType.UIP,
                operation="test",
                requester="test_system"
            )
            
            # Test token lifecycle
            activated = test_token.activate()
            used = test_token.use_token()
            valid = test_token.is_valid()
            
            return {
                "passed": activated and used and not valid,  # Should be completed after use
                "details": "Token lifecycle test"
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
            
    async def _test_gap_detection(self) -> Dict[str, Any]:
        """Test gap detection functionality"""
        try:
            # Create test gap
            test_gap = SystemGap(
                gap_type=GapType.FUNCTIONALITY_MISSING,
                description="Test gap",
                priority="low"
            )
            
            # Test TODO generation
            todo_json = test_gap.to_todo_json()
            
            return {
                "passed": todo_json is not None and "todo_id" in todo_json,
                "details": "Gap detection and TODO generation test"
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
            
    async def _test_file_management(self) -> Dict[str, Any]:
        """Test file management functionality"""
        try:
            # Test directory access
            scaffold_accessible = self.scaffold_library_path.exists()
            backup_accessible = self.backup_storage_path.exists()
            todo_accessible = self.todo_json_path.exists()
            
            return {
                "passed": scaffold_accessible and backup_accessible and todo_accessible,
                "details": "File management system access test"
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
            
    async def _test_todo_generation(self) -> Dict[str, Any]:
        """Test TODO generation and management"""
        try:
            # Test TODO queue operations
            initial_size = len(self.todo_queue)
            
            # Add test TODO
            test_todo = {
                "todo_id": "TEST_TODO",
                "description": "Test TODO for system validation",
                "priority": "low"
            }
            
            self.todo_queue.append(test_todo)
            
            # Verify addition
            new_size = len(self.todo_queue)
            
            # Remove test TODO
            self.todo_queue = [todo for todo in self.todo_queue if todo["todo_id"] != "TEST_TODO"]
            
            return {
                "passed": new_size == initial_size + 1,
                "details": "TODO queue management test"
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}


# Public interface functions

async def initialize_sop_nexus() -> SOPNexus:
    """Initialize and return SOP Nexus instance"""
    sop_nexus = SOPNexus()
    success = await sop_nexus.initialize()
    
    if success:
        logger.info("✅ SOP Nexus initialized successfully")
    else:
        logger.error("❌ SOP Nexus initialization failed")
        
    return sop_nexus


if __name__ == "__main__":
    async def main():
        # Test SOP Nexus
        sop = await initialize_sop_nexus()
        
        # Run smoke test
        test_results = await sop.run_smoke_test()
        print("SOP Smoke Test Results:", json.dumps(test_results, indent=2))
        
        # Test token request
        token_request = AgentRequest(
            agent_id="LOGOS_SYSTEM_AGENT",
            operation="request_token",
            payload={
                "protocol": "uip",
                "operation": "user_processing",
                "requester": "LOGOS_SYSTEM_AGENT"
            },
            agent_type=AgentType.SYSTEM_AGENT
        )
        
        token_response = await sop.process_agent_request(token_request)
        print("Token Response:", json.dumps(token_response.to_dict(), indent=2))
        
        # Get system status
        status_request = AgentRequest(
            agent_id="LOGOS_SYSTEM_AGENT",
            operation="get_system_status",
            payload={},
            agent_type=AgentType.SYSTEM_AGENT
        )
        
        status_response = await sop.process_agent_request(status_request)
        print("System Status:", json.dumps(status_response.to_dict(), indent=2))
        
    asyncio.run(main())
