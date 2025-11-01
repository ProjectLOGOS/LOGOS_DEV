# LOGOS Core - Proof-gated alignment components

from ..iel_integration import IELIntegration, get_iel_integration, initialize_iel_system
from ..language.natural_language_processor import NaturalLanguageProcessor
from ..learning.autonomous_learning import LearningCycleManager
from .archon_planner import ArchonPlannerGate
from ....alignment_protocols.compliance.integration_harmonizer import IntegrationHarmonizer
from .logos_nexus import LogosNexus
from .meta_reasoning.iel_evaluator import IELEvaluator
from .meta_reasoning.iel_generator import IELGenerator
from .meta_reasoning.iel_registry import IELRegistry
from .pxl_client import PXLClient
from .reference_monitor import KernelHashMismatchError, ProofGateError, ReferenceMonitor
from .runtime.iel_runtime_interface import ModalLogicEvaluator
from .unified_formalisms import UnifiedFormalismValidator

# Optional API server imports - handle missing dependencies gracefully
try:
    from ....deployment.monitoring.api_server import app as api_server_app
    from .demo_server import DemoServer
    from ....deployment.monitoring.health_server import HealthMonitor
    from .server import LogosAPIServer

    _api_servers_available = True
except ImportError as e:
    print(f"Warning: API server components not available: {e}")
    # Create fallback objects
    api_server_app = None
    LogosAPIServer = None
    HealthMonitor = None
    DemoServer = None
    _api_servers_available = False

from .coherence.coherence_metrics import CoherenceMetrics, TrinityCoherence
from .coherence.coherence_optimizer import CoherenceOptimizer
from .meta_reasoning.iel_signer import IELSigner
from .coherence.policy import PolicyManager

__all__ = [
    "ArchonPlannerGate",
    "IntegrationHarmonizer",
    "LogosNexus",
    "ReferenceMonitor",
    "ProofGateError",
    "KernelHashMismatchError",
    "PXLClient",
    "UnifiedFormalismValidator",
    "LearningCycleManager",
    "IELGenerator",
    "IELEvaluator",
    "IELRegistry",
    "NaturalLanguageProcessor",
    "ModalLogicEvaluator",
    "IELIntegration",
    "get_iel_integration",
    "initialize_iel_system",
]

# Conditionally add API server components to __all__
if _api_servers_available:
    __all__.extend(["api_server_app", "LogosAPIServer", "HealthMonitor", "DemoServer"])

__all__.extend(
    [
        "IELSigner",
        "PolicyManager",
        "CoherenceMetrics",
        "TrinityCoherence",
        "CoherenceOptimizer",
    ]
)
