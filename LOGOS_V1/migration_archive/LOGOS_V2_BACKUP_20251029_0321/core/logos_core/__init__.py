# LOGOS Core - Proof-gated alignment components

from .archon_planner import ArchonPlannerGate
from .integration_harmonizer import IntegrationHarmonizer
from .logos_nexus import LogosNexus
from .reference_monitor import ReferenceMonitor, ProofGateError, KernelHashMismatchError
from .pxl_client import PXLClient
from .unified_formalisms import UnifiedFormalismValidator
from ..learning.autonomous_learning import LearningCycleManager
from .meta_reasoning.iel_generator import IELGenerator
from .meta_reasoning.iel_evaluator import IELEvaluator
from .meta_reasoning.iel_registry import IELRegistry
from ..language.natural_language_processor import NaturalLanguageProcessor
from .runtime.iel_runtime_interface import ModalLogicEvaluator
from ..iel_integration import get_iel_integration, initialize_iel_system, IELIntegration

# Optional API server imports - handle missing dependencies gracefully
try:
    from .api_server import app as api_server_app
    from .server import LogosAPIServer
    from .health_server import HealthMonitor
    from .demo_server import DemoServer
    _api_servers_available = True
except ImportError as e:
    print(f"Warning: API server components not available: {e}")
    # Create fallback objects
    api_server_app = None
    LogosAPIServer = None
    HealthMonitor = None
    DemoServer = None
    _api_servers_available = False

from .governance.iel_signer import IELSigner
from .governance.policy import PolicyManager
from .coherence.coherence_metrics import CoherenceMetrics, TrinityCoherence
from .coherence.coherence_optimizer import CoherenceOptimizer

__all__ = [
    'ArchonPlannerGate',
    'IntegrationHarmonizer',
    'LogosNexus',
    'ReferenceMonitor',
    'ProofGateError',
    'KernelHashMismatchError',
    'PXLClient',
    'UnifiedFormalismValidator',
    'LearningCycleManager',
    'IELGenerator',
    'IELEvaluator',
    'IELRegistry',
    'NaturalLanguageProcessor',
    'ModalLogicEvaluator',
    'IELIntegration',
    'get_iel_integration',
    'initialize_iel_system'
]

# Conditionally add API server components to __all__
if _api_servers_available:
    __all__.extend([
        'api_server_app',
        'LogosAPIServer', 
        'HealthMonitor',
        'DemoServer'
    ])

__all__.extend([
    'IELSigner',
    'PolicyManager',
    'CoherenceMetrics',
    'TrinityCoherence', 
    'CoherenceOptimizer'
])
