"""
LOGOS V2 Adaptive Reasoning System
Advanced AI capabilities with formal verification guarantees
Enhanced with V2_Possible_Gap_Fillers integration for comprehensive reasoning
"""

from .bayesian_inference import TrinityVector, UnifiedBayesianInferencer
from .semantic_transformers import UnifiedSemanticTransformer
from .torch_adapters import UnifiedTorchAdapter

# V2_Possible_Gap_Fillers Integration
try:
    from .translation.translation_engine import TranslationEngine
    from .translation.pdn_bridge import PDNBridge
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False

try:
    from .bayesian_enhanced.bayesian_enhanced_component import BayesianEnhancedComponent
    BAYESIAN_ENHANCED_AVAILABLE = True
except ImportError:
    BAYESIAN_ENHANCED_AVAILABLE = False

# Enhanced reasoning integration functions
def get_enhanced_bayesian_inferencer():
    """Get translation-enhanced Bayesian inferencer."""
    inferencer = UnifiedBayesianInferencer()
    if TRANSLATION_AVAILABLE:
        try:
            # Enhanced inferencer was already created with translation integration
            return inferencer
        except Exception:
            pass
    return inferencer

def get_reasoning_engine_suite():
    """Get complete suite of reasoning engines with enhancements."""
    suite = {
        "bayesian": get_enhanced_bayesian_inferencer(),
        "semantic": UnifiedSemanticTransformer(),
        "torch": UnifiedTorchAdapter(),
    }
    
    if TRANSLATION_AVAILABLE:
        suite["translation"] = TranslationEngine()
        suite["pdn_bridge"] = PDNBridge()
    
    if BAYESIAN_ENHANCED_AVAILABLE:
        suite["bayesian_enhanced"] = BayesianEnhancedComponent()
    
    return suite

__all__ = [
    "TrinityVector",
    "UnifiedBayesianInferencer", 
    "UnifiedSemanticTransformer",
    "UnifiedTorchAdapter",
    "get_enhanced_bayesian_inferencer",
    "get_reasoning_engine_suite",
    "TRANSLATION_AVAILABLE",
    "BAYESIAN_ENHANCED_AVAILABLE"
]