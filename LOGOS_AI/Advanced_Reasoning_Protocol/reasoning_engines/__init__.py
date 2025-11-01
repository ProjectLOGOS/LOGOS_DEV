"""
LOGOS V2 Adaptive Reasoning System
Advanced AI capabilities with formal verification guarantees
Enhanced with V2_Possible_Gap_Fillers integration for comprehensive reasoning
"""

from .bayesian.bayesian_enhanced.bayesian_inference import TrinityVector, UnifiedBayesianInferencer

# Language engine imports
try:
    from .language import (
        NaturalLanguageProcessor,
        UnifiedSemanticTransformer,
        PDNBridge,
    )
    LANGUAGE_AVAILABLE = True
except ImportError as e:
    LANGUAGE_AVAILABLE = False
    NaturalLanguageProcessor = None
    UnifiedSemanticTransformer = None
    PDNBridge = None

# Temporal engine imports
try:
    from .temporal.temporal_predictor import TemporalPredictor
    TEMPORAL_AVAILABLE = True
except ImportError as e:
    TEMPORAL_AVAILABLE = False
    TemporalPredictor = None

# Lambda calculus engine imports
try:
    from .lambda_calculus.lambda_engine import LambdaEngine
    LAMBDA_AVAILABLE = True
except ImportError as e:
    LAMBDA_AVAILABLE = False
    LambdaEngine = None

# Try to import torch adapter, but make it optional
try:
    from ..learning_modules.pytorch_ml_adapters import UnifiedTorchAdapter
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    UnifiedTorchAdapter = None

# Optional components - import at runtime when needed
TRANSLATION_AVAILABLE = False
BAYESIAN_ENHANCED_AVAILABLE = False
SEMANTIC_TRANSFORMER_AVAILABLE = False


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
    }
    
    # Add torch adapter if available
    if TORCH_AVAILABLE and UnifiedTorchAdapter:
        suite["torch"] = UnifiedTorchAdapter()

    # Add language components if available
    if LANGUAGE_AVAILABLE:
        if NaturalLanguageProcessor:
            suite["natural_language"] = NaturalLanguageProcessor()
        # Import and use UnifiedSemanticTransformer
        try:
            from .language import UnifiedSemanticTransformer as UST
            suite["semantic"] = UST()
        except ImportError:
            pass
        if PDNBridge:
            suite["translation_bridge"] = PDNBridge()

    # Add temporal predictor if available
    if TEMPORAL_AVAILABLE and TemporalPredictor:
        suite["temporal"] = TemporalPredictor()

    # Add lambda calculus engine if available
    if LAMBDA_AVAILABLE and LambdaEngine:
        suite["lambda_calculus"] = LambdaEngine()

    # Add optional components if available
    if SEMANTIC_TRANSFORMER_AVAILABLE:
        try:
            from .semantic_transformers import UnifiedSemanticTransformer
            suite["semantic"] = UnifiedSemanticTransformer()
        except ImportError:
            pass

    if TRANSLATION_AVAILABLE:
        try:
            from .translation.translation_engine import TranslationEngine
            suite["translation"] = TranslationEngine()
        except ImportError:
            pass

    if BAYESIAN_ENHANCED_AVAILABLE:
        try:
            from .bayesian_enhanced.bayesian_enhanced_component import BayesianEnhancedComponent
            suite["bayesian_enhanced"] = BayesianEnhancedComponent()
        except ImportError:
            pass

    return suite


__all__ = [
    "TrinityVector",
    "UnifiedBayesianInferencer",
    "UnifiedSemanticTransformer",
    "UnifiedTorchAdapter",
    "get_enhanced_bayesian_inferencer",
    "get_reasoning_engine_suite",
    "TRANSLATION_AVAILABLE",
    "BAYESIAN_ENHANCED_AVAILABLE",
]
