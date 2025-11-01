"""
LOGOS V2 Language Systems
Natural language processing and communication capabilities
"""

# Natural Language Processing Components
from .natural_language_processor import (
    ConversationContext,
    LogicTranslator,
    CoqTranslator,
    NaturalLanguageProcessor,
)

# Semantic Transformation Components
from .semantic_transformers import (
    SemanticEmbedding,
    SemanticTransformation,
    UnifiedSemanticTransformer,
    encode_semantics,
    detect_concept_drift,
    example_semantic_transformation,
)

# Translation Bridge Components
from .tranlsation_bridge import (
    TranslationResult,
    PDNBridge,
    PDNBottleneckSolver,
)

__all__ = [
    # Natural Language Processing
    "ConversationContext",
    "LogicTranslator",
    "CoqTranslator",
    "NaturalLanguageProcessor",

    # Semantic Transformers
    "SemanticEmbedding",
    "SemanticTransformation",
    "UnifiedSemanticTransformer",
    "encode_semantics",
    "detect_concept_drift",
    "example_semantic_transformation",

    # Translation Engine
    "convert_to_nl",

    # Translation Bridge
    "TranslationResult",
    "PDNBridge",
    "PDNBottleneckSolver",
]
