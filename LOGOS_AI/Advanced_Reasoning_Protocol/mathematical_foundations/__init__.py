"""
LOGOS Mathematical Foundations

Complete Trinity-grounded mathematical foundation for the LOGOS AGI system.
Provides core mathematical operations, symbolic computation, and formal verification.
"""

# Core mathematical components
from .cores.logos_mathematical_core import (
    Quaternion,
    TrinityOptimizer,
    TrinityFractalSystem,
    OrbitAnalysis
)

# Mathematical categories and engines
from .math_categories import (
    get_enhanced_arithmetic_engine,
    get_enhanced_symbolic_processor,
    get_enhanced_proof_engine,
    ArithmeticEngine,
    TrinityArithmeticEngine,
    SymbolicMath,
    FractalSymbolicMath,
    ProofEngine,
    OntologicalProofEngine,
    ProofResult,
    ProofStatus
)

# PXL (Protopraxic Logic) components
from .pxl import (
    TrinityArithmeticEngine as PXL_TrinityArithmeticEngine,
    ArithmeticEngine as PXL_ArithmeticEngine,
    FractalSymbolicMath as PXL_FractalSymbolicMath,
    SymbolicMath as PXL_SymbolicMath,
    OntologicalProofEngine as PXL_OntologicalProofEngine,
    ProofEngine as PXL_ProofEngine,
    ProofResult as PXL_ProofResult,
    ProofStatus as PXL_ProofStatus
)

__version__ = "2.0.0"
__all__ = [
    # Core components
    "Quaternion",
    "TrinityOptimizer",
    "TrinityFractalSystem",
    "OrbitAnalysis",

    # Enhanced engines
    "get_enhanced_arithmetic_engine",
    "get_enhanced_symbolic_processor",
    "get_enhanced_proof_engine",

    # Direct imports
    "ArithmeticEngine",
    "TrinityArithmeticEngine",
    "SymbolicMath",
    "FractalSymbolicMath",
    "ProofEngine",
    "OntologicalProofEngine",
    "ProofResult",
    "ProofStatus",

    # PXL aliases
    "PXL_TrinityArithmeticEngine",
    "PXL_ArithmeticEngine",
    "PXL_FractalSymbolicMath",
    "PXL_SymbolicMath",
    "PXL_OntologicalProofEngine",
    "PXL_ProofEngine",
    "PXL_ProofResult",
    "PXL_ProofStatus",
]


def get_mathematical_foundations_info():
    """Get information about mathematical foundations capabilities."""
    return {
        "version": __version__,
        "components": {
            "core": "Trinity-grounded quaternion mathematics and optimization",
            "fractal_system": "Fractal analysis for cognitive coherence",
            "arithmetic_engine": "High-precision computation with Lambda integration",
            "symbolic_math": "Fractal-enhanced symbolic mathematics",
            "proof_engine": "Ontologically-validated theorem proving",
        },
        "theorem_verification": {
            "trinity_optimization": "O(n) minimized at n=3 ✓",
            "mathematical_proof": "Formal verification ready",
        }
    }