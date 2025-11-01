"""
Enhanced Arithmopraxis Infrastructure: Mathematical Reasoning Praxis

This infrastructure focuses on the praxis of mathematical reasoning, including:
- Trinity-grounded arithmetic computation and symbolic manipulation
- Ontologically-validated proof generation and verification
- Fractal-enhanced mathematical modeling and analysis
- Lambda calculus-integrated algorithmic mathematics

Enhanced with V2_Possible_Gap_Fillers integration:
- Lambda Engine for ontological computation
- Fractal Orbital Predictor for mathematical optimization
- Ontological Validator for proof verification
- Translation Engine for semantic mathematical processing
"""

"""
Enhanced Arithmopraxis Infrastructure: Mathematical Reasoning Praxis

This infrastructure focuses on the praxis of mathematical reasoning, including:
- Trinity-grounded arithmetic computation and symbolic manipulation
- Ontologically-validated proof generation and verification
- Fractal-enhanced mathematical modeling and analysis
- Lambda calculus-integrated algorithmic mathematics

Enhanced with V2_Possible_Gap_Fillers integration:
- Lambda Engine for ontological computation
- Fractal Orbital Predictor for mathematical optimization
- Ontological Validator for proof verification
- Translation Engine for semantic mathematical processing
"""

from .arithmopraxis.arithmetic_engine import ArithmeticEngine, TrinityArithmeticEngine
from .arithmopraxis.proof_engine import OntologicalProofEngine, ProofEngine, ProofResult, ProofStatus
from .arithmopraxis.symbolic_math import FractalSymbolicMath, SymbolicMath

__all__ = [
    "TrinityArithmeticEngine",
    "ArithmeticEngine",
    "FractalSymbolicMath",
    "SymbolicMath",
    "OntologicalProofEngine",
    "ProofEngine",
    "ProofResult",
    "ProofStatus",
]

# Version and capabilities information
__version__ = "2.0.0"
__enhanced__ = True


def get_arithmopraxis_info():
    """Get information about Arithmopraxis capabilities."""
    return {
        "version": __version__,
        "enhanced": __enhanced__,
        "components": {
            "arithmetic_engine": "Trinity-grounded high-precision computation",
            "symbolic_math": "Fractal-enhanced symbolic mathematics",
            "proof_engine": "Ontologically-validated theorem proving",
        },
        "integrations": {
            "lambda_engine": "Ontological computation via Lambda calculus",
            "fractal_orbital": "Mathematical optimization via fractal analysis",
            "ontological_validator": "Trinity-grounded proof validation",
            "translation_engine": "Semantic mathematical processing",
        },
    }
