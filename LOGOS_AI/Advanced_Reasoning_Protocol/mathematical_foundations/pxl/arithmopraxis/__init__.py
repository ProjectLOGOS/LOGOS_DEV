"""
Arithmopraxis: Mathematical Reasoning Praxis

Core mathematical reasoning components with Trinity grounding and Lambda integration.
"""

from .arithmetic_engine import ArithmeticEngine, TrinityArithmeticEngine
from .proof_engine import ProofEngine, OntologicalProofEngine, ProofResult, ProofStatus
from .symbolic_math import SymbolicMath, FractalSymbolicMath

__all__ = [
    "ArithmeticEngine",
    "TrinityArithmeticEngine",
    "ProofEngine",
    "OntologicalProofEngine",
    "ProofResult",
    "ProofStatus",
    "SymbolicMath",
    "FractalSymbolicMath",
]