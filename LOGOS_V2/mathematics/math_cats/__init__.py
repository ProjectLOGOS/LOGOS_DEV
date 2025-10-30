"""
Arithmopraxis Infrastructure: Mathematical Reasoning Praxis

This infrastructure focuses on the praxis of mathematical reasoning, including:
- Arithmetic computation and symbolic manipulation
- Proof generation and verification
- Mathematical modeling and analysis
- Algorithmic mathematics
- Trinity-enhanced mathematical processing (V2_Gap_Fillers integrated)
"""

from .arithmopraxis.arithmetic_engine import ArithmeticEngine
from .arithmopraxis.symbolic_math import SymbolicMath
from .arithmopraxis.proof_engine import ProofEngine

# V2_Possible_Gap_Fillers Enhanced Mathematical Integration
try:
    from ..pxl.arithmopraxis.trinity_arithmetic_engine import TrinityArithmeticEngine
    from ..pxl.arithmopraxis.fractal_symbolic_math import FractalSymbolicMath
    from ..pxl.arithmopraxis.ontological_proof_engine import OntologicalProofEngine
    
    # Enhanced mathematical processing functions
    def get_enhanced_arithmetic_engine():
        """Get Trinity-enhanced arithmetic engine with fallback."""
        try:
            return TrinityArithmeticEngine()
        except Exception:
            return ArithmeticEngine()
    
    def get_enhanced_symbolic_processor():
        """Get fractal-enhanced symbolic processor with fallback."""
        try:
            return FractalSymbolicMath()
        except Exception:
            return SymbolicMath()
    
    def get_enhanced_proof_engine():
        """Get ontologically-enhanced proof engine with fallback."""
        try:
            return OntologicalProofEngine()
        except Exception:
            return ProofEngine()
    
    ENHANCED_COMPONENTS_AVAILABLE = True
    
except ImportError as e:
    # Fallback when enhanced components not available
    def get_enhanced_arithmetic_engine():
        return ArithmeticEngine()
    
    def get_enhanced_symbolic_processor():
        return SymbolicMath()
    
    def get_enhanced_proof_engine():
        return ProofEngine()
    
    ENHANCED_COMPONENTS_AVAILABLE = False

__all__ = [
    'ArithmeticEngine', 'SymbolicMath', 'ProofEngine',
    'get_enhanced_arithmetic_engine', 'get_enhanced_symbolic_processor', 
    'get_enhanced_proof_engine', 'ENHANCED_COMPONENTS_AVAILABLE'
]