Λ Engine Interface Definitions

Defines the core interfaces between the Λ engine and THŌNOC components.
These interfaces standardize how the Lambda engine interacts with
the Translation Engine, Modal Inference System, and Fractal Database.

Dependencies: typing, abc
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Protocol, TypeVar, Generic
from abc import ABC, abstractmethod
import json

# --- Type Variables ---

T = TypeVar('T')
E = TypeVar('E')  # For expressions
R = TypeVar('R')  # For results

# --- Core Interface Protocols ---

class LambdaExpression(Protocol):
    """Protocol for all Lambda expressions."""
    
    def __str__(self) -> str:
        """String representation of expression."""
        ...

class LambdaType(Protocol):
    """Protocol for type definitions."""
    
    def __str__(self) -> str:
        """String representation of type."""
        ...

class TranslationResult(Protocol):
    """Protocol for translation results."""
    
    @property
    def raw_query(self) -> str:
        """Original query string."""
        ...
    
    @property
    def trinity_vector(self) -> Tuple[float, float, float]:
        """Trinity vector (existence, goodness, truth)."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        ...

class FractalPosition(Protocol):
    """Protocol for fractal position."""
    
    @property
    def c_real(self) -> float:
        """Real component of complex parameter."""
        ...
    
    @property
    def c_imag(self) -> float:
        """Imaginary component of complex parameter."""
        ...
    
    @property
    def in_set(self) -> bool:
        """Whether position is in Mandelbrot set."""
        ...
    
    @property
    def iterations(self) -> int:
        """Iteration count before escape."""
        ...

# --- Interface Abstractions ---

class ITypeSystem(ABC):
    """Interface for type system."""
    
    @abstractmethod
    def check_type(self, expr: LambdaExpression) -> Optional[LambdaType]:
        """Check type of expression.
        
        Args:
            expr: Lambda expression
            
        Returns:
            Type of expression or None if type error
        """
        pass
    
    @abstractmethod
    def is_subtype(self, t1: LambdaType, t2: LambdaType) -> bool:
        """Check if t1 is a subtype of t2.
        
        Args:
            t1: First type
            t2: Second type
            
        Returns:
            True if t1 is a subtype of t2
        """
        pass

class IEvaluator(ABC):
    """Interface for lambda evaluator."""
    
    @abstractmethod
    def evaluate(self, expr: LambdaExpression) -> LambdaExpression:
        """Evaluate expression.
        
        Args:
            expr: Lambda expression
            
        Returns:
            Evaluated expression
        """
        pass
    
    @abstractmethod
    def substitute(self, expr: LambdaExpression, var_name: str, value: LambdaExpression) -> LambdaExpression:
        """Substitute variable in expression.
        
        Args:
            expr: Expression to modify
            var_name: Variable name to replace
            value: Replacement value
            
        Returns:
            Expression with substitution applied
        """
        pass

class IModalBridge(ABC):
    """Interface for modal logic bridge."""
    
    @abstractmethod
    def evaluate_necessity(self, expr: LambdaExpression) -> bool:
        """Evaluate necessity of expression.
        
        Args:
            expr: Lambda expression
            
        Returns:
            True if expression is necessary
        """
        pass
    
    @abstractmethod
    def evaluate_possibility(self, expr: LambdaExpression) -> bool:
        """Evaluate possibility of expression.
        
        Args:
            expr: Lambda expression
            
        Returns:
            True if expression is possible
        """
        pass
    
    @abstractmethod
    def trinity_to_modal(self, trinity_vector: Tuple[float, float, float]) -> Dict[str, Any]:
        """Convert trinity vector to modal status.
        
        Args:
            trinity_vector: (existence, goodness, truth) values
            
        Returns:
            Modal status information
        """
        pass

class IFractalMapper(ABC):
    """Interface for fractal mapping."""
    
    @abstractmethod
    def expr_to_position(self, expr: LambdaExpression) -> FractalPosition:
        """Map expression to fractal position.
        
        Args:
            expr: Lambda expression
            
        Returns:
            Fractal position
        """
        pass
    
    @abstractmethod
    def trinity_to_position(self, trinity_vector: Tuple[float, float, float]) -> FractalPosition:
        """Map trinity vector to fractal position.
        
        Args:
            trinity_vector: (existence, goodness, truth) values
            
        Returns:
            Fractal position
        """
        pass
    
    @abstractmethod
    def find_entailments(self, position: FractalPosition, depth: int = 1) -> List[Tuple[FractalPosition, float]]:
        """Find logical entailments at position.
        
        Args:
            position: Fractal position
            depth: Search depth
            
        Returns:
            List of (position, strength) tuples
        """
        pass

class ITranslationBridge(ABC):
    """Interface for translation engine bridge."""
    
    @abstractmethod
    def expr_to_natural(self, expr: LambdaExpression) -> str:
        """Convert lambda expression to natural language.
        
        Args:
            expr: Lambda expression
            
        Returns:
            Natural language representation
        """
        pass
    
    @abstractmethod
    def natural_to_expr(self, query: str) -> Tuple[LambdaExpression, TranslationResult]:
        """Convert natural language to lambda expression.
        
        Args:
            query: Natural language query
            
        Returns:
            (Lambda expression, Translation result) tuple
        """
        pass
    
    @abstractmethod
    def trinity_to_expr(self, trinity_vector: Tuple[float, float, float]) -> LambdaExpression:
        """Convert trinity vector to lambda expression.
        
        Args:
            trinity_vector: (existence, goodness, truth) values
            
        Returns:
            Corresponding lambda expression
        """
        pass

class IPersistenceBridge(ABC):
    """Interface for knowledge persistence bridge."""
    
    @abstractmethod
    def store_expression(self, expr: LambdaExpression, metadata: Dict[str, Any] = None) -> str:
        """Store lambda expression in knowledge base.
        
        Args:
            expr: Lambda expression
            metadata: Optional additional metadata
            
        Returns:
            Expression identifier
        """
        pass
    
    @abstractmethod
    def retrieve_expression(self, expr_id: str) -> Optional[LambdaExpression]:
        """Retrieve lambda expression from knowledge base.
        
        Args:
            expr_id: Expression identifier
            
        Returns:
            Lambda expression or None if not found
        """
        pass
    
    @abstractmethod
    def find_similar(self, expr: LambdaExpression, limit: int = 5) -> List[Tuple[str, LambdaExpression, float]]:
        """Find similar expressions in knowledge base.
        
        Args:
            expr: Lambda expression
            limit: Maximum results
            
        Returns:
            List of (id, expression, similarity) tuples
        """
        pass

# --- Lambda Engine Interface ---

class ILambdaEngine(ABC):
    """Main interface for Lambda engine."""
    
    @property
    @abstractmethod
    def type_system(self) -> ITypeSystem:
        """Type system component."""
        pass
    
    @property
    @abstractmethod
    def evaluator(self) -> IEvaluator:
        """Evaluator component."""
        pass
    
    @property
    @abstractmethod
    def modal_bridge(self) -> IModalBridge:
        """Modal bridge component."""
        pass
    
    @property
    @abstractmethod
    def fractal_mapper(self) -> IFractalMapper:
        """Fractal mapper component."""
        pass
    
    @property
    @abstractmethod
    def translation_bridge(self) -> ITranslationBridge:
        """Translation bridge component."""
        pass
    
    @property
    @abstractmethod
    def persistence_bridge(self) -> IPersistenceBridge:
        """Persistence bridge component."""
        pass
    
    @abstractmethod
    def parse_expression(self, expr_str: str) -> LambdaExpression:
        """Parse expression string.
        
        Args:
            expr_str: Expression string
            
        Returns:
            Parsed lambda expression
        """
        pass
    
    @abstractmethod
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Processing results
        """
        pass
    
    @abstractmethod
    def create_lambda(self, var_name: str, var_type: str, body_expr: Union[str, LambdaExpression]) -> LambdaExpression:
        """Create lambda abstraction.
        
        Args:
            var_name: Variable name
            var_type: Variable type string (𝔼, 𝔾, 𝕋)
            body_expr: Body expression or string
            
        Returns:
            Lambda abstraction
        """
        pass
    
    @abstractmethod
    def apply(self, func_expr: Union[str, LambdaExpression], arg_expr: Union[str, LambdaExpression]) -> LambdaExpression:
        """Apply function to argument.
        
        Args:
            func_expr: Function expression or string
            arg_expr: Argument expression or string
            
        Returns:
            Function application
        """
        pass