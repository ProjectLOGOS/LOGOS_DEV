"""
Core Λ Engine - Ontological Lambda Calculus

This module implements the core Lambda engine that serves as the foundation
for the THŌNOC system, supporting:
- Typed lambda calculus with trinitarian ontological types
- Modal logic inference
- Fractal knowledge representation
- Sufficient reason operators

Dependencies: sympy, typing
"""

from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple, Callable
from enum import Enum
import math
import json

# --- Core Lambda Types ---

class OntologicalType(Enum):
    """Ontological dimensions in trinitarian framework."""
    EXISTENCE = "𝔼"  # Existence
    GOODNESS = "𝔾"   # Goodness
    TRUTH = "𝕋"      # Truth

class ModalOperator(Enum):
    """Modal logic operators."""
    NECESSARILY = "□"  # Box
    POSSIBLY = "◇"     # Diamond
    ACTUALLY = "A"     # Actuality

class LogicalLaw(Enum):
    """Fundamental logical laws with bijective mapping to ontological dimensions."""
    IDENTITY = "ID"             # Maps to EXISTENCE
    NON_CONTRADICTION = "NC"    # Maps to GOODNESS
    EXCLUDED_MIDDLE = "EM"      # Maps to TRUTH

# --- Bijective Mapping Between Ontology and Logic ---

ONTO_LOGIC_BIJECTION = {
    OntologicalType.EXISTENCE: LogicalLaw.IDENTITY,
    OntologicalType.GOODNESS: LogicalLaw.NON_CONTRADICTION,
    OntologicalType.TRUTH: LogicalLaw.EXCLUDED_MIDDLE
}

LOGIC_ONTO_BIJECTION = {
    LogicalLaw.IDENTITY: OntologicalType.EXISTENCE,
    LogicalLaw.NON_CONTRADICTION: OntologicalType.GOODNESS,
    LogicalLaw.EXCLUDED_MIDDLE: OntologicalType.TRUTH
}

# --- Lambda Expressions ---

class LambdaExpr:
    """Base class for all lambda expressions."""
    pass

class Variable(LambdaExpr):
    """Variable in lambda calculus."""
    
    def __init__(self, name: str, onto_type: OntologicalType):
        """Initialize variable with name and type.
        
        Args:
            name: Variable name
            onto_type: Ontological type
        """
        self.name = name
        self.onto_type = onto_type
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}:{self.onto_type.value}"

class Abstraction(LambdaExpr):
    """Lambda abstraction (λx.M)."""
    
    def __init__(self, var_name: str, var_type: OntologicalType, body: LambdaExpr):
        """Initialize lambda abstraction.
        
        Args:
            var_name: Variable name
            var_type: Variable type
            body: Function body
        """
        self.var_name = var_name
        self.var_type = var_type
        self.body = body
    
    def __str__(self) -> str:
        """String representation."""
        return f"λ{self.var_name}:{self.var_type.value}.{self.body}"

class Application(LambdaExpr):
    """Function application (M N)."""
    
    def __init__(self, func: LambdaExpr, arg: LambdaExpr):
        """Initialize function application.
        
        Args:
            func: Function
            arg: Argument
        """
        self.func = func
        self.arg = arg
    
    def __str__(self) -> str:
        """String representation."""
        return f"({self.func} {self.arg})"

class SufficientReason(LambdaExpr):
    """Sufficient reason operator (SR)."""
    
    def __init__(self, source_type: OntologicalType, target_type: OntologicalType, value: int):
        """Initialize sufficient reason operator.
        
        Args:
            source_type: Source ontological dimension
            target_type: Target ontological dimension
            value: Sufficient reason value
        """
        self.source_type = source_type
        self.target_type = target_type
        self.value = value
    
    def __str__(self) -> str:
        """String representation."""
        return f"SR[{self.source_type.value},{self.target_type.value}]={self.value}"

# --- Type System ---

class FunctionType:
    """Function type constructor."""
    
    def __init__(self, domain: OntologicalType, codomain: OntologicalType):
        """Initialize function type.
        
        Args:
            domain: Input type
            codomain: Output type
        """
        self.domain = domain
        self.codomain = codomain
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.domain.value} → {self.codomain.value}"

class TypeContext:
    """Type context for variable bindings."""
    
    def __init__(self):
        """Initialize empty type context."""
        self.bindings = {}
    
    def add(self, var_name: str, onto_type: Union[OntologicalType, FunctionType]) -> None:
        """Add variable binding.
        
        Args:
            var_name: Variable name
            onto_type: Ontological or function type
        """
        self.bindings[var_name] = onto_type
    
    def get(self, var_name: str) -> Optional[Union[OntologicalType, FunctionType]]:
        """Get variable type.
        
        Args:
            var_name: Variable name
            
        Returns:
            Bound type or None if not found
        """
        return self.bindings.get(var_name)

# --- Type Checker ---

class TypeChecker:
    """Type checking system for lambda expressions."""
    
    def __init__(self):
        """Initialize type checker."""
        self.context = TypeContext()
        
        # Add built-in SR operators
        self.add_sr_operator(OntologicalType.EXISTENCE, OntologicalType.GOODNESS, 3)
        self.add_sr_operator(OntologicalType.GOODNESS, OntologicalType.TRUTH, 2)
    
    def add_sr_operator(self, source: OntologicalType, target: OntologicalType, value: int) -> None:
        """Add sufficient reason operator.
        
        Args:
            source: Source ontological dimension
            target: Target ontological dimension
            value: Sufficient reason value
        """
        name = f"SR_{source.value}_{target.value}"
        self.context.add(name, FunctionType(source, target))
    
    def type_check(self, expr: LambdaExpr) -> Optional[Union[OntologicalType, FunctionType]]:
        """Check type of lambda expression.
        
        Args:
            expr: Lambda expression
            
        Returns:
            Type of expression or None if type error
        """
        if isinstance(expr, Variable):
            return self.context.get(expr.name)
        
        elif isinstance(expr, Abstraction):
            # Add variable to context
            old_type = self.context.get(expr.var_name)
            self.context.add(expr.var_name, expr.var_type)
            
            # Check body type
            body_type = self.type_check(expr.body)
            
            # Restore original binding if any
            if old_type:
                self.context.add(expr.var_name, old_type)
            else:
                del self.context.bindings[expr.var_name]
            
            if body_type:
                return FunctionType(expr.var_type, body_type)
            return None
        
        elif isinstance(expr, Application):
            # Check function type
            func_type = self.type_check(expr.func)
            
            if not isinstance(func_type, FunctionType):
                print(f"Type error: Expected function, got {func_type}")
                return None
            
            # Check argument type
            arg_type = self.type_check(expr.arg)
            
            if func_type.domain != arg_type:
                print(f"Type error: Expected {func_type.domain}, got {arg_type}")
                return None
            
            return func_type.codomain
        
        elif isinstance(expr, SufficientReason):
            # Check SR operator validity
            if expr.source_type == OntologicalType.EXISTENCE and expr.target_type == OntologicalType.GOODNESS:
                if expr.value == 3:
                    return FunctionType(expr.source_type, expr.target_type)
            elif expr.source_type == OntologicalType.GOODNESS and expr.target_type == OntologicalType.TRUTH:
                if expr.value == 2:
                    return FunctionType(expr.source_type, expr.target_type)
            
            print(f"Type error: Invalid SR operator: {expr}")
            return None
        
        print(f"Type error: Unknown expression type: {type(expr)}")
        return None

# --- Evaluator ---

class Evaluator:
    """Lambda calculus evaluator."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.type_checker = TypeChecker()
    
    def substitute(self, expr: LambdaExpr, var_name: str, value: LambdaExpr) -> LambdaExpr:
        """Substitute variable in expression.
        
        Args:
            expr: Expression to modify
            var_name: Variable name to replace
            value: Replacement value
            
        Returns:
            Expression with substitution applied
        """
        if isinstance(expr, Variable):
            if expr.name == var_name:
                return value
            return expr
        
        elif isinstance(expr, Abstraction):
            if expr.var_name == var_name:
                # Variable is bound in this scope, don't substitute
                return expr
            
            # Substitute in body
            new_body = self.substitute(expr.body, var_name, value)
            return Abstraction(expr.var_name, expr.var_type, new_body)
        
        elif isinstance(expr, Application):
            # Substitute in function and argument
            new_func = self.substitute(expr.func, var_name, value)
            new_arg = self.substitute(expr.arg, var_name, value)
            return Application(new_func, new_arg)
        
        elif isinstance(expr, SufficientReason):
            # SR doesn't contain variables
            return expr
        
        return expr
    
    def evaluate(self, expr: LambdaExpr) -> LambdaExpr:
        """Evaluate lambda expression (beta reduction).
        
        Args:
            expr: Lambda expression
            
        Returns:
            Evaluated expression
        """
        if isinstance(expr, Variable) or isinstance(expr, SufficientReason):
            # Variables and SR operators evaluate to themselves
            return expr
        
        elif isinstance(expr, Abstraction):
            # Evaluate body
            new_body = self.evaluate(expr.body)
            return Abstraction(expr.var_name, expr.var_type, new_body)
        
        elif isinstance(expr, Application):
            # Evaluate function and argument
            func = self.evaluate(expr.func)
            arg = self.evaluate(expr.arg)
            
            # Apply beta reduction if possible
            if isinstance(func, Abstraction):
                # (λx.M) N -> M[x:=N]
                substituted = self.substitute(func.body, func.var_name, arg)
                return self.evaluate(substituted)
            
            return Application(func, arg)
        
        return expr

# --- Modal Logic Integration ---

class ModalExpr:
    """Modal logic expression."""
    
    def __init__(self, operator: ModalOperator, content: LambdaExpr):
        """Initialize modal expression.
        
        Args:
            operator: Modal operator
            content: Expression content
        """
        self.operator = operator
        self.content = content
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.operator.value}({self.content})"

class ModalInference:
    """Modal inference system."""
    
    def __init__(self):
        """Initialize modal inference system."""
        self.evaluator = Evaluator()
    
    def necessarily(self, expr: LambdaExpr) -> ModalExpr:
        """Apply necessity operator.
        
        Args:
            expr: Expression
            
        Returns:
            Modal expression with necessity
        """
        return ModalExpr(ModalOperator.NECESSARILY, expr)
    
    def possibly(self, expr: LambdaExpr) -> ModalExpr:
        """Apply possibility operator.
        
        Args:
            expr: Expression
            
        Returns:
            Modal expression with possibility
        """
        return ModalExpr(ModalOperator.POSSIBLY, expr)
    
    def evaluate_modal(self, expr: ModalExpr) -> bool:
        """Evaluate modal expression truth.
        
        Args:
            expr: Modal expression
            
        Returns:
            Truth value
        """
        # Evaluate inner content
        content = self.evaluator.evaluate(expr.content)
        
        # Basic S5 evaluation (placeholder)
        if expr.operator == ModalOperator.NECESSARILY:
            # For now, necessity is true for expressions using only SR operators
            return isinstance(content, SufficientReason)
        
        elif expr.operator == ModalOperator.POSSIBLY:
            # Possibility is less strict
            return True
        
        return False

# --- Fractal Integration ---

class FractalMapping:
    """Maps ontological values to fractal space."""
    
    def __init__(self):
        """Initialize fractal mapping."""
        pass
    
    def map_to_complex(self, trinity_vector: Tuple[float, float, float]) -> complex:
        """Map trinity vector to complex plane.
        
        Args:
            trinity_vector: (existence, goodness, truth) values
            
        Returns:
            Complex number for fractal position
        """
        existence, goodness, truth = trinity_vector
        return complex(existence * truth, goodness)
    
    def compute_iterations(self, c: complex, max_iter: int = 100) -> Tuple[int, bool]:
        """Compute Mandelbrot iterations for point.
        
        Args:
            c: Complex parameter
            max_iter: Maximum iterations
            
        Returns:
            (iterations, in_set) tuple
        """
        z = complex(0, 0)
        for i in range(max_iter):
            z = z * z + c
            if abs(z) > 2.0:
                return i, False
        return max_iter, True

# --- Lambda Engine Core ---

class LambdaEngine:
    """Core Lambda engine for THŌNOC system."""
    
    def __init__(self):
        """Initialize Lambda engine."""
        self.type_checker = TypeChecker()
        self.evaluator = Evaluator()
        self.modal = ModalInference()
        self.fractal = FractalMapping()
    
    def create_variable(self, name: str, onto_type: OntologicalType) -> Variable:
        """Create variable with specified type.
        
        Args:
            name: Variable name
            onto_type: Ontological type
            
        Returns:
            Variable
        """
        return Variable(name, onto_type)
    
    def create_abstraction(self, var_name: str, var_type: OntologicalType, body: LambdaExpr) -> Abstraction:
        """Create lambda abstraction.
        
        Args:
            var_name: Variable name
            var_type: Variable type
            body: Function body
            
        Returns:
            Lambda abstraction
        """
        return Abstraction(var_name, var_type, body)
    
    def create_application(self, func: LambdaExpr, arg: LambdaExpr) -> Application:
        """Create function application.
        
        Args:
            func: Function
            arg: Argument
            
        Returns:
            Function application
        """
        return Application(func, arg)
    
    def create_sr(self, source: OntologicalType, target: OntologicalType, value: int) -> SufficientReason:
        """Create sufficient reason operator.
        
        Args:
            source: Source ontological dimension
            target: Target ontological dimension
            value: Sufficient reason value
            
        Returns:
            Sufficient reason operator
        """
        return SufficientReason(source, target, value)
    
    def check_type(self, expr: LambdaExpr) -> Optional[Union[OntologicalType, FunctionType]]:
        """Check type of lambda expression.
        
        Args:
            expr: Lambda expression
            
        Returns:
            Type of expression or None if type error
        """
        return self.type_checker.type_check(expr)
    
    def evaluate(self, expr: LambdaExpr) -> LambdaExpr:
        """Evaluate lambda expression.
        
        Args:
            expr: Lambda expression
            
        Returns:
            Evaluated expression
        """
        return self.evaluator.evaluate(expr)
    
    def map_to_fractal(self, trinity_vector: Tuple[float, float, float]) -> Dict[str, Any]:
        """Map trinity vector to fractal space.
        
        Args:
            trinity_vector: (existence, goodness, truth) values
            
        Returns:
            Fractal position information
        """
        c = self.fractal.map_to_complex(trinity_vector)
        iterations, in_set = self.fractal.compute_iterations(c)
        
        return {
            "c": (c.real, c.imag),
            "iterations": iterations,
            "in_set": in_set
        }
    
    def trinity_to_modal(self, trinity_vector: Tuple[float, float, float]) -> str:
        """Determine modal status from trinity vector.
        
        Args:
            trinity_vector: (existence, goodness, truth) values
            
        Returns:
            Modal status string
        """
        existence, goodness, truth = trinity_vector
        
        # Calculate coherence
        ideal_g = existence * truth
        coherence = goodness / ideal_g if ideal_g > 0 else 0.0
        if goodness >= ideal_g:
            coherence = 1.0
        
        # Determine modal status
        if truth > 0.95 and existence > 0.9 and coherence > 0.9:
            return "necessary"
        elif truth > 0.5 and existence > 0.5:
            return "actual"
        elif truth > 0.05 and existence > 0.05:
            return "possible"
        else:
            return "impossible"
    
    def process_expression(self, expr_str: str) -> Dict[str, Any]:
        """Process lambda expression string.
        
        Args:
            expr_str: Expression string
            
        Returns:
            Processing results
        """
        # This would include parsing, type checking, evaluation
        # For now, return a placeholder
        return {
            "parsed": expr_str,
            "type_checked": True,
            "evaluated": expr_str,
            "modal_status": "possible"
        }
    
    def bridge_to_3pdn(self, expr: LambdaExpr) -> Dict[str, Any]:
        """Bridge to 3PDN translation layer.
        
        Args:
            expr: Lambda expression
            
        Returns:
            3PDN translation data
        """
        # This would connect to the 3PDN translation engine
        # For now, return a placeholder
        return {
            "SIGN": str(expr),
            "MIND": "ontological",
            "BRIDGE": {
                "existence": 0.8,
                "goodness": 0.7,
                "truth": 0.9
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize Lambda engine
    engine = LambdaEngine()
    
    # Create lambda expressions
    x = engine.create_variable("x", OntologicalType.EXISTENCE)
    sr_eg = engine.create_sr(OntologicalType.EXISTENCE, OntologicalType.GOODNESS, 3)
    sr_gt = engine.create_sr(OntologicalType.GOODNESS, OntologicalType.TRUTH, 2)
    
    # Create application
    app = engine.create_application(sr_eg, x)
    
    # Check type
    app_type = engine.check_type(app)
    print(f"Application type: {app_type}")
    
    # Map to fractal
    fractal_pos = engine.map_to_fractal((0.8, 0.7, 0.9))
    print(f"Fractal position: {fractal_pos}")
    
    # Determine modal status
    modal = engine.trinity_to_modal((0.8, 0.7, 0.9))
    print(f"Modal status: {modal}")