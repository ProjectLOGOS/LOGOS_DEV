"""Lambda Engine Implementation

Core Lambda calculus implementation for THŌNOC system with ontological typing.
Provides typed lambda abstractions, applications, and sufficient reason operators
with evaluation and type checking capabilities.

Dependencies: typing, enum, json
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Set
from enum import Enum
import json
import logging

# Import from utils (adjust path as needed)
from ..utils.data_structures import OntologicalType, FunctionType

logger = logging.getLogger(__name__)

class LogosExpr:
    """Base class for all lambda expressions."""
    
    def __str__(self) -> str:
        """String representation."""
        return self._to_string()
    
    def _to_string(self) -> str:
        """Internal string representation method."""
        return "LogosExpr"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        return {"type": "expr"}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogosExpr':
        """Create expression from dictionary representation."""
        expr_type = data.get("type", "")
        if expr_type == "var":
            return Variable.from_dict(data)
        elif expr_type == "lambda":
            return Abstraction.from_dict(data)
        elif expr_type == "app":
            return Application.from_dict(data)
        elif expr_type == "value":
            return Value.from_dict(data)
        elif expr_type == "sr":
            return SufficientReason.from_dict(data)
        return cls()

class Variable(LogosExpr):
    """Variable in lambda calculus."""
    
    def __init__(self, name: str, onto_type: OntologicalType):
        """Initialize variable.
        
        Args:
            name: Variable name
            onto_type: Ontological type
        """
        self.name = name
        self.onto_type = onto_type
    
    def _to_string(self) -> str:
        """Return string representation."""
        return f"{self.name}:{self.onto_type.value}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "var",
            "name": self.name,
            "onto_type": self.onto_type.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Variable':
        """Create from dictionary representation."""
        return cls(
            name=data.get("name", ""),
            onto_type=OntologicalType(data.get("onto_type", "Prop"))
        )

class Value(LogosExpr):
    """Concrete value in Lambda calculus."""
    
    def __init__(self, value: str, onto_type: OntologicalType):
        """Initialize value.
        
        Args:
            value: Value identifier
            onto_type: Ontological type
        """
        self.value = value
        self.onto_type = onto_type
    
    def _to_string(self) -> str:
        """Return string representation."""
        return f"{self.value}:{self.onto_type.value}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "value",
            "value": self.value,
            "onto_type": self.onto_type.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Value':
        """Create from dictionary representation."""
        return cls(
            value=data.get("value", ""),
            onto_type=OntologicalType(data.get("onto_type", "Prop"))
        )

class Abstraction(LogosExpr):
    """Lambda abstraction (λx.M)."""
    
    def __init__(self, var_name: str, var_type: OntologicalType, body: LogosExpr):
        """Initialize lambda abstraction.
        
        Args:
            var_name: Variable name
            var_type: Variable type
            body: Function body
        """
        self.var_name = var_name
        self.var_type = var_type
        self.body = body
    
    def _to_string(self) -> str:
        """Return string representation."""
        return f"λ{self.var_name}:{self.var_type.value}.{self.body}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "lambda",
            "var_name": self.var_name,
            "var_type": self.var_type.value,
            "body": self.body.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Abstraction':
        """Create from dictionary representation."""
        body_data = data.get("body", {})
        return cls(
            var_name=data.get("var_name", ""),
            var_type=OntologicalType(data.get("var_type", "Prop")),
            body=LogosExpr.from_dict(body_data)
        )

class Application(LogosExpr):
    """Function application (M N)."""
    
    def __init__(self, func: LogosExpr, arg: LogosExpr):
        """Initialize function application.
        
        Args:
            func: Function
            arg: Argument
        """
        self.func = func
        self.arg = arg
    
    def _to_string(self) -> str:
        """Return string representation."""
        return f"({self.func} {self.arg})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "app",
            "func": self.func.to_dict(),
            "arg": self.arg.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Application':
        """Create from dictionary representation."""
        func_data = data.get("func", {})
        arg_data = data.get("arg", {})
        return cls(
            func=LogosExpr.from_dict(func_data),
            arg=LogosExpr.from_dict(arg_data)
        )

class SufficientReason(LogosExpr):
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
    
    def _to_string(self) -> str:
        """Return string representation."""
        return f"SR[{self.source_type.value},{self.target_type.value}]={self.value}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "sr",
            "source_type": self.source_type.value,
            "target_type": self.target_type.value,
            "value": self.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SufficientReason':
        """Create from dictionary representation."""
        return cls(
            source_type=OntologicalType(data.get("source_type", "EXISTENCE")),
            target_type=OntologicalType(data.get("target_type", "GOODNESS")),
            value=data.get("value", 0)
        )

class TypeChecker:
    """Type checking system for Lambda expressions."""
    
    def __init__(self):
        """Initialize type checker with initial environment."""
        self.env = {}
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize environment with built-in types and constants."""
        # Add SR operator types
        self.env["SR_E_G"] = FunctionType(OntologicalType.EXISTENCE, OntologicalType.GOODNESS)
        self.env["SR_G_T"] = FunctionType(OntologicalType.GOODNESS, OntologicalType.TRUTH)
    
    def check_type(self, expr: LogosExpr) -> Optional[Union[OntologicalType, FunctionType]]:
        """Check type of lambda expression.
        
        Args:
            expr: Expression to check
            
        Returns:
            Expression type or None if type error
        """
        if isinstance(expr, Variable):
            return self.env.get(expr.name, expr.onto_type)
        
        elif isinstance(expr, Value):
            return expr.onto_type
        
        elif isinstance(expr, Abstraction):
            # Save old binding if any
            old_binding = self.env.get(expr.var_name)
            
            # Add variable to environment
            self.env[expr.var_name] = expr.var_type
            
            # Check body type
            body_type = self.check_type(expr.body)
            
            # Restore old binding or remove current binding
            if old_binding:
                self.env[expr.var_name] = old_binding
            else:
                del self.env[expr.var_name]
            
            if body_type:
                return FunctionType(expr.var_type, body_type)
            return None
        
        elif isinstance(expr, Application):
            # Check function type
            func_type = self.check_type(expr.func)
            
            if not isinstance(func_type, FunctionType):
                logger.warning(f"Type error: Expected function, got {func_type}")
                return None
            
            # Check argument type
            arg_type = self.check_type(expr.arg)
            
            # Function domain must match argument type
            if func_type.domain != arg_type:
                logger.warning(f"Type error: Expected {func_type.domain}, got {arg_type}")
                return None
            
            return func_type.codomain
        
        elif isinstance(expr, SufficientReason):
            # Check valid SR combinations
            if (expr.source_type == OntologicalType.EXISTENCE and 
                expr.target_type == OntologicalType.GOODNESS and
                expr.value == 3):
                return FunctionType(expr.source_type, expr.target_type)
            
            elif (expr.source_type == OntologicalType.GOODNESS and 
                 expr.target_type == OntologicalType.TRUTH and
                 expr.value == 2):
                return FunctionType(expr.source_type, expr.target_type)
            
            logger.warning(f"Type error: Invalid SR operator: {expr}")
            return None
        
        logger.warning(f"Type error: Unknown expression type: {type(expr)}")
        return None

class Evaluator:
    """Evaluator for lambda expressions."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.env = {}
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize environment with built-in values and constants."""
        # Add fundamental values
        self.env["ei"] = Value("ei", OntologicalType.EXISTENCE)
        self.env["og"] = Value("og", OntologicalType.GOODNESS)
        self.env["at"] = Value("at", OntologicalType.TRUTH)
    
    def evaluate(self, expr: LogosExpr, max_depth: int = 100) -> LogosExpr:
        """Evaluate lambda expression.
        
        Args:
            expr: Expression to evaluate
            max_depth: Maximum evaluation depth
            
        Returns:
            Evaluated expression
        """
        return self._evaluate(expr, 0, max_depth)
    
    def _evaluate(self, expr: LogosExpr, depth: int, max_depth: int) -> LogosExpr:
        """Internal recursive evaluation.
        
        Args:
            expr: Expression to evaluate
            depth: Current evaluation depth
            max_depth: Maximum evaluation depth
            
        Returns:
            Evaluated expression
        """
        if depth >= max_depth:
            logger.warning(f"Evaluation depth limit reached: {max_depth}")
            return expr
        
        if isinstance(expr, Variable):
            # Look up variable in environment
            return self.env.get(expr.name, expr)
        
        elif isinstance(expr, Value) or isinstance(expr, SufficientReason):
            # Values and SR operators evaluate to themselves
            return expr
        
        elif isinstance(expr, Abstraction):
            # Evaluate body within abstraction
            new_body = self._evaluate(expr.body, depth + 1, max_depth)
            return Abstraction(expr.var_name, expr.var_type, new_body)
        
        elif isinstance(expr, Application):
            # Evaluate function and argument
            func = self._evaluate(expr.func, depth + 1, max_depth)
            arg = self._evaluate(expr.arg, depth + 1, max_depth)
            
            # Apply beta reduction if possible
            if isinstance(func, Abstraction):
                # (λx.M) N -> M[x:=N]
                substituted = self.substitute(func.body, func.var_name, arg)
                return self._evaluate(substituted, depth + 1, max_depth)
            
            # If no reduction applies, return application
            return Application(func, arg)
        
        return expr
    
    def substitute(self, expr: LogosExpr, var_name: str, value: LogosExpr) -> LogosExpr:
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
        
        elif isinstance(expr, Value) or isinstance(expr, SufficientReason):
            # Values and SR operators don't contain variables
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
        
        return expr

class LambdaEngine:
    """Core Lambda engine for THŌNOC system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Lambda engine.
        
        Args:
            config: Engine configuration
        """
        self.config = config or {}
        self.type_checker = TypeChecker()
        self.evaluator = Evaluator()
        self.expr_cache = {}
        
        logger.info("Lambda Engine initialized")
    
    def create_variable(self, name: str, type_str: str) -> Variable:
        """Create variable with specified type.
        
        Args:
            name: Variable name
            type_str: Type string
            
        Returns:
            Variable expression
        """
        return Variable(name, OntologicalType(type_str))
    
    def create_value(self, value: str, type_str: str) -> Value:
        """Create value with specified type.
        
        Args:
            value: Value identifier
            type_str: Type string
            
        Returns:
            Value expression
        """
        return Value(value, OntologicalType(type_str))
    
    def create_abstraction(self, var_name: str, var_type: str, body: LogosExpr) -> Abstraction:
        """Create lambda abstraction.
        
        Args:
            var_name: Variable name
            var_type: Variable type string
            body: Expression body
            
        Returns:
            Abstraction expression
        """
        return Abstraction(var_name, OntologicalType(var_type), body)
    
    def create_application(self, func: LogosExpr, arg: LogosExpr) -> Application:
        """Create function application.
        
        Args:
            func: Function expression
            arg: Argument expression
            
        Returns:
            Application expression
        """
        return Application(func, arg)
    
    def create_sr(self, source_type: str, target_type: str, value: int) -> SufficientReason:
        """Create sufficient reason operator.
        
        Args:
            source_type: Source type string
            target_type: Target type string
            value: SR value
            
        Returns:
            Sufficient reason operator
        """
        return SufficientReason(OntologicalType(source_type), OntologicalType(target_type), value)
    
    def check_type(self, expr: LogosExpr) -> Optional[Union[OntologicalType, FunctionType]]:
        """Check type of expression.
        
        Args:
            expr: Expression to check
            
        Returns:
            Expression type or None if type error
        """
        return self.type_checker.check_type(expr)
    
    def evaluate(self, expr: LogosExpr) -> LogosExpr:
        """Evaluate expression.
        
        Args:
            expr: Expression to evaluate
            
        Returns:
            Evaluated expression
        """
        max_depth = self.config.get("max_evaluation_depth", 100)
        return self.evaluator.evaluate(expr, max_depth)
    
    def parse_from_dict(self, data: Dict[str, Any]) -> LogosExpr:
        """Parse expression from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Parsed expression
        """
        return LogosExpr.from_dict(data)
    
    def expr_to_dict(self, expr: LogosExpr) -> Dict[str, Any]:
        """Convert expression to dictionary representation.
        
        Args:
            expr: Expression to convert
            
        Returns:
            Dictionary representation
        """
        return expr.to_dict()
    
    def apply_sr_eg(self, expr: LogosExpr) -> LogosExpr:
        """Apply Existence-to-Goodness SR operator to expression.
        
        Args:
            expr: Expression to transform
            
        Returns:
            Transformed expression
        """
        sr_eg = self.create_sr("EXISTENCE", "GOODNESS", 3)
        return self.create_application(sr_eg, expr)
    
    def apply_sr_gt(self, expr: LogosExpr) -> LogosExpr:
        """Apply Goodness-to-Truth SR operator to expression.
        
        Args:
            expr: Expression to transform
            
        Returns:
            Transformed expression
        """
        sr_gt = self.create_sr("GOODNESS", "TRUTH", 2)
        return self.create_application(sr_gt, expr)
    
    def cache_expression(self, expr_id: str, expr: LogosExpr) -> None:
        """Cache expression for reuse.
        
        Args:
            expr_id: Expression identifier
            expr: Expression to cache
        """
        self.expr_cache[expr_id] = expr
    
    def get_cached_expression(self, expr_id: str) -> Optional[LogosExpr]:
        """Get cached expression.
        
        Args:
            expr_id: Expression identifier
            
        Returns:
            Cached expression or None if not found
        """
        return self.expr_cache.get(expr_id)