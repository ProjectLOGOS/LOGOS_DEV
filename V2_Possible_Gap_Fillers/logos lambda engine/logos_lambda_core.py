"""Logos Lambda Core Engine

Implements the fundamental components of Lambda Logos calculus with ontological typing.
Provides typed lambda abstractions, applications, and sufficient reason operators
for the THŌNOC system.

Key components:
- Ontological types (𝔼, 𝔾, 𝕋)
- Lambda expressions
- Logical connectives
- Sufficient reason operators
- Type checking and evaluation

Dependencies: typing, enum
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Set
from enum import Enum
import json
import uuid

# --- Core Types ---

class OntologicalType(Enum):
    """Fundamental ontological dimensions."""
    EXISTENCE = "𝔼"
    GOODNESS = "𝔾"
    TRUTH = "𝕋"
    PROP = "Prop"  # Propositional type

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

# --- Expression Types ---

class LogosExpr:
    """Base class for Lambda Logos expressions."""
    def __str__(self) -> str:
        """Return string representation."""
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
        expr_type = data.get("type")
        if expr_type == "var":
            return Variable.from_dict(data)
        elif expr_type == "lambda":
            return Abstraction.from_dict(data)
        elif expr_type == "app":
            return Application.from_dict(data)
        elif expr_type == "const":
            return Constant.from_dict(data)
        elif expr_type == "value":
            return Value.from_dict(data)
        elif expr_type == "sr":
            return SufficientReason.from_dict(data)
        return cls()  # Default case

class Variable(LogosExpr):
    """Variable in lambda calculus."""
    
    def __init__(self, name: str, ont_type: OntologicalType):
        """Initialize variable.
        
        Args:
            name: Variable name
            ont_type: Ontological type
        """
        self.name = name
        self.ont_type = ont_type
    
    def _to_string(self) -> str:
        """Return string representation."""
        return f"{self.name}:{self.ont_type.value}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "var",
            "name": self.name,
            "ont_type": self.ont_type.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Variable':
        """Create from dictionary representation."""
        return cls(
            name=data.get("name", ""),
            ont_type=OntologicalType(data.get("ont_type", "Prop"))
        )

class Value(LogosExpr):
    """Concrete value in Lambda Logos."""
    
    def __init__(self, value: str, ont_type: OntologicalType):
        """Initialize value.
        
        Args:
            value: Value identifier
            ont_type: Ontological type
        """
        self.value = value
        self.ont_type = ont_type
    
    def _to_string(self) -> str:
        """Return string representation."""
        return f"{self.value}:{self.ont_type.value}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": "value",
            "value": self.value,
            "ont_type": self.ont_type.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Value':
        """Create from dictionary representation."""
        return cls(
            value=data.get("value", ""),
            ont_type=OntologicalType(data.get("ont_type", "Prop"))
        )

class FunctionType:
    """Function type constructor."""
    
    def __init__(self, domain: OntologicalType, codomain: Union[OntologicalType, 'FunctionType']):
        """Initialize function type.
        
        Args:
            domain: Input type
            codomain: Output type
        """
        self.domain = domain
        self.codomain = codomain
    
    def __str__(self) -> str:
        """Return string representation."""
        domain_str = self.domain.value
        codomain_str = str(self.codomain)
        return f"{domain_str} → {codomain_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        if isinstance(self.codomain, FunctionType):
            codomain_dict = self.codomain.to_dict()
        else:
            codomain_dict = self.codomain.value
        
        return [
            self.domain.value,
            "→",
            codomain_dict
        ]
    
    @classmethod
    def from_dict(cls, data: List[Any]) -> 'FunctionType':
        """Create from dictionary representation."""
        domain = OntologicalType(data[0])
        if isinstance(data[2], list):
            codomain = FunctionType.from_dict(data[2])
        else:
            codomain = OntologicalType(data[2])
        return cls(domain, codomain)

class Constant(LogosExpr):
    """Logical constant in Lambda Logos."""
    
    def __init__(self, name: str, const_type: Union[OntologicalType, FunctionType], value: Optional[LogosExpr] = None):
        """Initialize constant.
        
        Args:
            name: Constant name
            const_type: Constant type
            value: Optional definition
        """
        self.name = name
        self.const_type = const_type
        self.value = value
    
    def _to_string(self) -> str:
        """Return string representation."""
        type_str = str(self.const_type)
        if self.value:
            return f"{self.name}:{type_str} = {self.value}"
        return f"{self.name}:{type_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        if isinstance(self.const_type, FunctionType):
            type_dict = self.const_type.to_dict()
        else:
            type_dict = self.const_type.value
        
        result = {
            "type": "const",
            "name": self.name,
            "const_type": type_dict
        }
        
        if self.value:
            result["value"] = self.value.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Constant':
        """Create from dictionary representation."""
        const_type_data = data.get("const_type")
        if isinstance(const_type_data, list):
            const_type = FunctionType.from_dict(const_type_data)
        else:
            const_type = OntologicalType(const_type_data)
        
        value_data = data.get("value")
        value = LogosExpr.from_dict(value_data) if value_data else None
        
        return cls(
            name=data.get("name", ""),
            const_type=const_type,
            value=value
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
            "var": self.var_name,
            "varType": self.var_type.value,
            "body": self.body.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Abstraction':
        """Create from dictionary representation."""
        body_data = data.get("body", {})
        return cls(
            var_name=data.get("var", ""),
            var_type=OntologicalType(data.get("varType", "Prop")),
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
            source_type=OntologicalType(data.get("source_type", "𝔼")),
            target_type=OntologicalType(data.get("target_type", "𝔾")),
            value=data.get("value", 0)
        )

# --- Logical Connectives ---

def create_logical_connectives() -> Dict[str, Constant]:
    """Create logical connective constants.
    
    Returns:
        Dictionary of logical connectives
    """
    prop_type = OntologicalType.PROP
    
    # Create function types
    prop_to_prop = FunctionType(prop_type, prop_type)
    prop_to_prop_to_prop = FunctionType(prop_type, FunctionType(prop_type, prop_type))
    
    # Create connectives
    connectives = {}
    
    # NOT: Prop → Prop
    connectives["NOT"] = Constant("NOT", prop_to_prop)
    
    # AND, OR, IMPLIES: Prop → Prop → Prop
    connectives["AND"] = Constant("AND", prop_to_prop_to_prop)
    connectives["OR"] = Constant("OR", prop_to_prop_to_prop)
    connectives["IMPLIES"] = Constant("IMPLIES", prop_to_prop_to_prop)
    
    # EQ: Generic equality predicate
    # Note: In a full implementation, this would be polymorphic
    connectives["EQ"] = Constant("EQ", prop_to_prop_to_prop)
    
    return connectives

# --- Core Axioms ---

def create_core_axioms() -> Dict[str, Constant]:
    """Create core Lambda Logos axioms.
    
    Returns:
        Dictionary of axiom constants
    """
    prop_type = OntologicalType.PROP
    
    # Create ontological values
    ei_val = Value("ei", OntologicalType.EXISTENCE)
    og_val = Value("og", OntologicalType.GOODNESS)
    at_val = Value("at", OntologicalType.TRUTH)
    
    # Create logical law values
    id_val = Value("ID", prop_type)
    nc_val = Value("NC", prop_type)
    em_val = Value("EM", prop_type)
    
    # Create EQ operator
    eq_op = Constant("EQ", FunctionType(prop_type, FunctionType(prop_type, prop_type)))
    
    # Create axioms
    axioms = {}
    
    # Axiom AOI: EI ≡ ID
    aoi_expr = Application(
        Application(eq_op, ei_val),
        id_val
    )
    axioms["AxiomAOI"] = Constant("AxiomAOI", prop_type, aoi_expr)
    
    # Axiom AEC: OG ≡ NC
    aec_expr = Application(
        Application(eq_op, og_val),
        nc_val
    )
    axioms["AxiomAEC"] = Constant("AxiomAEC", prop_type, aec_expr)
    
    # Axiom AEF: AT ≡ EM
    aef_expr = Application(
        Application(eq_op, at_val),
        em_val
    )
    axioms["AxiomAEF"] = Constant("AxiomAEF", prop_type, aef_expr)
    
    return axioms

# --- Type Checker ---

class TypeChecker:
    """Type checking system for Lambda Logos expressions."""
    
    def __init__(self):
        """Initialize type checker with initial environment."""
        self.env = {}
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize environment with built-in types and constants."""
        # Add logical connectives
        for name, const in create_logical_connectives().items():
            self.env[name] = const.const_type
        
        # Add axioms
        for name, axiom in create_core_axioms().items():
            self.env[name] = axiom.const_type
        
        # Add sufficient reason operators
        self.env["SR_E_G"] = FunctionType(OntologicalType.EXISTENCE, OntologicalType.GOODNESS)
        self.env["SR_G_T"] = FunctionType(OntologicalType.GOODNESS, OntologicalType.TRUTH)
    
    def check_type(self, expr: LogosExpr) -> Optional[Union[OntologicalType, FunctionType]]:
        """Check type of Lambda Logos expression.
        
        Args:
            expr: Expression to check
            
        Returns:
            Expression type or None if type error
        """
        if isinstance(expr, Variable):
            return self.env.get(expr.name, expr.ont_type)
        
        elif isinstance(expr, Value):
            return expr.ont_type
        
        elif isinstance(expr, Constant):
            return expr.const_type
        
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
                print(f"Type error: Expected function, got {func_type}")
                return None
            
            # Check argument type
            arg_type = self.check_type(expr.arg)
            
            # Function domain must match argument type
            if isinstance(func_type.domain, OntologicalType) and func_type.domain != arg_type:
                print(f"Type error: Expected {func_type.domain}, got {arg_type}")
                return None
            
            return func_type.codomain
        
        elif isinstance(expr, SufficientReason):
            # Check valid SR combinations
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
    """Evaluator for Lambda Logos expressions."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.env = {}
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize environment with built-in values and constants."""
        # Add logical connectives
        for name, const in create_logical_connectives().items():
            self.env[name] = const
        
        # Add axioms
        for name, axiom in create_core_axioms().items():
            self.env[name] = axiom
    
    def evaluate(self, expr: LogosExpr) -> LogosExpr:
        """Evaluate Lambda Logos expression.
        
        Args:
            expr: Expression to evaluate
            
        Returns:
            Evaluated expression
        """
        if isinstance(expr, Variable):
            # Look up variable in environment
            return self.env.get(expr.name, expr)
        
        elif isinstance(expr, Value) or isinstance(expr, Constant) or isinstance(expr, SufficientReason):
            # Values, constants, and SR operators evaluate to themselves
            return expr
        
        elif isinstance(expr, Abstraction):
            # Evaluate body within abstraction
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
            
            # Handle connectives and SR operators
            if isinstance(func, Constant):
                if func.name == "NOT" and isinstance(arg, Value):
                    # Basic NOT implementation for values
                    if arg.value == "TrueProp":
                        return Value("FalseProp", OntologicalType.PROP)
                    elif arg.value == "FalseProp":
                        return Value("TrueProp", OntologicalType.PROP)
                
                # For other connectives, would handle similar logic
            
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
        
        elif isinstance(expr, Value) or isinstance(expr, Constant) or isinstance(expr, SufficientReason):
            # Values, constants, and SR operators don't contain variables
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

# --- Lambda Logos Core Engine ---

class LambdaLogosEngine:
    """Core Lambda Logos engine."""
    
    def __init__(self):
        """Initialize Lambda Logos engine."""
        self.type_checker = TypeChecker()
        self.evaluator = Evaluator()
        self.expr_cache = {}
    
    def create_variable(self, name: str, type_str: str) -> Variable:
        """Create variable with specified type.
        
        Args:
            name: Variable name
            type_str: Type string (𝔼, 𝔾, 𝕋, Prop)
            
        Returns:
            Variable expression
        """
        return Variable(name, OntologicalType(type_str))
    
    def create_value(self, value: str, type_str: str) -> Value:
        """Create value with specified type.
        
        Args:
            value: Value identifier
            type_str: Type string (𝔼, 𝔾, 𝕋, Prop)
            
        Returns:
            Value expression
        """
        return Value(value, OntologicalType(type_str))
    
    def create_abstraction(self, var_name: str, var_type: str, body: LogosExpr) -> Abstraction:
        """Create lambda abstraction.
        
        Args:
            var_name: Variable name
            var_type: Variable type string (𝔼, 𝔾, 𝕋, Prop)
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
            source_type: Source type string (𝔼, 𝔾, 𝕋)
            target_type: Target type string (𝔼, 𝔾, 𝕋)
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
        return self.evaluator.evaluate(expr)
    
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

# --- Predefined Constants ---

def get_predefined_values() -> Dict[str, Value]:
    """Get predefined ontological values.
    
    Returns:
        Dictionary of predefined values
    """
    return {
        "ei": Value("ei", OntologicalType.EXISTENCE),
        "og": Value("og", OntologicalType.GOODNESS),
        "at": Value("at", OntologicalType.TRUTH),
        "TrueProp": Value("TrueProp", OntologicalType.PROP),
        "FalseProp": Value("FalseProp", OntologicalType.PROP)
    }

# Example usage
if __name__ == "__main__":
    # Initialize Lambda Logos engine
    engine = LambdaLogosEngine()
    
    # Create some basic expressions
    x = engine.create_variable("x", "𝔼")
    ei = engine.create_value("ei", "𝔼")
    sr_eg = engine.create_sr("𝔼", "𝔾", 3)
    
    # Create application
    app = engine.create_application(sr_eg, ei)
    
    # Check type
    app_type = engine.check_type(app)
    print(f"Application type: {app_type}")
    
    # Evaluate
    result = engine.evaluate(app)
    print(f"Evaluation result: {result}")
    
    # Serialize to dictionary
    app_dict = engine.expr_to_dict(app)
    print(f"Dictionary representation: {json.dumps(app_dict, indent=2)}")
    
    # Parse from dictionary
    parsed = engine.parse_from_dict(app_dict)
    print(f"Parsed expression: {parsed}")
    """Enhanced Evaluator for Lambda Logos Core

Implements full reduction rules for logical connectives within the Lambda Logos
evaluation system. Provides consistent logical operations on propositions, enforcing
truth preservation across evaluation steps.

Extends the base Evaluator class with comprehensive connective handling.
"""

class EnhancedEvaluator(Evaluator):
    """Enhanced evaluator with complete logical connective support."""
    
    def __init__(self):
        """Initialize enhanced evaluator with truth values."""
        super().__init__()
        
        # Add truth values to environment
        self.true_prop = Value("TrueProp", OntologicalType.PROP)
        self.false_prop = Value("FalseProp", OntologicalType.PROP)
        self.env["TrueProp"] = self.true_prop
        self.env["FalseProp"] = self.false_prop
    
    def evaluate(self, expr: LogosExpr) -> LogosExpr:
        """Evaluate Lambda Logos expression with enhanced logical connective support.
        
        Args:
            expr: Expression to evaluate
            
        Returns:
            Evaluated expression
        """
        if isinstance(expr, Variable):
            # Look up variable in environment
            return self.env.get(expr.name, expr)
        
        elif isinstance(expr, Value) or isinstance(expr, Constant) or isinstance(expr, SufficientReason):
            # Values, constants, and SR operators evaluate to themselves
            return expr
        
        elif isinstance(expr, Abstraction):
            # Evaluate body within abstraction
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
            
            # Handle logical connectives
            if isinstance(func, Constant):
                # Handle NOT
                if func.name == "NOT":
                    return self._evaluate_not(arg)
                
                # Handle AND (first application)
                elif func.name == "AND":
                    # Return function waiting for second argument
                    return self._make_and_partial(arg)
                
                # Handle OR (first application)
                elif func.name == "OR":
                    # Return function waiting for second argument
                    return self._make_or_partial(arg)
                
                # Handle IMPLIES (first application)
                elif func.name == "IMPLIES":
                    # Return function waiting for second argument
                    return self._make_implies_partial(arg)
                
                # Handle EQ (first application)
                elif func.name == "EQ":
                    # Return function waiting for second argument
                    return self._make_eq_partial(arg)
            
            # Handle partial applications of logical connectives
            if isinstance(func, _PartialApplication):
                return func.apply(arg, self)
            
            # If no reduction applies, return application
            return Application(func, arg)
        
        return expr
    
    def _evaluate_not(self, arg: LogosExpr) -> LogosExpr:
        """Evaluate NOT operation.
        
        Args:
            arg: Argument expression
            
        Returns:
            Evaluated expression
        """
        if arg == self.true_prop:
            return self.false_prop
        elif arg == self.false_prop:
            return self.true_prop
        
        # If arg is not a truth value, return NOT application
        return Application(self.env["NOT"], arg)
    
    def _make_and_partial(self, arg1: LogosExpr) -> '_PartialApplication':
        """Create partial AND application.
        
        Args:
            arg1: First argument
            
        Returns:
            Partial application
        """
        return _PartialApplication("AND", arg1, self._evaluate_and)
    
    def _evaluate_and(self, arg1: LogosExpr, arg2: LogosExpr) -> LogosExpr:
        """Evaluate AND operation.
        
        Args:
            arg1: First argument
            arg2: Second argument
            
        Returns:
            Evaluated expression
        """
        # Truth table for AND
        if arg1 == self.false_prop or arg2 == self.false_prop:
            return self.false_prop
        elif arg1 == self.true_prop and arg2 == self.true_prop:
            return self.true_prop
        
        # If not both truth values, return AND application
        return Application(Application(self.env["AND"], arg1), arg2)
    
    def _make_or_partial(self, arg1: LogosExpr) -> '_PartialApplication':
        """Create partial OR application.
        
        Args:
            arg1: First argument
            
        Returns:
            Partial application
        """
        return _PartialApplication("OR", arg1, self._evaluate_or)
    
    def _evaluate_or(self, arg1: LogosExpr, arg2: LogosExpr) -> LogosExpr:
        """Evaluate OR operation.
        
        Args:
            arg1: First argument
            arg2: Second argument
            
        Returns:
            Evaluated expression
        """
        # Truth table for OR
        if arg1 == self.true_prop or arg2 == self.true_prop:
            return self.true_prop
        elif arg1 == self.false_prop and arg2 == self.false_prop:
            return self.false_prop
        
        # If not both truth values, return OR application
        return Application(Application(self.env["OR"], arg1), arg2)
    
    def _make_implies_partial(self, arg1: LogosExpr) -> '_PartialApplication':
        """Create partial IMPLIES application.
        
        Args:
            arg1: First argument
            
        Returns:
            Partial application
        """
        return _PartialApplication("IMPLIES", arg1, self._evaluate_implies)
    
    def _evaluate_implies(self, arg1: LogosExpr, arg2: LogosExpr) -> LogosExpr:
        """Evaluate IMPLIES operation.
        
        Args:
            arg1: First argument (antecedent)
            arg2: Second argument (consequent)
            
        Returns:
            Evaluated expression
        """
        # Truth table for IMPLIES
        if arg1 == self.false_prop or arg2 == self.true_prop:
            return self.true_prop
        elif arg1 == self.true_prop and arg2 == self.false_prop:
            return self.false_prop
        
        # If not both truth values, return IMPLIES application
        return Application(Application(self.env["IMPLIES"], arg1), arg2)
    
    def _make_eq_partial(self, arg1: LogosExpr) -> '_PartialApplication':
        """Create partial EQ application.
        
        Args:
            arg1: First argument
            
        Returns:
            Partial application
        """
        return _PartialApplication("EQ", arg1, self._evaluate_eq)
    
    def _evaluate_eq(self, arg1: LogosExpr, arg2: LogosExpr) -> LogosExpr:
        """Evaluate EQ (logical equivalence) operation.
        
        Args:
            arg1: First argument
            arg2: Second argument
            
        Returns:
            Evaluated expression
        """
        # Truth table for logical equivalence
        if (arg1 == self.true_prop and arg2 == self.true_prop) or \
           (arg1 == self.false_prop and arg2 == self.false_prop):
            return self.true_prop
        elif (arg1 == self.true_prop and arg2 == self.false_prop) or \
             (arg1 == self.false_prop and arg2 == self.true_prop):
            return self.false_prop
        
        # If not both truth values, return EQ application
        return Application(Application(self.env["EQ"], arg1), arg2)


class _PartialApplication:
    """Helper class for partial applications of logical connectives."""
    
    def __init__(self, op_name: str, arg1: LogosExpr, evaluator_func: Callable):
        """Initialize partial application.
        
        Args:
            op_name: Operator name
            arg1: First argument
            evaluator_func: Function to evaluate full application
        """
        self.op_name = op_name
        self.arg1 = arg1
        self.evaluator_func = evaluator_func
    
    def apply(self, arg2: LogosExpr, evaluator: EnhancedEvaluator) -> LogosExpr:
        """Apply second argument to complete the application.
        
        Args:
            arg2: Second argument
            evaluator: Evaluator instance
            
        Returns:
            Evaluated result
        """
        return self.evaluator_func(self.arg1, arg2)
    
    def __str__(self) -> str:
        """Return string representation."""
        return f"({self.op_name} {self.arg1} ?)"


def create_core_axioms_enhanced() -> Dict[str, Constant]:
    """Create enhanced core Lambda Logos axioms using logical connectives.
    
    Returns:
        Dictionary of axiom constants
    """
    prop_type = OntologicalType.PROP
    
    # Create ontological values
    ei_val = Value("ei", OntologicalType.EXISTENCE)
    og_val = Value("og", OntologicalType.GOODNESS)
    at_val = Value("at", OntologicalType.TRUTH)
    
    # Create logical law values
    id_val = Value("ID", prop_type)
    nc_val = Value("NC", prop_type)
    em_val = Value("EM", prop_type)
    
    # Get logical connectives
    eq_op = Constant("EQ", FunctionType(prop_type, FunctionType(prop_type, prop_type)))
    and_op = Constant("AND", FunctionType(prop_type, FunctionType(prop_type, prop_type)))
    or_op = Constant("OR", FunctionType(prop_type, FunctionType(prop_type, prop_type)))
    not_op = Constant("NOT", FunctionType(prop_type, prop_type))
    implies_op = Constant("IMPLIES", FunctionType(prop_type, FunctionType(prop_type, prop_type)))
    
    # Create axioms
    axioms = {}
    
    # Axiom AOI: EI ≡ ID
    aoi_expr = Application(
        Application(eq_op, ei_val),
        id_val
    )
    axioms["AxiomAOI"] = Constant("AxiomAOI", prop_type, aoi_expr)
    
    # Axiom AEC: OG ≡ NC
    aec_expr = Application(
        Application(eq_op, og_val),
        nc_val
    )
    axioms["AxiomAEC"] = Constant("AxiomAEC", prop_type, aec_expr)
    
    # Axiom AEF: AT ≡ EM
    aef_expr = Application(
        Application(eq_op, at_val),
        em_val
    )
    axioms["AxiomAEF"] = Constant("AxiomAEF", prop_type, aef_expr)
    
    # Additional axioms demonstrating logical connectives
    
    # Law of Non-Contradiction: ¬(p ∧ ¬p)
    nc_demo = Application(
        not_op,
        Application(
            Application(and_op, Variable("p", prop_type)),
            Application(not_op, Variable("p", prop_type))
        )
    )
    axioms["NonContradiction"] = Constant("NonContradiction", prop_type, nc_demo)
    
    # Law of Excluded Middle: p ∨ ¬p
    em_demo = Application(
        Application(or_op, Variable("p", prop_type)),
        Application(not_op, Variable("p", prop_type))
    )
    axioms["ExcludedMiddle"] = Constant("ExcludedMiddle", prop_type, em_demo)
    
    # Transitivity: (p → q) ∧ (q → r) → (p → r)
    trans_left = Application(
        Application(and_op,
            Application(
                Application(implies_op, Variable("p", prop_type)),
                Variable("q", prop_type)
            )
        ),
        Application(
            Application(implies_op, Variable("q", prop_type)),
            Variable("r", prop_type)
        )
    )
    trans_right = Application(
        Application(implies_op, Variable("p", prop_type)),
        Variable("r", prop_type)
    )
    trans_demo = Application(
        Application(implies_op, trans_left),
        trans_right
    )
    axioms["Transitivity"] = Constant("Transitivity", prop_type, trans_demo)
    
    return axioms