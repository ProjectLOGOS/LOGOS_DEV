"""
Lambda Calculus Processing Engine - UIP Step 1 Component
=======================================================

Advanced Lambda calculus evaluation engine with type inference, reduction strategies,
and functional programming constructs for symbolic reasoning and computation.

Adapted from: V2_Possible_Gap_Fillers/lambda_calculus_core.py
Enhanced with: Type system integration, Church encoding, combinatory logic, Trinity vector functions
"""

from protocols.shared.system_imports import *
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy
import inspect


class LambdaType(Enum):
    """Lambda expression types"""
    VARIABLE = "var"
    ABSTRACTION = "abs"  # λx.M
    APPLICATION = "app"  # M N
    CONSTANT = "const"
    CHURCH_NUMERAL = "church"
    COMBINATOR = "comb"


class ReductionStrategy(Enum):
    """Lambda calculus reduction strategies"""
    NORMAL_ORDER = "normal"      # Leftmost-outermost
    APPLICATIVE_ORDER = "applicative"  # Leftmost-innermost  
    CALL_BY_NAME = "call_by_name"
    CALL_BY_VALUE = "call_by_value"
    LAZY_EVALUATION = "lazy"


class TypeCategory(Enum):
    """Type system categories"""
    SIMPLE = "simple"            # Simple types (A, B, C...)
    FUNCTION = "function"        # A -> B
    PRODUCT = "product"          # A × B
    SUM = "sum"                 # A + B
    POLYMORPHIC = "poly"        # ∀α.τ
    TRINITY = "trinity"         # Trinity vector type


@dataclass
class LambdaType_:
    """Type representation in lambda calculus"""
    category: TypeCategory
    components: List['LambdaType_'] = field(default_factory=list)
    name: Optional[str] = None
    
    def __str__(self) -> str:
        if self.category == TypeCategory.SIMPLE:
            return self.name or "τ"
        elif self.category == TypeCategory.FUNCTION:
            if len(self.components) >= 2:
                return f"({self.components[0]} → {self.components[1]})"
            return "(_ → _)"
        elif self.category == TypeCategory.PRODUCT:
            return " × ".join(str(c) for c in self.components)
        elif self.category == TypeCategory.SUM:
            return " + ".join(str(c) for c in self.components)
        elif self.category == TypeCategory.POLYMORPHIC:
            return f"∀{self.name}.{self.components[0]}" if self.components else f"∀{self.name}.τ"
        else:
            return self.name or str(self.category.value)


@dataclass 
class LambdaExpression:
    """Lambda calculus expression representation"""
    expr_type: LambdaType
    value: Any
    bound_var: Optional[str] = None  # For abstractions
    body: Optional['LambdaExpression'] = None  # For abstractions
    function: Optional['LambdaExpression'] = None  # For applications
    argument: Optional['LambdaExpression'] = None  # For applications
    type_annotation: Optional[LambdaType_] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """String representation of lambda expression"""
        if self.expr_type == LambdaType.VARIABLE:
            return str(self.value)
        elif self.expr_type == LambdaType.ABSTRACTION:
            return f"λ{self.bound_var}.{self.body}"
        elif self.expr_type == LambdaType.APPLICATION:
            return f"({self.function} {self.argument})"
        elif self.expr_type == LambdaType.CONSTANT:
            return str(self.value)
        elif self.expr_type == LambdaType.CHURCH_NUMERAL:
            return f"Church({self.value})"
        else:
            return str(self.value)
    
    def free_variables(self) -> Set[str]:
        """Get free variables in expression"""
        if self.expr_type == LambdaType.VARIABLE:
            return {self.value}
        elif self.expr_type == LambdaType.ABSTRACTION:
            body_free = self.body.free_variables() if self.body else set()
            return body_free - {self.bound_var}
        elif self.expr_type == LambdaType.APPLICATION:
            func_free = self.function.free_variables() if self.function else set()
            arg_free = self.argument.free_variables() if self.argument else set()
            return func_free | arg_free
        else:
            return set()
    
    def substitute(self, var: str, replacement: 'LambdaExpression') -> 'LambdaExpression':
        """Substitute variable with expression (with α-conversion for capture avoidance)"""
        if self.expr_type == LambdaType.VARIABLE:
            if self.value == var:
                return copy.deepcopy(replacement)
            else:
                return copy.deepcopy(self)
        
        elif self.expr_type == LambdaType.ABSTRACTION:
            if self.bound_var == var:
                # Variable is bound, no substitution
                return copy.deepcopy(self)
            elif self.bound_var in replacement.free_variables():
                # Need α-conversion to avoid capture
                fresh_var = self._generate_fresh_var(replacement.free_variables())
                renamed_body = self.body.substitute(self.bound_var, LambdaExpression(LambdaType.VARIABLE, fresh_var))
                substituted_body = renamed_body.substitute(var, replacement)
                return LambdaExpression(
                    LambdaType.ABSTRACTION,
                    None,
                    fresh_var,
                    substituted_body
                )
            else:
                # Safe to substitute in body
                new_body = self.body.substitute(var, replacement)
                return LambdaExpression(
                    LambdaType.ABSTRACTION,
                    None,
                    self.bound_var,
                    new_body
                )
        
        elif self.expr_type == LambdaType.APPLICATION:
            new_function = self.function.substitute(var, replacement)
            new_argument = self.argument.substitute(var, replacement)
            return LambdaExpression(
                LambdaType.APPLICATION,
                None,
                function=new_function,
                argument=new_argument
            )
        
        else:
            # Constants don't contain variables
            return copy.deepcopy(self)
    
    def _generate_fresh_var(self, avoid_vars: Set[str]) -> str:
        """Generate fresh variable name"""
        base = f"{self.bound_var}'"
        candidate = base
        counter = 1
        
        while candidate in avoid_vars:
            candidate = f"{base}{counter}"
            counter += 1
        
        return candidate


@dataclass
class ReductionResult:
    """Result of lambda expression reduction"""
    expression: LambdaExpression
    steps: List[str]
    is_normal_form: bool
    reduction_count: int
    type_info: Optional[LambdaType_]


class ChurchEncoding:
    """Church encoding utilities for natural numbers and data structures"""
    
    @staticmethod
    def encode_number(n: int) -> LambdaExpression:
        """Encode natural number as Church numeral"""
        # λf.λx.f^n(x) where f^n means f applied n times
        
        # Build nested applications: f(f(...f(x)...))
        def build_applications(count: int) -> LambdaExpression:
            if count == 0:
                return LambdaExpression(LambdaType.VARIABLE, "x")
            else:
                inner = build_applications(count - 1)
                return LambdaExpression(
                    LambdaType.APPLICATION,
                    None,
                    function=LambdaExpression(LambdaType.VARIABLE, "f"),
                    argument=inner
                )
        
        body_x = build_applications(n)
        body_f = LambdaExpression(LambdaType.ABSTRACTION, None, "x", body_x)
        church_num = LambdaExpression(LambdaType.ABSTRACTION, None, "f", body_f)
        church_num.expr_type = LambdaType.CHURCH_NUMERAL
        church_num.value = n
        
        return church_num
    
    @staticmethod
    def encode_boolean(value: bool) -> LambdaExpression:
        """Encode boolean as Church boolean"""
        if value:
            # TRUE = λx.λy.x
            inner = LambdaExpression(LambdaType.VARIABLE, "x")
            outer = LambdaExpression(LambdaType.ABSTRACTION, None, "y", inner)
            return LambdaExpression(LambdaType.ABSTRACTION, None, "x", outer)
        else:
            # FALSE = λx.λy.y
            inner = LambdaExpression(LambdaType.VARIABLE, "y")
            outer = LambdaExpression(LambdaType.ABSTRACTION, None, "y", inner)
            return LambdaExpression(LambdaType.ABSTRACTION, None, "x", outer)
    
    @staticmethod
    def successor() -> LambdaExpression:
        """Church numeral successor function"""
        # SUCC = λn.λf.λx.f(nfx)
        
        # Build nfx application
        nf_app = LambdaExpression(
            LambdaType.APPLICATION, None,
            function=LambdaExpression(LambdaType.VARIABLE, "n"),
            argument=LambdaExpression(LambdaType.VARIABLE, "f")
        )
        nfx_app = LambdaExpression(
            LambdaType.APPLICATION, None,
            function=nf_app,
            argument=LambdaExpression(LambdaType.VARIABLE, "x")
        )
        
        # f(nfx)
        result_app = LambdaExpression(
            LambdaType.APPLICATION, None,
            function=LambdaExpression(LambdaType.VARIABLE, "f"),
            argument=nfx_app
        )
        
        # Build lambda abstractions
        x_abs = LambdaExpression(LambdaType.ABSTRACTION, None, "x", result_app)
        f_abs = LambdaExpression(LambdaType.ABSTRACTION, None, "f", x_abs)
        n_abs = LambdaExpression(LambdaType.ABSTRACTION, None, "n", f_abs)
        
        return n_abs


class Combinators:
    """Standard combinatory logic combinators"""
    
    @staticmethod
    def I() -> LambdaExpression:
        """Identity combinator: I = λx.x"""
        return LambdaExpression(
            LambdaType.COMBINATOR,
            None,
            "x",
            LambdaExpression(LambdaType.VARIABLE, "x"),
            metadata={"name": "I", "description": "Identity"}
        )
    
    @staticmethod  
    def K() -> LambdaExpression:
        """Constant combinator: K = λx.λy.x"""
        inner = LambdaExpression(LambdaType.VARIABLE, "x")
        outer = LambdaExpression(LambdaType.ABSTRACTION, None, "y", inner)
        return LambdaExpression(
            LambdaType.COMBINATOR,
            None,
            "x",
            outer,
            metadata={"name": "K", "description": "Constant"}
        )
    
    @staticmethod
    def S() -> LambdaExpression:
        """Substitution combinator: S = λx.λy.λz.xz(yz)"""
        
        # Build xz
        xz_app = LambdaExpression(
            LambdaType.APPLICATION, None,
            function=LambdaExpression(LambdaType.VARIABLE, "x"),
            argument=LambdaExpression(LambdaType.VARIABLE, "z")
        )
        
        # Build yz  
        yz_app = LambdaExpression(
            LambdaType.APPLICATION, None,
            function=LambdaExpression(LambdaType.VARIABLE, "y"),
            argument=LambdaExpression(LambdaType.VARIABLE, "z")
        )
        
        # Build xz(yz)
        result_app = LambdaExpression(
            LambdaType.APPLICATION, None,
            function=xz_app,
            argument=yz_app
        )
        
        # Build abstractions
        z_abs = LambdaExpression(LambdaType.ABSTRACTION, None, "z", result_app)
        y_abs = LambdaExpression(LambdaType.ABSTRACTION, None, "y", z_abs)
        x_abs = LambdaExpression(LambdaType.ABSTRACTION, None, "x", y_abs)
        x_abs.expr_type = LambdaType.COMBINATOR
        x_abs.metadata = {"name": "S", "description": "Substitution"}
        
        return x_abs
    
    @staticmethod
    def Y() -> LambdaExpression:
        """Y combinator (fixed point): Y = λf.(λx.f(xx))(λx.f(xx))"""
        
        # Build f(xx) 
        xx_app = LambdaExpression(
            LambdaType.APPLICATION, None,
            function=LambdaExpression(LambdaType.VARIABLE, "x"),
            argument=LambdaExpression(LambdaType.VARIABLE, "x")
        )
        fxx_app = LambdaExpression(
            LambdaType.APPLICATION, None,
            function=LambdaExpression(LambdaType.VARIABLE, "f"),
            argument=xx_app
        )
        
        # Build λx.f(xx)
        x_abs = LambdaExpression(LambdaType.ABSTRACTION, None, "x", fxx_app)
        
        # Build (λx.f(xx))(λx.f(xx))
        self_app = LambdaExpression(
            LambdaType.APPLICATION, None,
            function=copy.deepcopy(x_abs),
            argument=copy.deepcopy(x_abs)
        )
        
        # Build λf.(λx.f(xx))(λx.f(xx))
        f_abs = LambdaExpression(LambdaType.ABSTRACTION, None, "f", self_app)
        f_abs.expr_type = LambdaType.COMBINATOR
        f_abs.metadata = {"name": "Y", "description": "Fixed point combinator"}
        
        return f_abs


class TypeInferencer:
    """Type inference engine for lambda expressions"""
    
    def __init__(self):
        self.type_var_counter = 0
        self.constraints: List[Tuple[LambdaType_, LambdaType_]] = []
        
    def fresh_type_var(self) -> LambdaType_:
        """Generate fresh type variable"""
        name = f"α{self.type_var_counter}"
        self.type_var_counter += 1
        return LambdaType_(TypeCategory.SIMPLE, [], name)
    
    def infer_type(
        self, 
        expr: LambdaExpression, 
        context: Optional[Dict[str, LambdaType_]] = None
    ) -> LambdaType_:
        """Infer type of lambda expression"""
        
        if context is None:
            context = {}
        
        if expr.expr_type == LambdaType.VARIABLE:
            if expr.value in context:
                return context[expr.value]
            else:
                # Free variable - assign fresh type
                return self.fresh_type_var()
        
        elif expr.expr_type == LambdaType.CONSTANT:
            # Determine type from value
            if isinstance(expr.value, int):
                return LambdaType_(TypeCategory.SIMPLE, [], "Int")
            elif isinstance(expr.value, bool):
                return LambdaType_(TypeCategory.SIMPLE, [], "Bool")
            elif isinstance(expr.value, str):
                return LambdaType_(TypeCategory.SIMPLE, [], "String")
            else:
                return self.fresh_type_var()
        
        elif expr.expr_type == LambdaType.ABSTRACTION:
            # λx.M : τ₁ → τ₂ where x : τ₁ and M : τ₂
            var_type = self.fresh_type_var()
            new_context = context.copy()
            new_context[expr.bound_var] = var_type
            
            body_type = self.infer_type(expr.body, new_context)
            
            return LambdaType_(
                TypeCategory.FUNCTION,
                [var_type, body_type]
            )
        
        elif expr.expr_type == LambdaType.APPLICATION:
            # M N where M : τ₁ → τ₂ and N : τ₁, result is τ₂
            func_type = self.infer_type(expr.function, context)
            arg_type = self.infer_type(expr.argument, context)
            
            result_type = self.fresh_type_var()
            expected_func_type = LambdaType_(
                TypeCategory.FUNCTION,
                [arg_type, result_type]
            )
            
            # Add constraint
            self.constraints.append((func_type, expected_func_type))
            
            return result_type
        
        elif expr.expr_type == LambdaType.CHURCH_NUMERAL:
            # Church numerals have type (α → α) → α → α
            alpha = self.fresh_type_var()
            func_type = LambdaType_(TypeCategory.FUNCTION, [alpha, alpha])
            church_type = LambdaType_(TypeCategory.FUNCTION, [func_type, alpha])
            return LambdaType_(TypeCategory.FUNCTION, [church_type, alpha])
        
        else:
            return self.fresh_type_var()


class LambdaCalculusEngine:
    """Main lambda calculus processing engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_reduction_steps = self.config.get('max_reduction_steps', 1000)
        self.default_strategy = ReductionStrategy(
            self.config.get('default_strategy', 'normal')
        )
        
        # Type inferencer
        self.type_inferencer = TypeInferencer()
        
        # Built-in combinators
        self.combinators = {
            'I': Combinators.I(),
            'K': Combinators.K(), 
            'S': Combinators.S(),
            'Y': Combinators.Y()
        }
        
        # Church encoding utilities
        self.church = ChurchEncoding()
        
        self.logger.info("Lambda calculus engine initialized")
    
    def parse_expression(self, expr_str: str) -> LambdaExpression:
        """Parse string representation into LambdaExpression"""
        # Simplified parser - in production, use proper parsing
        expr_str = expr_str.strip()
        
        # Handle Church numerals
        if expr_str.startswith('Church(') and expr_str.endswith(')'):
            num_str = expr_str[7:-1]
            try:
                num = int(num_str)
                return self.church.encode_number(num)
            except ValueError:
                pass
        
        # Handle combinators
        if expr_str in self.combinators:
            return self.combinators[expr_str]
        
        # Handle lambda abstractions
        if expr_str.startswith('λ') or expr_str.startswith('\\'):
            # Simple parsing: λx.body
            dot_pos = expr_str.find('.')
            if dot_pos > 0:
                var_part = expr_str[1:dot_pos].strip()
                body_part = expr_str[dot_pos+1:].strip()
                
                body_expr = self.parse_expression(body_part)
                return LambdaExpression(
                    LambdaType.ABSTRACTION,
                    None,
                    var_part,
                    body_expr
                )
        
        # Handle applications - simplified
        if '(' in expr_str and ')' in expr_str:
            # Find balanced parentheses and split
            # This is a very simplified approach
            pass
        
        # Default to variable
        return LambdaExpression(LambdaType.VARIABLE, expr_str)
    
    def reduce_expression(
        self,
        expr: LambdaExpression,
        strategy: Optional[ReductionStrategy] = None
    ) -> ReductionResult:
        """
        Reduce lambda expression using specified strategy
        
        Args:
            expr: Lambda expression to reduce
            strategy: Reduction strategy (uses default if None)
            
        Returns:
            ReductionResult: Comprehensive reduction results
        """
        try:
            if strategy is None:
                strategy = self.default_strategy
            
            steps = [f"Initial: {expr}"]
            current_expr = copy.deepcopy(expr)
            reduction_count = 0
            
            # Perform reductions
            while reduction_count < self.max_reduction_steps:
                reduced_expr, step_description = self._single_reduction_step(
                    current_expr, strategy
                )
                
                if reduced_expr is None:
                    # No more reductions possible
                    break
                
                current_expr = reduced_expr
                reduction_count += 1
                steps.append(f"Step {reduction_count}: {step_description} → {current_expr}")
                
                # Check if we've reached normal form
                if self._is_normal_form(current_expr):
                    break
            
            # Infer type
            try:
                inferred_type = self.type_inferencer.infer_type(current_expr)
            except Exception as e:
                self.logger.warning(f"Type inference failed: {e}")
                inferred_type = None
            
            is_normal = self._is_normal_form(current_expr)
            
            result = ReductionResult(
                expression=current_expr,
                steps=steps,
                is_normal_form=is_normal,
                reduction_count=reduction_count,
                type_info=inferred_type
            )
            
            self.logger.debug(f"Reduction completed: {reduction_count} steps, normal form: {is_normal}")
            return result
            
        except Exception as e:
            self.logger.error(f"Lambda reduction failed: {e}")
            return ReductionResult(
                expression=expr,
                steps=[f"Error: {str(e)}"],
                is_normal_form=False,
                reduction_count=0,
                type_info=None
            )
    
    def _single_reduction_step(
        self,
        expr: LambdaExpression,
        strategy: ReductionStrategy
    ) -> Tuple[Optional[LambdaExpression], str]:
        """Perform single reduction step"""
        
        if strategy == ReductionStrategy.NORMAL_ORDER:
            return self._normal_order_reduction(expr)
        elif strategy == ReductionStrategy.APPLICATIVE_ORDER:
            return self._applicative_order_reduction(expr)
        else:
            return self._normal_order_reduction(expr)  # Default
    
    def _normal_order_reduction(self, expr: LambdaExpression) -> Tuple[Optional[LambdaExpression], str]:
        """Normal order (leftmost-outermost) reduction"""
        
        if expr.expr_type == LambdaType.APPLICATION:
            # Check if we can perform beta reduction
            if (expr.function.expr_type == LambdaType.ABSTRACTION or 
                expr.function.expr_type == LambdaType.COMBINATOR):
                
                # Perform beta reduction
                result = expr.function.body.substitute(
                    expr.function.bound_var, 
                    expr.argument
                )
                return result, f"β-reduction: {expr.function.bound_var} := {expr.argument}"
            
            # Try to reduce function first
            reduced_func, step = self._normal_order_reduction(expr.function)
            if reduced_func is not None:
                new_expr = LambdaExpression(
                    LambdaType.APPLICATION,
                    None,
                    function=reduced_func,
                    argument=expr.argument
                )
                return new_expr, f"Reduce function: {step}"
            
            # Try to reduce argument
            reduced_arg, step = self._normal_order_reduction(expr.argument)
            if reduced_arg is not None:
                new_expr = LambdaExpression(
                    LambdaType.APPLICATION,
                    None,
                    function=expr.function,
                    argument=reduced_arg
                )
                return new_expr, f"Reduce argument: {step}"
        
        elif expr.expr_type == LambdaType.ABSTRACTION:
            # Try to reduce body
            if expr.body:
                reduced_body, step = self._normal_order_reduction(expr.body)
                if reduced_body is not None:
                    new_expr = LambdaExpression(
                        LambdaType.ABSTRACTION,
                        None,
                        expr.bound_var,
                        reduced_body
                    )
                    return new_expr, f"Reduce body: {step}"
        
        # No reduction possible
        return None, "No reduction"
    
    def _applicative_order_reduction(self, expr: LambdaExpression) -> Tuple[Optional[LambdaExpression], str]:
        """Applicative order (leftmost-innermost) reduction"""
        
        if expr.expr_type == LambdaType.APPLICATION:
            # Reduce arguments first
            reduced_func, func_step = self._applicative_order_reduction(expr.function)
            if reduced_func is not None:
                new_expr = LambdaExpression(
                    LambdaType.APPLICATION,
                    None,
                    function=reduced_func,
                    argument=expr.argument
                )
                return new_expr, f"Reduce function: {func_step}"
            
            reduced_arg, arg_step = self._applicative_order_reduction(expr.argument)
            if reduced_arg is not None:
                new_expr = LambdaExpression(
                    LambdaType.APPLICATION,
                    None,
                    function=expr.function,
                    argument=reduced_arg
                )
                return new_expr, f"Reduce argument: {arg_step}"
            
            # Now try beta reduction
            if (expr.function.expr_type == LambdaType.ABSTRACTION or
                expr.function.expr_type == LambdaType.COMBINATOR):
                
                result = expr.function.body.substitute(
                    expr.function.bound_var,
                    expr.argument
                )
                return result, f"β-reduction: {expr.function.bound_var} := {expr.argument}"
        
        elif expr.expr_type == LambdaType.ABSTRACTION:
            # In applicative order, don't reduce under lambda
            pass
        
        return None, "No reduction"
    
    def _is_normal_form(self, expr: LambdaExpression) -> bool:
        """Check if expression is in normal form (no redexes)"""
        
        if expr.expr_type == LambdaType.APPLICATION:
            # Check for beta redex
            if (expr.function.expr_type == LambdaType.ABSTRACTION or
                expr.function.expr_type == LambdaType.COMBINATOR):
                return False  # Beta redex present
            
            # Check subexpressions
            return (self._is_normal_form(expr.function) and 
                   self._is_normal_form(expr.argument))
        
        elif expr.expr_type == LambdaType.ABSTRACTION:
            return self._is_normal_form(expr.body) if expr.body else True
        
        else:
            return True  # Variables and constants are in normal form
    
    def create_trinity_function(self, e: float, g: float, t: float) -> LambdaExpression:
        """Create lambda function that processes Trinity vectors"""
        
        # Trinity processor: λf.f e g t
        # This applies function f to the three Trinity dimensions
        
        # Build f e g t as nested applications
        fe_app = LambdaExpression(
            LambdaType.APPLICATION, None,
            function=LambdaExpression(LambdaType.VARIABLE, "f"),
            argument=LambdaExpression(LambdaType.CONSTANT, e)
        )
        
        feg_app = LambdaExpression(
            LambdaType.APPLICATION, None,
            function=fe_app,
            argument=LambdaExpression(LambdaType.CONSTANT, g)
        )
        
        fegt_app = LambdaExpression(
            LambdaType.APPLICATION, None,
            function=feg_app,
            argument=LambdaExpression(LambdaType.CONSTANT, t)
        )
        
        trinity_func = LambdaExpression(
            LambdaType.ABSTRACTION, None,
            "f", fegt_app,
            metadata={
                "trinity_vector": (e, g, t),
                "description": "Trinity vector processor"
            }
        )
        
        return trinity_func
    
    def evaluate_with_context(
        self,
        expr: LambdaExpression,
        context: Dict[str, Any]
    ) -> ReductionResult:
        """Evaluate expression with additional context (Trinity vectors, etc.)"""
        
        # Substitute context values
        expr_with_context = copy.deepcopy(expr)
        
        for var_name, value in context.items():
            if isinstance(value, tuple) and len(value) == 3:
                # Trinity vector
                trinity_func = self.create_trinity_function(*value)
                expr_with_context = expr_with_context.substitute(
                    var_name, trinity_func
                )
            elif isinstance(value, (int, float, bool)):
                # Constant value
                const_expr = LambdaExpression(LambdaType.CONSTANT, value)
                expr_with_context = expr_with_context.substitute(
                    var_name, const_expr
                )
        
        # Perform reduction
        return self.reduce_expression(expr_with_context)


# Global lambda calculus engine instance
lambda_calculus_engine = LambdaCalculusEngine()


__all__ = [
    'LambdaType',
    'ReductionStrategy',
    'TypeCategory',
    'LambdaType_',
    'LambdaExpression',
    'ReductionResult',
    'ChurchEncoding',
    'Combinators',
    'TypeInferencer',
    'LambdaCalculusEngine',
    'lambda_calculus_engine'
]