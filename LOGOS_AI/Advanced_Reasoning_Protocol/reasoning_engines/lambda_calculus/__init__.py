"""
LOGOS Lambda Engine
Typed lambda calculus implementation with ontological reasoning and 3PDN integration
"""

# Core Lambda Calculus Expressions
from .lambda_engine import (
    LogosExpr,
    Variable,
    Abstraction,
    Application,
    SufficientReason,
    Value,
)

# Ontological Lambda Calculus Engine
from .lambda_onto_calculus_engine import (
    OntologicalType,
    ModalOperator,
    LogicalLaw,
    LambdaExpr as OntoLambdaExpr,
    Variable as OntoVariable,
    Abstraction as OntoAbstraction,
    Application as OntoApplication,
    SufficientReason as OntoSufficientReason,
    FunctionType,
    TypeContext,
    TypeChecker,
    Evaluator,
)

# Parsing Components
from .lambda_parser import (
    TokenType,
    Token,
    Lexer,
    Parser,
    parse_expr,
)

# LOGOS-Specific Components
from .logos_lambda_core import (
    LambdaMLCore,
)

# Integration Components
from .lambda_engine import (
    LambdaEngine,
)
from .lambda_onto_calculus_engine import (
    LambdaExpr,
)
from .logos_lambda_integration import (
    PDNBridge,
)

# Utility Functions
# from .lambda_calculus import (
#     LogosExpr as CalculusLogosExpr,
# )

__all__ = [
    # Core Lambda Calculus
    "LogosExpr",
    "Variable",
    "Abstraction",
    "Application",
    "SufficientReason",
    "Value",

    # Ontological Lambda Calculus
    "OntologicalType",
    "ModalOperator",
    "LogicalLaw",
    "OntoLambdaExpr",
    "OntoVariable",
    "OntoAbstraction",
    "OntoApplication",
    "OntoSufficientReason",
    "FunctionType",
    "TypeContext",
    "TypeChecker",
    "Evaluator",

    # Parsing
    "TokenType",
    "Token",
    "Lexer",
    "Parser",
    "parse_expr",

    # LOGOS Core
    "LambdaMLCore",

    # Integration
    "LambdaEngine",
    "LambdaExpr",
    "PDNBridge",
    "LogosPDNBridge",

    # Utilities
    # "CalculusLogosExpr",
]