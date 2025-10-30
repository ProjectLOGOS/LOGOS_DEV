"""
Lambda Processors Module - UIP Step 2 IEL Ontological Synthesis Gateway

Lambda calculus normalization and structure processing.
Simplified to single core processor per refactor requirements.
"""

from .lambda_core import normalize_structure

__all__ = ['normalize_structure']