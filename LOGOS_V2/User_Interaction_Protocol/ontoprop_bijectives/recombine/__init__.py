"""
Recombine Module - UIP Step 2 IEL Ontological Synthesis Gateway

Handles IEL output merging and synthesis operations.
"""

from .recombine_core import merge_outputs
from .entropy_metrics import assess_distribution

__all__ = ['merge_outputs', 'assess_distribution']