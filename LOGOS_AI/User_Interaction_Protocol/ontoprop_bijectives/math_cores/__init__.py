"""
Math Cores Module - UIP Step 2 IEL Ontological Synthesis Gateway

Numeric stability utilities for vector operations and normalization.
Excludes complex number and Mandelbrot constants per refactor requirements.
"""

from .vector_norms import compute_stability_metrics, normalize_vector

__all__ = ["normalize_vector", "compute_stability_metrics"]
