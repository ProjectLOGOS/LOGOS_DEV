"""
Validation Module - UIP Step 2 IEL Ontological Synthesis Gateway

Pipeline validation and testing for recombine → λ → translate → OBDC workflow.
Renamed from tests/ per refactor requirements.
"""

from test_pipeline import run_validation, validate_workflow_integration

__all__ = ['run_validation', 'validate_workflow_integration']