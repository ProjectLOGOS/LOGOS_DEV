"""
OBDC Module - UIP Step 2 IEL Ontological Synthesis Gateway

Output Buffer and Data Communication for tokenization and stream output.
Final dispatcher for processed IEL ontological synthesis results.
"""

from .obdc_kernel import emit_tokens

__all__ = ["emit_tokens"]
