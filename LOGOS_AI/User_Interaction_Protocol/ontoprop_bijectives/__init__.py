"""
IEL_ONTO_KIT - UIP Step 2 IEL Ontological Synthesis Gateway

This package provides the complete IEL ↔ Ontological Property bijective processing
framework for UIP Step 2, bridging PXL compliance validation with IEL overlay analysis.

Core Workflow:
1. IEL bijective activation → iel_registry.load_active_domains()
2. Recombine phase → recombine_core.merge_outputs()
3. Entropy check → entropy_metrics.assess_distribution() [optional]
4. λ-processing → lambda_core.normalize_structure()
5. Translation → translation_engine.convert_to_nl()
6. Tokenization → obdc_kernel.emit_tokens()
7. Validation hook → validation.test_pipeline.run()

Version: IEL_ONTO_KIT_v2
Integration: UIP Step 2 — IEL Ontological Synthesis Gateway
"""

import logging

# Configure standardized logging
logger = logging.getLogger("IEL_ONTO_KIT")

# Package version
__version__ = "2.0.0"

from .lambda_processors import lambda_core
from .math_cores import vector_norms
from .obdc import obdc_kernel

# Core modules
from .recombine import recombine_core
from .registry import iel_registry
from .translators import translation_engine

__all__ = [
    "recombine_core",
    "translation_engine",
    "vector_norms",
    "lambda_core",
    "obdc_kernel",
    "iel_registry",
]


class IELKitError(Exception):
    """Base exception for IEL_ONTO_KIT operations"""

    pass


def execute_step2_synthesis(input_data: dict) -> dict:
    """
    Execute complete Step 2 IEL synthesis workflow

    Args:
        input_data: Processed linguistic context from Step 1

    Returns:
        dict: Complete synthesis result ready for Step 3
    """
    try:
        # 1. Load active IEL domains
        domains = iel_registry.load_active_domains()

        # 2. Merge IEL outputs
        recombined = recombine_core.merge_outputs(input_data, domains)

        # 3. Lambda processing
        normalized = lambda_core.normalize_structure(recombined)

        # 4. Translation
        translated = translation_engine.convert_to_nl(normalized)

        # 5. Tokenization
        tokens = obdc_kernel.emit_tokens(translated)

        return {
            "status": "ok",
            "payload": tokens,
            "metadata": {
                "stage": "complete_synthesis",
                "domains_processed": len(domains),
                "workflow_version": __version__,
            },
        }

    except Exception as e:
        logger.error(f"Step 2 synthesis failed: {e}")
        raise IELKitError(f"Synthesis workflow error: {str(e)}")
