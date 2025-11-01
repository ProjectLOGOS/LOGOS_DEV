"""
LOGOS Bayesian Reasoning Engine
===============================

Consolidated Bayesian inference system for the LOGOS AGI framework.
Provides unified access to all Bayesian reasoning capabilities.

This module consolidates:
- Core Bayesian inference (trinity vectors, probabilistic reasoning)
- Enhanced Bayesian components (MCMC, hierarchical networks, real-time updates)
- Bayesian interfaces and data structures
- Nexus integration for cross-protocol communication
"""

# Core Bayesian components
from .bayesian_enhanced.bayesian_inference import (
    TrinityVector,
    UnifiedBayesianInferencer,
    IELEpistemicState
)

from .bayesian_enhanced.bayesian_interface import (
    BayesianInterface,
    ProbabilisticResult,
    TrueP
)

# Enhanced Bayesian components
from .bayesian_enhanced.bayesian_inferencer import BayesianTrinityInferencer
from .bayesian_enhanced.hierarchical_bayes_network import (
    execute_HBN,
    run_HBN_analysis,
    query_intent_analyzer
)
from .bayesian_enhanced.bayes_update_real_time import (
    resolve_priors_path,
    load_priors,
    save_priors,
    score_data_point,
    assign_confidence,
    filter_and_score,
    predictive_refinement,
    run_BERT_pipeline
)
from .bayesian_enhanced.bayesian_recursion import BayesianPrediction, ModelState, BayesianMLModel
from .bayesian_enhanced.bayesian_nexus import BayesianNexus
from .bayesian_enhanced.bayesian_data_parser import BayesianDataHandler
# from .bayesian_enhanced.mcmc_engine import run_mcmc_model, example_model

# Legacy compatibility - import from enhanced versions
from .bayesian_enhanced import bayesian_inference as _legacy_inference
from .bayesian_enhanced import bayesian_interface as _legacy_interface

# Re-export for backward compatibility
TrinityVectorInference = TrinityVector
BayesianInferencer = UnifiedBayesianInferencer

__all__ = [
    # Core components
    'BayesianInterface',
    'ProbabilisticResult',
    'TrueP',
    'TrinityVector',
    'UnifiedBayesianInferencer',
    'IELEpistemicState',

    # Enhanced components
    'BayesianTrinityInferencer',
    'execute_HBN',
    'run_HBN_analysis',
    'query_intent_analyzer',
    'resolve_priors_path',
    'load_priors',
    'save_priors',
    'score_data_point',
    'assign_confidence',
    'filter_and_score',
    'predictive_refinement',
    'run_BERT_pipeline',
    'BayesianPrediction',
    'ModelState',
    'BayesianMLModel',
    'BayesianNexus',
    'BayesianDataHandler',
    # 'run_mcmc_model',
    # 'example_model',

    # Legacy compatibility
    'TrinityVectorInference',
    'BayesianInferencer'
]