"""
Enhanced Bayesian Reasoning Components
======================================

Advanced Bayesian inference system with enhanced capabilities for LOGOS AGI.
Includes MCMC engines, hierarchical networks, real-time updates, and recursive reasoning.
"""

# Core Bayesian inference (consolidated from multiple files)
from .bayesian_inference import (
    TrinityVector,
    UnifiedBayesianInferencer,
    IELEpistemicState,
    # Interface components
    ProbabilisticResult,
    TrueP as TrueP_predicate,
    FalseP,
    UncertainP,
    BayesianInterface,
    BayesianNetwork,
    # MCMC components
    run_mcmc_model,
    example_model,
    # Trinity inferencer
    BayesianTrinityInferencer,
    # Nexus orchestrator
    BayesianNexus,
)

# Bayesian updates (consolidated from hierarchical_bayes_network.py + bayes_update_real_time.py)
from .bayesian_updates import (
    resolve_priors_path,
    load_priors,
    load_static_priors,
    save_priors,
    score_data_point,
    assign_confidence,
    filter_and_score,
    predictive_refinement,
    run_BERT_pipeline,
    query_intent_analyzer,
    preprocess_query,
    run_HBN_analysis,
    execute_HBN,
)

# Bayesian ML (consolidated from bayesian_recursion.py + bayesian_data_parser.py)
from .bayesian_ml import (
    BayesianPrediction,
    ModelState,
    BayesianMLModel,
    BayesianDataHandler,
)

__all__ = [
    # Core inference
    'TrinityVector',
    'UnifiedBayesianInferencer',
    'IELEpistemicState',

    # Interface components
    'ProbabilisticResult',
    'TrueP_predicate',
    'FalseP',
    'UncertainP',
    'BayesianInterface',
    'BayesianNetwork',

    # MCMC components
    'run_mcmc_model',
    'example_model',

    # Trinity inferencer
    'BayesianTrinityInferencer',

    # Nexus orchestrator
    'BayesianNexus',

    # Bayesian updates
    'resolve_priors_path',
    'load_priors',
    'load_static_priors',
    'save_priors',
    'score_data_point',
    'assign_confidence',
    'filter_and_score',
    'predictive_refinement',
    'run_BERT_pipeline',
    'query_intent_analyzer',
    'preprocess_query',
    'run_HBN_analysis',
    'execute_HBN',

    # Bayesian ML
    'BayesianPrediction',
    'ModelState',
    'BayesianMLModel',
    'BayesianDataHandler',
]
