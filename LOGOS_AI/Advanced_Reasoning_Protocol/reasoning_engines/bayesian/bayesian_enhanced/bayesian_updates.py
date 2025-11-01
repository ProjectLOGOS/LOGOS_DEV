"""
bayesian_updates.py

Consolidated Bayesian updates and hierarchical network analysis.
Combines real-time Bayesian updates with hierarchical Bayesian network functionality.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

# Configuration constants
CONFIDENCE_THRESHOLD = 0.755
MAX_ITERATIONS = 2

_DEFAULT_PRIORS_PATH = (
    Path(__file__).resolve().parents[3]
    / "reasoning_pipeline"
    / "interfaces"
    / "services"
    / "workers"
    / "config_bayes_priors.json"
)


# =============================================================================
# PRIOR MANAGEMENT FUNCTIONS
# =============================================================================

def resolve_priors_path(priors_path: Optional[Union[str, Path]]) -> Path:
    """Resolve the priors file location with sensible fallbacks."""

    candidates: List[Path] = []
    if priors_path:
        provided = Path(priors_path).expanduser()
        candidates.append(provided)
        if not provided.is_absolute():
            module_root = Path(__file__).resolve().parent
            candidates.append(module_root / provided)
            candidates.append(Path(__file__).resolve().parents[3] / provided)

    candidates.append(_DEFAULT_PRIORS_PATH)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Unable to locate Bayesian priors file. Checked: {searched}")


def load_priors(path: Optional[Union[str, Path]]) -> Dict:
    """Load priors from JSON file"""
    resolved = resolve_priors_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_static_priors(path: str = "config/bayes_priors.json") -> dict:
    """Load static priors (alias for load_priors for backward compatibility)"""
    return load_priors(path)


def save_priors(data: Dict, path: Optional[Union[str, Path]]) -> None:
    """Save priors to JSON file"""
    resolved = resolve_priors_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


# =============================================================================
# REAL-TIME BAYESIAN UPDATE FUNCTIONS
# =============================================================================

def score_data_point(data_point: Dict) -> int:
    """Score a data point based on EGTC criteria"""
    score = 0
    if data_point.get("exists", False):
        score += 1
    if data_point.get("good", False):
        score += 1
    if data_point.get("true", False):
        score += 1
    if data_point.get("coherent", False):
        score += 1
    return score


def assign_confidence(score: int) -> float:
    """Assign confidence based on EGTC score"""
    if score < 3:
        return 0.0
    weight_map = {3: 0.755, 4: 1.0}
    return weight_map.get(score, 0.0)


def filter_and_score(raw_data: List[Dict]) -> List[Dict]:
    """Filter and score raw data points"""
    valid_points = []
    for data_point in raw_data:
        score = score_data_point(data_point)
        confidence = assign_confidence(score)
        if confidence >= CONFIDENCE_THRESHOLD:
            data_point["EGTC_score"] = score
            data_point["confidence"] = confidence
            valid_points.append(data_point)
    return valid_points


def predictive_refinement(query: str, tier: int = 1) -> List[Dict]:
    """Placeholder for downstream integration hook"""
    return []


def run_BERT_pipeline(
    priors_path: Optional[Union[str, Path]], query: str
) -> Tuple[bool, str]:
    """Run BERT-based real-time Bayesian update pipeline"""
    priors_file = resolve_priors_path(priors_path)
    priors = load_priors(priors_file)
    attempt_log: List[str] = []

    for attempt in range(MAX_ITERATIONS):
        tier = 1 if attempt == 0 else 2
        raw_data = predictive_refinement(query, tier=tier)
        filtered = filter_and_score(raw_data)

        if not filtered:
            attempt_log.append(
                f"Attempt {attempt + 1}: No valid priors passed EGTC threshold."
            )
            continue

        average_confidence = sum(dp["confidence"] for dp in filtered) / len(filtered)

        if average_confidence >= CONFIDENCE_THRESHOLD:
            for data_point in filtered:
                priors[data_point["label"]] = {
                    "value": data_point["value"],
                    "confidence": data_point["confidence"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "EGTC_score": data_point["EGTC_score"],
                }
            save_priors(priors, priors_file)
            return (
                True,
                f"Success on attempt {attempt + 1} with average confidence {average_confidence:.3f}.",
            )

        attempt_log.append(
            f"Attempt {attempt + 1}: Average confidence {average_confidence:.3f} below threshold."
        )

    return False, "BERT failed all refinement attempts:\n" + "\n".join(attempt_log)


# =============================================================================
# HIERARCHICAL BAYESIAN NETWORK FUNCTIONS
# =============================================================================

def query_intent_analyzer(q: str) -> dict:
    """Analyze query intent and flag inappropriate content"""
    flags = []
    lw = q.lower()
    if any(w in lw for w in ["dragon", "wizard", "hogwarts"]):
        flags.append("fictional")
    return {
        "is_valid": not flags,
        "flags": flags,
        "action": ("reroute" if flags else "proceed"),
    }


def preprocess_query(q: str) -> str:
    """Preprocess query text for analysis"""
    return re.sub(r"[^\\w\\s]", "", q.lower())


def run_HBN_analysis(query: str, priors: dict) -> dict:
    """Run hierarchical Bayesian network analysis"""
    cats = list(priors.keys())
    vals = np.array([priors.get(c, 0) for c in cats]).reshape(-1, 1)
    sc = StandardScaler().fit_transform(vals)
    mdl = BayesianRidge().fit(np.arange(len(cats)).reshape(-1, 1), sc.ravel())
    idx = len(preprocess_query(query)) % len(cats)
    return {"prediction": mdl.predict([[idx]])[0], "category": cats[idx]}


def execute_HBN(query: str) -> dict:
    """Execute complete HBN pipeline with intent analysis"""
    p = load_static_priors()
    intent = query_intent_analyzer(query)
    if not intent["is_valid"]:
        print("Flags:", intent["flags"])
        return {}
    return run_HBN_analysis(query, p)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Prior management
    "resolve_priors_path",
    "load_priors",
    "load_static_priors",
    "save_priors",

    # Real-time updates
    "score_data_point",
    "assign_confidence",
    "filter_and_score",
    "predictive_refinement",
    "run_BERT_pipeline",

    # Hierarchical Bayesian Network
    "query_intent_analyzer",
    "preprocess_query",
    "run_HBN_analysis",
    "execute_HBN",
]