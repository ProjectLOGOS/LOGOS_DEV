"""Real-time Bayesian update pipeline used by the Bayesian nexus."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
    resolved = resolve_priors_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_priors(data: Dict, path: Optional[Union[str, Path]]) -> None:
    resolved = resolve_priors_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def score_data_point(data_point: Dict) -> int:
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
    if score < 3:
        return 0.0
    weight_map = {3: 0.755, 4: 1.0}
    return weight_map.get(score, 0.0)


def filter_and_score(raw_data: List[Dict]) -> List[Dict]:
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
    # Placeholder for downstream integration hook.
    return []


def run_BERT_pipeline(
    priors_path: Optional[Union[str, Path]], query: str
) -> Tuple[bool, str]:
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
