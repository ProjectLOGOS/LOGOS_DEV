"""
CLI for summarizing fractal orbital predictions.
Scaffold + operational code
"""

import json


def load_predictions(path="prediction_log.jsonl"):
    with open(path) as f:
        return [json.loads(line) for line in f]


def summarize_predictions(preds: list) -> dict:
    summary = {"total": len(preds), "modal_counts": {}, "coherence_avg": 0.0}
    total_coh = 0
    for p in preds:
        s = p.get("modal_status")
        summary["modal_counts"][s] = summary["modal_counts"].get(s, 0) + 1
        total_coh += p.get("coherence", 0)
    summary["coherence_avg"] = round(total_coh / len(preds), 3) if preds else 0
    return summary


if __name__ == "__main__":
    prs = load_predictions()
    print(json.dumps(summarize_predictions(prs), indent=2))
