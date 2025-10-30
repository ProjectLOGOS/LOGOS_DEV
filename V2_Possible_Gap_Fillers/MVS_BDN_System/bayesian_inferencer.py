# bayesian_inferencer.py

import json
from typing import List, Dict, Tuple, Optional

class BayesianTrinityInferencer:
    def __init__(self, prior_path: str = "bayes_priors.json"):
        with open(prior_path, "r") as f:
            self.priors = json.load(f)

    def infer(self, keywords: List[str], weights: Optional[List[float]] = None) -> Dict:
        """
        Infer trinity vector and initial complex value based on keyword priors.
        Args:
            keywords: list of key concepts
            weights: optional list of weights (same length)
        Returns:
            {
              "trinity": (E, G, T),
              "c": complex,
              "source_terms": [terms used]
            }
        """
        if not keywords:
            raise ValueError("Must provide at least one keyword.")

        if weights and len(weights) != len(keywords):
            raise ValueError("Length of weights must match keywords.")

        total_e, total_g, total_t = 0.0, 0.0, 0.0
        count = 0

        for i, term in enumerate(keywords):
            entry = self.priors.get(term.lower())
            if entry:
                w = weights[i] if weights else 1.0
                total_e += entry["E"] * w
                total_g += entry["G"] * w
                total_t += entry["T"] * w
                count += w
            else:
                print(f"[WARN] No prior found for '{term}'")

        if count == 0:
            raise ValueError("No valid priors found for given keywords.")

        avg_e = total_e / count
        avg_g = total_g / count
        avg_t = total_t / count
        c = complex(avg_e * avg_t, avg_g)

        return {
            "trinity": (round(avg_e, 3), round(avg_g, 3), round(avg_t, 3)),
            "c": c,
            "source_terms": keywords
        }
