# prediction_module.py

from typing import List, Optional, Dict, Any
from bayesian_inferencer import BayesianTrinityInferencer
from Ontological_Node import OntologicalNode
from CONSCIOUS_Modal_Inference_System import ThonocVerifier
import time
import json

class TrinityPredictionEngine:
    def __init__(self, prior_path="bayes_priors.json"):
        self.inferencer = BayesianTrinityInferencer(prior_path)

    def predict(self, keywords: List[str], weights: Optional[List[float]] = None, log=False, comment=None) -> Dict[str, Any]:
        """
        Run a prediction based on ontological priors and return a detailed forecast.

        Args:
            keywords: Concepts or topics
            weights: Optional weights per concept
            log: If True, export to prediction log
            comment: Optional note to include in log

        Returns:
            Prediction dictionary
        """
        prior_result = self.inferencer.infer(keywords, weights)
        trinity = prior_result["trinity"]
        c = prior_result["c"]
        terms = prior_result["source_terms"]

        # Orbit analysis
        node = OntologicalNode(c)
        orbit_props = node.orbit_properties

        # Modal judgment
        modal_result = ThonocVerifier.calculate_status(*trinity)

        # Package prediction
        result = {
            "timestamp": time.time(),
            "source_terms": terms,
            "trinity": trinity,
            "c_value": str(c),
            "modal_status": modal_result["status"],
            "coherence": modal_result["coherence"],
            "fractal": {
                "iterations": orbit_props["depth"],
                "in_set": orbit_props["in_set"],
                "type": orbit_props["type"]
            },
            "comment": comment
        }

        if log:
            self.log_prediction(result)

        return result

    def log_prediction(self, result: Dict[str, Any], path="prediction_log.jsonl"):
        """Append a result to log file."""
        with open(path, "a") as f:
            f.write(json.dumps(result) + "\n")
