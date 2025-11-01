# divergence_engine.py

from typing import Tuple, List, Dict, Any
from Ontological_Node import OntologicalNode
from CONSCIOUS_Modal_Inference_System import ThonocVerifier
import itertools

class DivergenceTreeEngine:
    def __init__(self, delta: float = 0.05, branches: int = 8):
        self.delta = delta
        self.branches = branches

    def generate_variants(self, base: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        Generate variants by applying +/- delta combinations.
        Returns unique trinity perturbations.
        """
        e0, g0, t0 = base
        shifts = [-self.delta, 0.0, self.delta]
        variants = set()

        for delta_e, delta_g, delta_t in itertools.product(shifts, repeat=3):
            new_e = min(max(round(e0 + delta_e, 3), 0.0), 1.0)
            new_g = min(max(round(g0 + delta_g, 3), 0.0), 1.0)
            new_t = min(max(round(t0 + delta_t, 3), 0.0), 1.0)
            variants.add((new_e, new_g, new_t))

        return list(variants)[:self.branches]

    def evaluate_branch(self, trinity: Tuple[float, float, float]) -> Dict[str, Any]:
        c = complex(trinity[0] * trinity[2], trinity[1])
        node = OntologicalNode(c)
        modal = ThonocVerifier.calculate_status(*trinity)

        return {
            "trinity": trinity,
            "modal_status": modal["status"],
            "coherence": modal["coherence"],
            "fractal": {
                "depth": node.orbit_properties["depth"],
                "in_set": node.orbit_properties["in_set"],
                "type": node.orbit_properties["type"]
            }
        }

    def simulate_tree(self, base: Tuple[float, float, float], sort_by: str = "coherence") -> List[Dict[str, Any]]:
        variants = self.generate_variants(base)
        results = [self.evaluate_branch(variant) for variant in variants]

        if sort_by == "coherence":
            results.sort(key=lambda x: x["coherence"], reverse=True)
        elif sort_by == "depth":
            results.sort(key=lambda x: x["fractal"]["depth"], reverse=True)
        elif sort_by == "modal":
            modal_order = {"necessary": 3, "actual": 2, "possible": 1, "impossible": 0}
            results.sort(key=lambda x: modal_order.get(x["modal_status"], -1), reverse=True)

        return results
