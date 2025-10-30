"""
modal_inference.py

Full S5 modal-logic evaluator for THŌNOC.
Enhanced with V2_Possible_Gap_Fillers modal prediction capabilities.
"""

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

# Enhanced Modal Predictor Integration
try:
    from .modal_predictor.class_modal_validator import ModalValidator
    from .modal_predictor.modal_inference import EnhancedModalInference

    ENHANCED_MODAL_AVAILABLE = True
except ImportError:
    ENHANCED_MODAL_AVAILABLE = False


class ModalOperator(Enum):
    NECESSARILY = "□"
    POSSIBLY = "◇"
    ACTUALLY = "A"


class ModalFormula:
    """Represents a modal logic formula with optional operator."""

    def __init__(self, content: str, operator: Optional[ModalOperator] = None):
        self.content = content
        self.operator = operator
        self.subformulas = []

    def __str__(self) -> str:
        return (
            f"{self.operator.value}({self.content})" if self.operator else self.content
        )

    def add_subformula(self, sub: "ModalFormula"):
        sub.parent = self
        self.subformulas.append(sub)

    def is_necessity(self) -> bool:
        return self.operator == ModalOperator.NECESSARILY

    def is_possibility(self) -> bool:
        return self.operator == ModalOperator.POSSIBLY

    def dual(self) -> "ModalFormula":
        if self.is_necessity():
            return ModalFormula(f"¬{self.content}", ModalOperator.POSSIBLY)
        if self.is_possibility():
            return ModalFormula(f"¬{self.content}", ModalOperator.NECESSARILY)
        return self


class WorldNode:
    """Possible world in Kripke model."""

    def __init__(self, name: str, assignments: Dict[str, bool] = None):
        self.name = name
        self.assignments = assignments or {}

    def assign(self, prop: str, val: bool):
        self.assignments[prop] = val

    def evaluate(self, prop: str) -> bool:
        return self.assignments.get(prop, False)


class KripkeModel:
    """Graph of worlds + accessibility for modal semantics."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.worlds = {}

    def add_world(self, name: str, assigns=None):
        w = WorldNode(name, assigns)
        self.worlds[name] = w
        self.graph.add_node(name)
        return w

    def add_access(self, w1: str, w2: str):
        self.graph.add_edge(w1, w2)

    def make_s5(self):
        for n in list(self.worlds):
            self.graph.add_edge(n, n)
        for u, v in list(self.graph.edges()):
            self.graph.add_edge(v, u)
        self.graph = nx.transitive_closure(self.graph)

    def neighbors(self, w):
        return list(self.graph.neighbors(w))

    def eval_necessity(self, prop, w):
        return all(self.worlds[n].evaluate(prop) for n in self.neighbors(w))

    def eval_possibility(self, prop, w):
        return any(self.worlds[n].evaluate(prop) for n in self.neighbors(w))

    def eval(self, formula: ModalFormula, w: str):
        if formula.is_necessity():
            return self.eval_necessity(formula.content, w)
        if formula.is_possibility():
            return self.eval_possibility(formula.content, w)
        return self.worlds[w].evaluate(formula.content)


class S5ModalSystem:
    """Encapsulates an S5 Kripke model for multiple formulas."""

    def __init__(self):
        self.model = KripkeModel()
        self.actual = "w0"
        self.model.add_world(self.actual)
        self.model.make_s5()

    def set_val(self, prop: str, val: bool, world=None):
        w = world or self.actual
        if w not in self.model.worlds:
            self.model.add_world(w)
            self.model.make_s5()
        self.model.worlds[w].assign(prop, val)

    def evaluate(self, formula: ModalFormula, world=None):
        return self.model.eval(formula, world or self.actual)

    def validate_entailment(self, premises: List[ModalFormula], concl: ModalFormula):
        for w in self.model.worlds:
            if all(self.evaluate(p, w) for p in premises) and not self.evaluate(
                concl, w
            ):
                return False
        return True


class ThonocModalInference:
    """High-level modal inference for THŌNOC."""

    def __init__(self):
        self.s5 = S5ModalSystem()
        self.registry = {}
        self.graph = nx.DiGraph()

        # Enhanced Modal Predictor Integration
        if ENHANCED_MODAL_AVAILABLE:
            self.enhanced_modal = EnhancedModalInference()
            self.modal_validator = ModalValidator()
        else:
            self.enhanced_modal = None
            self.modal_validator = None

    def register(self, prop_id: str, content: str, trinity: Tuple[float, float, float]):
        e, g, t = trinity
        nec = t > 0.95 and e > 0.9
        poss = t > 0.05 and e > 0.05
        val = nec or poss
        self.s5.set_val(prop_id, val)
        self.registry[prop_id] = {"content": content, "trinity": trinity}
        self.graph.add_node(prop_id)

    def entail(self, prem: str, concl: str, strength: float):
        if prem in self.registry and concl in self.registry:
            self.graph.add_edge(prem, concl, strength=strength)
            if self.registry[prem].get("necessary"):
                for s in self.graph.successors(prem):
                    self.registry[s]["necessary"] = True

    def trinity_to_modal_status(self, trinity: Tuple[float, float, float]):
        frm = ModalFormula("x")  # dummy
        return {
            "status": self.s5.evaluate(frm),
            "coherence": trinity[0] * trinity[1] * trinity[2],
        }

    def enhanced_modal_inference(
        self,
        proposition: str,
        trinity_context: Tuple[float, float, float],
        modal_depth: int = 3,
    ) -> Dict[str, Any]:
        """
        Enhanced modal inference using V2_Possible_Gap_Fillers modal prediction.

        Args:
            proposition: Proposition for modal analysis
            trinity_context: Trinity vector context (E, G, T)
            modal_depth: Depth of modal operator nesting

        Returns:
            Enhanced modal inference results
        """
        if not ENHANCED_MODAL_AVAILABLE:
            # Fallback to standard modal inference
            return {
                "enhanced": False,
                "standard_result": self.trinity_to_modal_status(trinity_context),
                "proposition": proposition,
            }

        try:
            # Enhanced modal inference
            enhanced_result = self.enhanced_modal.advanced_modal_inference(
                proposition, trinity_context, modal_depth
            )

            # Modal validation
            validation_result = self.modal_validator.validate_modal_proposition(
                proposition, trinity_context
            )

            # Standard inference for comparison
            standard_result = self.trinity_to_modal_status(trinity_context)

            return {
                "enhanced": True,
                "enhanced_result": enhanced_result,
                "validation": validation_result,
                "standard_result": standard_result,
                "coherence_improvement": self._calculate_coherence_improvement(
                    standard_result, enhanced_result
                ),
                "proposition": proposition,
                "trinity_context": trinity_context,
            }

        except Exception as e:
            return {
                "enhanced": False,
                "error": str(e),
                "fallback_result": self.trinity_to_modal_status(trinity_context),
            }

    def _calculate_coherence_improvement(
        self, standard: Dict[str, Any], enhanced: Dict[str, Any]
    ) -> float:
        """Calculate coherence improvement from enhanced inference."""
        try:
            standard_coherence = standard.get("coherence", 0)
            enhanced_coherence = enhanced.get("coherence", standard_coherence)

            if standard_coherence == 0:
                return 1.0 if enhanced_coherence > 0 else 0.0

            return enhanced_coherence / standard_coherence
        except:
            return 1.0  # Neutral improvement

    def batch_enhanced_inference(
        self,
        propositions: List[Tuple[str, Tuple[float, float, float]]],
        parallel_processing: bool = True,
    ) -> Dict[str, Any]:
        """
        Batch processing of enhanced modal inference for multiple propositions.

        Args:
            propositions: List of (proposition, trinity_context) tuples
            parallel_processing: Whether to use parallel processing

        Returns:
            Batch inference results
        """
        if not ENHANCED_MODAL_AVAILABLE:
            return {"enhanced": False, "reason": "Enhanced modal predictor unavailable"}

        results = []
        coherence_scores = []

        for prop, trinity in propositions:
            result = self.enhanced_modal_inference(prop, trinity)
            results.append(result)

            # Collect coherence scores
            if result.get("enhanced"):
                coherence = result.get("enhanced_result", {}).get("coherence", 0)
                coherence_scores.append(coherence)

        # Calculate batch statistics
        if coherence_scores:
            avg_coherence = sum(coherence_scores) / len(coherence_scores)
            max_coherence = max(coherence_scores)
            min_coherence = min(coherence_scores)
        else:
            avg_coherence = max_coherence = min_coherence = 0

        return {
            "enhanced": True,
            "batch_size": len(propositions),
            "results": results,
            "batch_statistics": {
                "average_coherence": avg_coherence,
                "max_coherence": max_coherence,
                "min_coherence": min_coherence,
                "successful_enhancements": len(coherence_scores),
            },
        }
