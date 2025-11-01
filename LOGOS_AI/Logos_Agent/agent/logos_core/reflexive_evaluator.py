# reflexive_self_evaluator.py

from .onto_lattice import OntologicalLattice
from .modal_privative_overlays import ModalEvaluator
from .modal_privative_overlays import Privative

class ReflexiveSelfEvaluator:
    def __init__(self, agent_identity: str, lattice: OntologicalLattice):
        self.agent_identity = agent_identity
        self.lattice = lattice
        self.modal = ModalEvaluator()
        self.privative = Privative

    def evaluate_self_identity(self) -> bool:
        """Confirm that the agent's identity exists and is non-contradictory."""
        # Check if agent identity is registered in the ontological lattice
        return self.agent_identity in [p.name for p in self.lattice.first_order.values()]

    def verify_modal_self_possibility(self) -> bool:
        """Determine if agent's self-model is modally possible in all accessible worlds."""
        return self.modal.is_possible(self.agent_identity)

    def detect_privation_failures(self) -> list:
        """Return list of ontological attributes where agent's instantiation is void or non-permissible."""
        failed = []
        for prop_name, prop in self.lattice.transcendentals.items():
            # Check if property is deprived for this agent
            privation = self.privative(prop_name.lower())
            if self.modal.evaluate_privation(privation):
                failed.append(prop_name)
        return failed

    def self_reflexive_report(self) -> dict:
        """Consolidated self-evaluation report for introspective analysis and adaptive correction."""
        identity_check = self.evaluate_self_identity()
        modal_check = self.verify_modal_self_possibility()
        deprivations = self.detect_privation_failures()
        return {
            "identity_consistent": identity_check,
            "modal_valid": modal_check,
            "deprived_properties": deprivations,
            "fully_self_coherent": identity_check and modal_check and not deprivations
        }
