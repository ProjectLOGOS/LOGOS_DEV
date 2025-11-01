# iel_overlay.py

from enum import Enum, auto
from .modal_privative_overlays import ModalEvaluator, Modality, Privative

class IELDomain(Enum):
    COHERENCE = auto()
    TRUTH = auto()
    EXISTENCE = auto()
    GOODNESS = auto()
    IDENTITY = auto()
    NON_CONTRADICTION = auto()
    EXCLUDED_MIDDLE = auto()
    DISTINCTION = auto()
    RELATION = auto()
    AGENCY = auto()

class IELOverlay:
    def __init__(self):
        self.overlay = {}
        self.modal_evaluator = ModalEvaluator()

    def define_iel(self, domain: IELDomain, modality: Modality, privation=None):
        self.overlay[domain] = {
            "modality": modality,
            "privation": privation or Privative("default")
        }

    def is_viable(self, domain: IELDomain) -> bool:
        """Check if domain is viable using modal evaluation"""
        if domain in self.overlay:
            modality = self.overlay[domain]["modality"]
            # Simple viability check - could be more sophisticated
            return modality in [Modality.NECESSARY, Modality.POSSIBLE]
        return False

    def get_profile(self):
        profile = {}
        for domain in IELDomain:
            viable = self.is_viable(domain)
            profile[domain.name] = {
                "Modality": self.overlay[domain]["modality"].name,
                "Privation": self.overlay[domain]["privation"].name,
                "Viable": viable
            }
        return profile

    def verify_dependency(self, domain1: str, domain2: str) -> bool:
        """Verify dependency relationship between domains"""
        # Simple dependency check - domain1 must be viable for domain2
        d1 = getattr(IELDomain, domain1.upper(), None)
        d2 = getattr(IELDomain, domain2.upper(), None)
        if d1 and d2:
            return self.is_viable(d1) and self.is_viable(d2)
        return False

    def verify_isomorphism(self, domain1: str, domain2: str) -> bool:
        """Verify isomorphic relationship between domains"""
        # Check if domains have compatible modalities
        d1 = getattr(IELDomain, domain1.upper(), None)
        d2 = getattr(IELDomain, domain2.upper(), None)
        if d1 and d2 and d1 in self.overlay and d2 in self.overlay:
            mod1 = self.overlay[d1]["modality"]
            mod2 = self.overlay[d2]["modality"]
            # Isomorphic if they have the same modality
            return mod1 == mod2
        return False

# Example IEL registration
if __name__ == "__main__":
    iel = IELOverlay()
    iel.define_iel(IELDomain.COHERENCE, Modality.NECESSARY)
    iel.define_iel(IELDomain.TRUTH, Modality.NECESSARY)
    iel.define_iel(IELDomain.EXISTENCE, Modality.NECESSARY)
    iel.define_iel(IELDomain.GOODNESS, Modality.NECESSARY)
    iel.define_iel(IELDomain.IDENTITY, Modality.NECESSARY)
    iel.define_iel(IELDomain.NON_CONTRADICTION, Modality.NECESSARY)
    iel.define_iel(IELDomain.EXCLUDED_MIDDLE, Modality.NECESSARY)
    iel.define_iel(IELDomain.DISTINCTION, Modality.NECESSARY)
    iel.define_iel(IELDomain.RELATION, Modality.NECESSARY)
    iel.define_iel(IELDomain.AGENCY, Modality.NECESSARY)

    iel.print_profile()
