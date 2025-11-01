# ontological_lattice.py

from .iel_overlays import IELOverlay
from .dual_bijective_logic import OntologicalPrimitive

class OntologicalLattice:
    def __init__(self):
        self.first_order = {
            'Identity': OntologicalPrimitive("Identity"),
            'NonContradiction': OntologicalPrimitive("NonContradiction"),
            'ExcludedMiddle': OntologicalPrimitive("ExcludedMiddle")
        }
        self.second_order = {
            'Distinction': OntologicalPrimitive("Distinction"),
            'Relation': OntologicalPrimitive("Relation"),
            'Agency': OntologicalPrimitive("Agency")
        }
        self.transcendentals = {
            'Coherence': OntologicalPrimitive("Coherence"),
            'Truth': OntologicalPrimitive("Truth"),
            'Existence': OntologicalPrimitive("Existence"),
            'Goodness': OntologicalPrimitive("Goodness")
        }
        self.iel = IELOverlay()

    def validate_interdependencies(self):
        assert self.iel.verify_dependency('Identity', 'Distinction')
        assert self.iel.verify_dependency('NonContradiction', 'Relation')
        assert self.iel.verify_dependency('ExcludedMiddle', 'Agency')
        assert self.iel.verify_isomorphism('Coherence', 'Truth')
        assert self.iel.verify_isomorphism('Existence', 'Goodness')

    def export(self):
        return {
            "first_order": list(self.first_order.keys()),
            "second_order": list(self.second_order.keys()),
            "transcendentals": list(self.transcendentals.keys()),
            "iel_bindings": self.iel.export()
        }
