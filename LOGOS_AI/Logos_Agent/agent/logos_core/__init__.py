"""
LOGOS Core Logic Module
========================

Consolidated foundational logic frameworks for the LOGOS AI system.

This module contains:
- Modal and privative logic operators (PXL)
- Integrated Epistemic Logic (IEL) domains
- Dual bijective ontological mappings
- Ontological lattice structures
- Reflexive self-evaluation systems
- Core PXL identity and relation management

All components are designed to work together to provide the logical foundation
for consciousness, reasoning, and agent decision-making.
"""

from .modal_privative_overlays import ModalEvaluator, Modality, Privative
from .pxl_core_logic import PXLLogicCore, PXLIdentity, PXLRelation, LogicalState
from .iel_overlays import IELOverlay, IELDomain
from .dual_bijective_logic import DualBijectiveSystem, OntologicalPrimitive
from .onto_lattice import OntologicalLattice
from .reflexive_evaluator import ReflexiveSelfEvaluator

__version__ = "1.0.0"
__all__ = [
    # Modal Logic
    'ModalEvaluator', 'Modality', 'Privative',

    # PXL Core Logic
    'PXLLogicCore', 'PXLIdentity', 'PXLRelation', 'LogicalState',

    # IEL Domains
    'IELOverlay', 'IELDomain',

    # Dual Bijective Logic
    'DualBijectiveSystem', 'OntologicalPrimitive',

    # Ontological Lattice
    'OntologicalLattice',

    # Reflexive Evaluation
    'ReflexiveSelfEvaluator'
]