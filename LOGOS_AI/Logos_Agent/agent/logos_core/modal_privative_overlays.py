"""
modal_privative_overlays.py

Defines modal and privative logic overlays for the PXL logic system.
"""

from enum import Enum

class Modality(Enum):
    POSSIBLE = "◇"
    NECESSARY = "□"
    IMPOSSIBLE = "¬◇"
    CONTINGENT = "¬□"

class Privative:
    def __init__(self, property_name: str):
        self.property = property_name

    def __repr__(self):
        return f"¬{self.property}"

    def is_privative_of(self, other: str) -> bool:
        return self.property == other

class ModalEvaluator:
    def __init__(self):
        self.known_worlds = {}

    def add_world(self, world_name: str, truths: set):
        self.known_worlds[world_name] = truths

    def is_possible(self, proposition: str) -> bool:
        return any(proposition in truths for truths in self.known_worlds.values())

    def is_necessary(self, proposition: str) -> bool:
        return all(proposition in truths for truths in self.known_worlds.values())

    def is_impossible(self, proposition: str) -> bool:
        return not self.is_possible(proposition)

    def is_contingent(self, proposition: str) -> bool:
        return self.is_possible(proposition) and not self.is_necessary(proposition)

    def evaluate_privation(self, privative: Privative) -> bool:
        return all(privative.property not in truths for truths in self.known_worlds.values())
