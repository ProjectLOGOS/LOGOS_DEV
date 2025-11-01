# pxl_core_logic.py

from enum import Enum
from typing import Any


class LogicalState(Enum):
    TRUE = 1
    FALSE = 0
    UNKNOWN = -1


class PXLIdentity:
    def __init__(self, label: str):
        self.label = label

    def is_distinct_from(self, other: 'PXLIdentity') -> bool:
        return self.label != other.label

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, PXLIdentity) and self.label == other.label

    def __hash__(self):
        return hash(self.label)

    def __str__(self):
        return f"ID<{self.label}>"


class PXLRelation:
    def __init__(self, source: PXLIdentity, target: PXLIdentity, relation_type: str):
        self.source = source
        self.target = target
        self.relation_type = relation_type

    def is_valid(self) -> bool:
        return self.source != self.target


class PXLLogicCore:
    def __init__(self):
        self.entities = set()
        self.relations = []

    def register_entity(self, label: str) -> PXLIdentity:
        entity = PXLIdentity(label)
        self.entities.add(entity)
        return entity

    def add_relation(self, source: PXLIdentity, target: PXLIdentity, rel_type: str) -> None:
        relation = PXLRelation(source, target, rel_type)
        if relation.is_valid():
            self.relations.append(relation)

    def get_relations_for_entity(self, entity: PXLIdentity) -> list:
        return [r for r in self.relations if r.source == entity or r.target == entity]

    def validate_consistency(self) -> bool:
        """Basic consistency check for the PXL knowledge graph"""
        # Check for self-referencing relations
        for relation in self.relations:
            if not relation.is_valid():
                return False
        return True
