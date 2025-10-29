"""
Ontological Properties Validation Engine - UIP Step 1 Component
==============================================================

Comprehensive ontological reasoning system with property validation, consistency checking,
and metaphysical analysis for theological and philosophical concepts.

Adapted from: V2_Possible_Gap_Fillers/ontological_properties_database.py
Enhanced with: Trinity ontology integration, modal metaphysics, scholastic categories
"""

from protocols.shared.system_imports import *
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import itertools
from abc import ABC, abstractmethod


class OntologicalCategory(Enum):
    """Fundamental ontological categories"""
    SUBSTANCE = "substance"         # Primary beings (Aristotelian substances)
    ATTRIBUTE = "attribute"         # Properties of substances
    RELATION = "relation"           # Connections between entities
    MODE = "mode"                   # Ways of being
    ESSENCE = "essence"             # What something is (quiddity)
    EXISTENCE = "existence"         # That something is (esse)
    ACCIDENT = "accident"           # Non-essential properties
    UNIVERSAL = "universal"         # Abstract concepts
    PARTICULAR = "particular"       # Concrete instances
    TRINITY_PERSON = "trinity_person"  # Divine persons in Trinity


class PropertyType(Enum):
    """Types of ontological properties"""
    ESSENTIAL = "essential"         # Properties that define essence
    ACCIDENTAL = "accidental"       # Non-defining properties
    RELATIONAL = "relational"       # Properties involving relations
    TEMPORAL = "temporal"           # Time-dependent properties
    MODAL = "modal"                 # Possibility/necessity properties
    DIVINE = "divine"               # Specifically theological properties
    LOGICAL = "logical"             # Logically necessary properties
    METAPHYSICAL = "metaphysical"   # Metaphysically necessary


class ConsistencyLevel(Enum):
    """Levels of ontological consistency"""
    LOGICALLY_CONSISTENT = "logical"
    METAPHYSICALLY_CONSISTENT = "metaphysical"  
    THEOLOGICALLY_CONSISTENT = "theological"
    PRACTICALLY_CONSISTENT = "practical"
    INCONSISTENT = "inconsistent"


@dataclass
class OntologicalProperty:
    """Individual ontological property"""
    name: str
    category: OntologicalCategory
    property_type: PropertyType
    definition: str
    necessary_conditions: List[str] = field(default_factory=list)
    sufficient_conditions: List[str] = field(default_factory=list)
    incompatible_properties: Set[str] = field(default_factory=set)
    implies_properties: Set[str] = field(default_factory=set)
    modal_status: str = "contingent"  # "necessary", "contingent", "impossible"
    theological_significance: Optional[str] = None
    scholastic_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def is_compatible_with(self, other_property: str) -> bool:
        """Check if this property is compatible with another"""
        return other_property not in self.incompatible_properties
    
    def implies(self, other_property: str) -> bool:
        """Check if this property implies another"""
        return other_property in self.implies_properties


@dataclass
class OntologicalEntity:
    """Entity with ontological properties"""
    name: str
    category: OntologicalCategory
    properties: Set[str] = field(default_factory=set)
    relations: Dict[str, List[str]] = field(default_factory=dict)
    essence: Optional[str] = None
    existence_mode: str = "actual"  # "actual", "possible", "necessary", "impossible"
    trinity_role: Optional[str] = None  # For Trinity persons
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_property(self, property_name: str) -> bool:
        """Check if entity has given property"""
        return property_name in self.properties
    
    def add_property(self, property_name: str) -> None:
        """Add property to entity"""
        self.properties.add(property_name)
    
    def remove_property(self, property_name: str) -> None:
        """Remove property from entity"""
        self.properties.discard(property_name)
    
    def get_relations(self, relation_type: str) -> List[str]:
        """Get entities related by specific relation type"""
        return self.relations.get(relation_type, [])


@dataclass
class ConsistencyCheck:
    """Result of ontological consistency checking"""
    is_consistent: bool
    consistency_level: ConsistencyLevel
    violations: List[str]
    warnings: List[str]
    implications: List[str]
    theological_analysis: Dict[str, Any]
    confidence: float


@dataclass
class OntologicalValidationResult:
    """Comprehensive ontological validation result"""
    entity_name: str
    is_valid: bool
    consistency_check: ConsistencyCheck
    property_analysis: Dict[str, Any]
    modal_analysis: Dict[str, Any]
    trinity_analysis: Optional[Dict[str, Any]]
    scholastic_evaluation: Dict[str, Any]
    recommendations: List[str]


class PropertyDatabase:
    """Database of ontological properties and their relationships"""
    
    def __init__(self):
        self.properties: Dict[str, OntologicalProperty] = {}
        self.entities: Dict[str, OntologicalEntity] = {}
        self._initialize_core_properties()
        self._initialize_divine_properties()
        self._initialize_trinity_properties()
    
    def _initialize_core_properties(self) -> None:
        """Initialize fundamental ontological properties"""
        
        # Existence properties
        self.add_property(OntologicalProperty(
            name="exists",
            category=OntologicalCategory.EXISTENCE,
            property_type=PropertyType.ESSENTIAL,
            definition="Has actual being or reality",
            modal_status="contingent",
            implies_properties={"self_identical"},
            scholastic_analysis={"actus_essendi": True, "esse": "actual"}
        ))
        
        # Essential properties
        self.add_property(OntologicalProperty(
            name="self_identical",
            category=OntologicalCategory.ESSENCE,
            property_type=PropertyType.LOGICAL,
            definition="Identical to itself (x = x)",
            modal_status="necessary",
            scholastic_analysis={"principle_identity": True}
        ))
        
        self.add_property(OntologicalProperty(
            name="rational",
            category=OntologicalCategory.ESSENCE,
            property_type=PropertyType.ESSENTIAL,
            definition="Capable of reason and logical thought",
            necessary_conditions=["conscious", "exists"],
            implies_properties={"conscious", "capable_of_knowledge"},
            scholastic_analysis={"rational_soul": True, "intellect": True}
        ))
        
        self.add_property(OntologicalProperty(
            name="conscious",
            category=OntologicalCategory.ATTRIBUTE,
            property_type=PropertyType.ESSENTIAL,
            definition="Has subjective experience and awareness",
            necessary_conditions=["exists"],
            implies_properties={"capable_of_knowledge"},
            scholastic_analysis={"sensitive_soul": True}
        ))
        
        # Modal properties
        self.add_property(OntologicalProperty(
            name="contingent",
            category=OntologicalCategory.MODE,
            property_type=PropertyType.MODAL,
            definition="Could exist or not exist",
            incompatible_properties={"necessary", "impossible"},
            modal_status="contingent"
        ))
        
        self.add_property(OntologicalProperty(
            name="necessary",
            category=OntologicalCategory.MODE,
            property_type=PropertyType.MODAL,
            definition="Must exist; cannot fail to exist",
            incompatible_properties={"contingent", "impossible"},
            modal_status="necessary",
            implies_properties={"exists", "eternal"}
        ))
        
        # Temporal properties
        self.add_property(OntologicalProperty(
            name="temporal",
            category=OntologicalCategory.MODE,
            property_type=PropertyType.TEMPORAL,
            definition="Exists within time; has temporal extension",
            incompatible_properties={"eternal", "timeless"},
            necessary_conditions=["exists"]
        ))
        
        self.add_property(OntologicalProperty(
            name="eternal",
            category=OntologicalCategory.MODE,
            property_type=PropertyType.TEMPORAL,
            definition="Exists outside of time; has no temporal limits",
            incompatible_properties={"temporal"},
            implies_properties={"necessary", "immutable"}
        ))
    
    def _initialize_divine_properties(self) -> None:
        """Initialize divine/theological properties"""
        
        self.add_property(OntologicalProperty(
            name="omnipotent",
            category=OntologicalCategory.ATTRIBUTE,
            property_type=PropertyType.DIVINE,
            definition="All-powerful; capable of any logically possible action",
            necessary_conditions=["exists", "divine"],
            implies_properties={"capable_of_creation", "sovereign"},
            theological_significance="Classical divine attribute",
            scholastic_analysis={"potentia_absoluta": True, "divine_power": "infinite"}
        ))
        
        self.add_property(OntologicalProperty(
            name="omniscient",
            category=OntologicalCategory.ATTRIBUTE,
            property_type=PropertyType.DIVINE,
            definition="All-knowing; knows all actual and possible truths",
            necessary_conditions=["exists", "divine", "rational"],
            implies_properties={"perfect_knowledge", "infallible"},
            theological_significance="Classical divine attribute",
            scholastic_analysis={"scientia_divina": True, "divine_intellect": "perfect"}
        ))
        
        self.add_property(OntologicalProperty(
            name="omnipresent",
            category=OntologicalCategory.RELATION,
            property_type=PropertyType.DIVINE,
            definition="Present everywhere; not limited by space",
            necessary_conditions=["exists", "divine"],
            implies_properties={"immutable", "simple"},
            theological_significance="Divine ubiquity",
            scholastic_analysis={"divine_presence": "substantial", "ubiquity": True}
        ))
        
        self.add_property(OntologicalProperty(
            name="divine",
            category=OntologicalCategory.ESSENCE,
            property_type=PropertyType.ESSENTIAL,
            definition="Having the nature of God; supremely perfect being",
            implies_properties={"perfect", "necessary", "eternal", "immutable", "simple"},
            modal_status="necessary",
            theological_significance="Divine essence",
            scholastic_analysis={"ipsum_esse_subsistens": True, "pure_act": True}
        ))
        
        self.add_property(OntologicalProperty(
            name="perfect",
            category=OntologicalCategory.ATTRIBUTE,
            property_type=PropertyType.DIVINE,
            definition="Lacking no excellence; maximally great",
            implies_properties={"good", "true", "beautiful", "complete"},
            theological_significance="Divine perfection",
            scholastic_analysis={"perfectio_pura": True, "excellence": "maximal"}
        ))
    
    def _initialize_trinity_properties(self) -> None:
        """Initialize Trinity-specific properties"""
        
        self.add_property(OntologicalProperty(
            name="trinity_person",
            category=OntologicalCategory.TRINITY_PERSON,
            property_type=PropertyType.DIVINE,
            definition="Distinct person within the Trinity",
            necessary_conditions=["divine", "personal", "relational"],
            implies_properties={"consubstantial", "co_eternal", "co_equal"},
            theological_significance="Trinitarian personhood",
            scholastic_analysis={"persona": True, "subsistentia": True, "relatio_subsistens": True}
        ))
        
        self.add_property(OntologicalProperty(
            name="consubstantial",
            category=OntologicalCategory.ESSENCE,
            property_type=PropertyType.DIVINE,
            definition="Sharing the same divine substance/essence",
            necessary_conditions=["trinity_person"],
            theological_significance="Nicene doctrine of consubstantiality",
            scholastic_analysis={"homoousios": True, "una_substantia": True}
        ))
        
        self.add_property(OntologicalProperty(
            name="paternal",
            category=OntologicalCategory.RELATION,
            property_type=PropertyType.DIVINE,
            definition="Having the relation of Father within Trinity",
            necessary_conditions=["trinity_person"],
            implies_properties={"unbegotten", "spirator"},
            incompatible_properties={"filial", "spirated"},
            theological_significance="Father's unique relation",
            scholastic_analysis={"paternitas": True, "principium_sine_principio": True}
        ))
        
        self.add_property(OntologicalProperty(
            name="filial",
            category=OntologicalCategory.RELATION,
            property_type=PropertyType.DIVINE,
            definition="Having the relation of Son within Trinity",
            necessary_conditions=["trinity_person"],
            implies_properties={"begotten", "spirator"},
            incompatible_properties={"paternal", "spirated"},
            theological_significance="Son's unique relation",
            scholastic_analysis={"filiatio": True, "verbum": True}
        ))
        
        self.add_property(OntologicalProperty(
            name="spirated",
            category=OntologicalCategory.RELATION,
            property_type=PropertyType.DIVINE,
            definition="Proceeding as Spirit within Trinity",
            necessary_conditions=["trinity_person"],
            implies_properties={"proceeds_from_father_and_son"},
            incompatible_properties={"paternal", "filial"},
            theological_significance="Spirit's unique relation",
            scholastic_analysis={"spiratio_passiva": True, "amor": True}
        ))
    
    def add_property(self, prop: OntologicalProperty) -> None:
        """Add property to database"""
        self.properties[prop.name] = prop
    
    def get_property(self, name: str) -> Optional[OntologicalProperty]:
        """Get property by name"""
        return self.properties.get(name)
    
    def add_entity(self, entity: OntologicalEntity) -> None:
        """Add entity to database"""
        self.entities[entity.name] = entity
    
    def get_entity(self, name: str) -> Optional[OntologicalEntity]:
        """Get entity by name"""
        return self.entities.get(name)
    
    def get_properties_by_category(self, category: OntologicalCategory) -> List[OntologicalProperty]:
        """Get all properties of given category"""
        return [prop for prop in self.properties.values() if prop.category == category]


class ConsistencyChecker:
    """Ontological consistency checking engine"""
    
    def __init__(self, property_db: PropertyDatabase):
        self.property_db = property_db
        self.logger = logging.getLogger(__name__)
    
    def check_entity_consistency(self, entity: OntologicalEntity) -> ConsistencyCheck:
        """Check consistency of entity's properties"""
        
        violations = []
        warnings = []
        implications = []
        
        # Check property compatibility
        entity_props = list(entity.properties)
        for i, prop1_name in enumerate(entity_props):
            prop1 = self.property_db.get_property(prop1_name)
            if not prop1:
                warnings.append(f"Unknown property: {prop1_name}")
                continue
            
            # Check incompatibilities
            for j, prop2_name in enumerate(entity_props):
                if i != j:
                    if not prop1.is_compatible_with(prop2_name):
                        violations.append(f"Incompatible properties: {prop1_name} and {prop2_name}")
            
            # Check necessary conditions
            for necessary in prop1.necessary_conditions:
                if necessary not in entity.properties:
                    violations.append(f"Property {prop1_name} requires {necessary}")
            
            # Check implications
            for implied in prop1.implies_properties:
                if implied not in entity.properties:
                    implications.append(f"Property {prop1_name} implies {implied}")
        
        # Determine consistency level
        consistency_level = self._determine_consistency_level(violations, warnings, entity)
        
        # Theological analysis
        theological_analysis = self._analyze_theological_consistency(entity)
        
        # Calculate confidence
        confidence = self._calculate_consistency_confidence(violations, warnings, implications)
        
        return ConsistencyCheck(
            is_consistent=len(violations) == 0,
            consistency_level=consistency_level,
            violations=violations,
            warnings=warnings,
            implications=implications,
            theological_analysis=theological_analysis,
            confidence=confidence
        )
    
    def _determine_consistency_level(
        self, 
        violations: List[str], 
        warnings: List[str],
        entity: OntologicalEntity
    ) -> ConsistencyLevel:
        """Determine the level of consistency"""
        
        if violations:
            return ConsistencyLevel.INCONSISTENT
        
        # Check for logical consistency
        logical_violations = [v for v in violations if "logical" in v.lower()]
        if logical_violations:
            return ConsistencyLevel.INCONSISTENT
        
        # If Trinity-related, check theological consistency
        if entity.trinity_role or "divine" in entity.properties:
            if warnings:
                return ConsistencyLevel.PRACTICALLY_CONSISTENT
            else:
                return ConsistencyLevel.THEOLOGICALLY_CONSISTENT
        
        # Check metaphysical consistency
        modal_properties = {"necessary", "contingent", "possible", "impossible"}
        entity_modalities = entity.properties & modal_properties
        if len(entity_modalities) > 1:
            return ConsistencyLevel.PRACTICALLY_CONSISTENT
        
        if warnings:
            return ConsistencyLevel.PRACTICALLY_CONSISTENT
        
        return ConsistencyLevel.METAPHYSICALLY_CONSISTENT
    
    def _analyze_theological_consistency(self, entity: OntologicalEntity) -> Dict[str, Any]:
        """Analyze theological consistency of entity"""
        
        analysis = {
            "is_theological_entity": False,
            "divine_attributes_consistent": True,
            "trinity_analysis": {},
            "scholastic_evaluation": {}
        }
        
        # Check if entity has divine properties
        divine_props = {prop for prop in entity.properties 
                       if self.property_db.get_property(prop) and 
                       self.property_db.get_property(prop).property_type == PropertyType.DIVINE}
        
        if divine_props:
            analysis["is_theological_entity"] = True
            
            # Check divine attribute consistency
            if "omnipotent" in entity.properties and "omniscient" in entity.properties:
                # Check for theological paradoxes
                if "creates_stone_too_heavy" in entity.properties:
                    analysis["divine_attributes_consistent"] = False
                    analysis["omnipotence_paradox"] = True
            
            # Trinity analysis
            if entity.trinity_role:
                analysis["trinity_analysis"] = self._analyze_trinity_person(entity)
            
            # Scholastic evaluation
            analysis["scholastic_evaluation"] = self._scholastic_evaluation(entity)
        
        return analysis
    
    def _analyze_trinity_person(self, entity: OntologicalEntity) -> Dict[str, Any]:
        """Analyze Trinity person consistency"""
        
        trinity_analysis = {
            "valid_trinity_person": True,
            "unique_relations": [],
            "shared_attributes": [],
            "relational_consistency": True
        }
        
        # Check for unique relational properties
        trinity_relations = {"paternal", "filial", "spirated"}
        entity_relations = entity.properties & trinity_relations
        
        if len(entity_relations) > 1:
            trinity_analysis["valid_trinity_person"] = False
            trinity_analysis["relational_consistency"] = False
            trinity_analysis["error"] = "Trinity person cannot have multiple relational properties"
        elif len(entity_relations) == 1:
            trinity_analysis["unique_relations"] = list(entity_relations)
        
        # Check shared divine attributes
        divine_attrs = {"omnipotent", "omniscient", "omnipresent", "eternal", "immutable"}
        shared_attrs = entity.properties & divine_attrs
        trinity_analysis["shared_attributes"] = list(shared_attrs)
        
        return trinity_analysis
    
    def _scholastic_evaluation(self, entity: OntologicalEntity) -> Dict[str, Any]:
        """Evaluate entity using scholastic principles"""
        
        evaluation = {
            "actus_et_potentia": {},
            "esse_et_essentia": {},
            "simplicitas": False,
            "perfectio": {}
        }
        
        # Act and potency analysis
        if "perfect" in entity.properties:
            evaluation["actus_et_potentia"] = {
                "pure_act": True,
                "no_potency": True,
                "fully_actualized": True
            }
        
        # Essence and existence
        if "necessary" in entity.properties:
            evaluation["esse_et_essentia"] = {
                "essence_is_existence": True,
                "ipsum_esse_subsistens": "divine" in entity.properties
            }
        
        # Simplicity
        if "divine" in entity.properties:
            evaluation["simplicitas"] = True
        
        # Perfection analysis
        perfection_props = {"perfect", "good", "true", "beautiful"}
        entity_perfections = entity.properties & perfection_props
        evaluation["perfectio"] = {
            "transcendental_properties": list(entity_perfections),
            "convertibility": len(entity_perfections) == len(perfection_props)
        }
        
        return evaluation
    
    def _calculate_consistency_confidence(
        self,
        violations: List[str],
        warnings: List[str], 
        implications: List[str]
    ) -> float:
        """Calculate confidence in consistency assessment"""
        
        # Base confidence
        base_confidence = 1.0
        
        # Reduce confidence for violations
        base_confidence -= len(violations) * 0.2
        
        # Reduce confidence for warnings  
        base_confidence -= len(warnings) * 0.1
        
        # Slight reduction for unresolved implications
        base_confidence -= len(implications) * 0.05
        
        return max(0.0, min(1.0, base_confidence))


class OntologicalValidator:
    """Main ontological validation engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.property_db = PropertyDatabase()
        self.consistency_checker = ConsistencyChecker(self.property_db)
        
        # Validation cache
        self._validation_cache: Dict[str, OntologicalValidationResult] = {}
        
        self.logger.info("Ontological validator initialized")
    
    def validate_entity(
        self,
        entity_name: str,
        properties: List[str],
        category: Optional[OntologicalCategory] = None,
        trinity_role: Optional[str] = None
    ) -> OntologicalValidationResult:
        """
        Validate ontological entity with given properties
        
        Args:
            entity_name: Name of entity to validate
            properties: List of property names
            category: Ontological category (inferred if None)
            trinity_role: Trinity role if applicable
            
        Returns:
            OntologicalValidationResult: Comprehensive validation results
        """
        try:
            # Create entity
            entity = OntologicalEntity(
                name=entity_name,
                category=category or self._infer_category(properties),
                properties=set(properties),
                trinity_role=trinity_role
            )
            
            # Check cache
            cache_key = f"{entity_name}_{sorted(properties)}_{category}_{trinity_role}"
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]
            
            # Perform consistency check
            consistency_check = self.consistency_checker.check_entity_consistency(entity)
            
            # Property analysis
            property_analysis = self._analyze_properties(entity)
            
            # Modal analysis
            modal_analysis = self._analyze_modal_properties(entity)
            
            # Trinity analysis (if applicable)
            trinity_analysis = None
            if trinity_role or "trinity_person" in properties:
                trinity_analysis = self._analyze_trinity_entity(entity)
            
            # Scholastic evaluation
            scholastic_evaluation = self._scholastic_entity_evaluation(entity)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                entity, consistency_check, property_analysis
            )
            
            # Determine overall validity
            is_valid = consistency_check.is_consistent and len(consistency_check.violations) == 0
            
            result = OntologicalValidationResult(
                entity_name=entity_name,
                is_valid=is_valid,
                consistency_check=consistency_check,
                property_analysis=property_analysis,
                modal_analysis=modal_analysis,
                trinity_analysis=trinity_analysis,
                scholastic_evaluation=scholastic_evaluation,
                recommendations=recommendations
            )
            
            # Cache result
            self._validation_cache[cache_key] = result
            
            self.logger.debug(f"Ontological validation completed for {entity_name}: {is_valid}")
            return result
            
        except Exception as e:
            self.logger.error(f"Ontological validation failed: {e}")
            
            # Return error result
            return OntologicalValidationResult(
                entity_name=entity_name,
                is_valid=False,
                consistency_check=ConsistencyCheck(
                    is_consistent=False,
                    consistency_level=ConsistencyLevel.INCONSISTENT,
                    violations=[f"Validation error: {str(e)}"],
                    warnings=[],
                    implications=[],
                    theological_analysis={},
                    confidence=0.0
                ),
                property_analysis={},
                modal_analysis={},
                trinity_analysis=None,
                scholastic_evaluation={},
                recommendations=[f"Fix validation error: {str(e)}"]
            )
    
    def _infer_category(self, properties: List[str]) -> OntologicalCategory:
        """Infer ontological category from properties"""
        
        if "trinity_person" in properties:
            return OntologicalCategory.TRINITY_PERSON
        elif "divine" in properties:
            return OntologicalCategory.SUBSTANCE
        elif "rational" in properties:
            return OntologicalCategory.SUBSTANCE
        elif any(prop.endswith("_property") for prop in properties):
            return OntologicalCategory.ATTRIBUTE
        else:
            return OntologicalCategory.PARTICULAR
    
    def _analyze_properties(self, entity: OntologicalEntity) -> Dict[str, Any]:
        """Analyze entity's properties comprehensively"""
        
        analysis = {
            "property_count": len(entity.properties),
            "essential_properties": [],
            "accidental_properties": [],
            "divine_properties": [],
            "modal_properties": [],
            "temporal_properties": [],
            "property_implications": {},
            "missing_implied_properties": []
        }
        
        for prop_name in entity.properties:
            prop = self.property_db.get_property(prop_name)
            if prop:
                # Categorize by type
                if prop.property_type == PropertyType.ESSENTIAL:
                    analysis["essential_properties"].append(prop_name)
                elif prop.property_type == PropertyType.ACCIDENTAL:
                    analysis["accidental_properties"].append(prop_name)
                elif prop.property_type == PropertyType.DIVINE:
                    analysis["divine_properties"].append(prop_name)
                elif prop.property_type == PropertyType.MODAL:
                    analysis["modal_properties"].append(prop_name)
                elif prop.property_type == PropertyType.TEMPORAL:
                    analysis["temporal_properties"].append(prop_name)
                
                # Check implications
                implied = prop.implies_properties - entity.properties
                if implied:
                    analysis["property_implications"][prop_name] = list(implied)
                    analysis["missing_implied_properties"].extend(implied)
        
        return analysis
    
    def _analyze_modal_properties(self, entity: OntologicalEntity) -> Dict[str, Any]:
        """Analyze modal aspects of entity"""
        
        modal_analysis = {
            "existence_mode": entity.existence_mode,
            "modal_status": "contingent",  # default
            "possible_worlds_analysis": {},
            "necessity_assessment": {}
        }
        
        # Determine modal status
        if "necessary" in entity.properties:
            modal_analysis["modal_status"] = "necessary"
        elif "impossible" in entity.properties:
            modal_analysis["modal_status"] = "impossible"
        elif "contingent" in entity.properties:
            modal_analysis["modal_status"] = "contingent"
        
        # Necessity assessment
        essential_props = [prop for prop in entity.properties 
                          if self.property_db.get_property(prop) and 
                          self.property_db.get_property(prop).property_type == PropertyType.ESSENTIAL]
        
        modal_analysis["necessity_assessment"] = {
            "has_essential_properties": len(essential_props) > 0,
            "essential_property_count": len(essential_props),
            "metaphysically_necessary": "divine" in entity.properties or "perfect" in entity.properties
        }
        
        return modal_analysis
    
    def _analyze_trinity_entity(self, entity: OntologicalEntity) -> Dict[str, Any]:
        """Analyze Trinity-specific aspects of entity"""
        
        trinity_analysis = {
            "is_trinity_person": "trinity_person" in entity.properties,
            "trinitarian_relations": [],
            "divine_attributes": [],
            "consubstantial": "consubstantial" in entity.properties,
            "trinitarian_consistency": True
        }
        
        # Check trinitarian relations
        trinity_relations = {"paternal", "filial", "spirated"}
        entity_relations = entity.properties & trinity_relations
        trinity_analysis["trinitarian_relations"] = list(entity_relations)
        
        # Check divine attributes
        divine_attrs = {"omnipotent", "omniscient", "omnipresent", "eternal", "immutable"}
        entity_divine_attrs = entity.properties & divine_attrs
        trinity_analysis["divine_attributes"] = list(entity_divine_attrs)
        
        # Consistency check
        if len(entity_relations) > 1:
            trinity_analysis["trinitarian_consistency"] = False
            trinity_analysis["consistency_error"] = "Cannot have multiple trinitarian relations"
        
        return trinity_analysis
    
    def _scholastic_entity_evaluation(self, entity: OntologicalEntity) -> Dict[str, Any]:
        """Comprehensive scholastic evaluation"""
        
        evaluation = {
            "thomistic_analysis": {},
            "aristotelian_categories": {},
            "divine_attributes_analysis": {},
            "causality_analysis": {}
        }
        
        # Thomistic analysis
        evaluation["thomistic_analysis"] = {
            "essence_existence_composition": "contingent" in entity.properties,
            "act_potency_composition": not ("perfect" in entity.properties),
            "divine_simplicity": "divine" in entity.properties and "simple" in entity.properties
        }
        
        # Aristotelian categories
        evaluation["aristotelian_categories"] = {
            "primary_substance": entity.category == OntologicalCategory.SUBSTANCE,
            "secondary_substance": entity.category == OntologicalCategory.UNIVERSAL,
            "accidents": entity.category == OntologicalCategory.ACCIDENT
        }
        
        return evaluation
    
    def _generate_recommendations(
        self,
        entity: OntologicalEntity,
        consistency_check: ConsistencyCheck,
        property_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for entity improvement"""
        
        recommendations = []
        
        # Address violations
        for violation in consistency_check.violations:
            if "incompatible" in violation.lower():
                recommendations.append(f"Resolve incompatibility: {violation}")
            elif "requires" in violation.lower():
                recommendations.append(f"Add required property: {violation}")
        
        # Address missing implications
        for missing in property_analysis.get("missing_implied_properties", []):
            recommendations.append(f"Consider adding implied property: {missing}")
        
        # Trinity-specific recommendations
        if entity.trinity_role and "consubstantial" not in entity.properties:
            recommendations.append("Add 'consubstantial' property for Trinity person")
        
        # Divine entity recommendations
        if "divine" in entity.properties:
            required_divine_attrs = {"perfect", "eternal", "immutable"}
            missing_divine = required_divine_attrs - entity.properties
            for missing in missing_divine:
                recommendations.append(f"Add divine attribute: {missing}")
        
        return recommendations


# Global ontological validator instance
ontological_validator = OntologicalValidator()


__all__ = [
    'OntologicalCategory',
    'PropertyType',
    'ConsistencyLevel',
    'OntologicalProperty',
    'OntologicalEntity',
    'ConsistencyCheck',
    'OntologicalValidationResult',
    'PropertyDatabase',
    'ConsistencyChecker',
    'OntologicalValidator',
    'ontological_validator'
]