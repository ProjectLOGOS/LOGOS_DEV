"""
Modal Logic Validation Engine - UIP Step 1 Component
===================================================

Advanced modal logic validation and inference engine for UIP pipeline.
Provides comprehensive modal reasoning with world semantics and accessibility relations.

Adapted from: V2_Possible_Gap_Fillers/logos_modal_logic.py
Enhanced with: Kripke model validation, world state management, accessibility analysis
"""

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from protocols.shared.system_imports import *


class ModalOperator(Enum):
    """Modal logic operators"""

    NECESSITY = "□"  # Box (necessarily)
    POSSIBILITY = "◊"  # Diamond (possibly)
    KNOWLEDGE = "K"  # Knowledge operator
    BELIEF = "B"  # Belief operator
    OBLIGATION = "O"  # Deontic obligation
    PERMISSION = "P"  # Deontic permission
    TEMPORAL_ALWAYS = "G"  # Temporal always
    TEMPORAL_EVENTUALLY = "F"  # Temporal eventually


class LogicalConnective(Enum):
    """Standard logical connectives"""

    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"
    BICONDITIONAL = "↔"


@dataclass
class World:
    """Possible world in Kripke model"""

    id: str
    propositions: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)

    def satisfies(self, proposition: str) -> bool:
        """Check if world satisfies proposition"""
        return proposition in self.propositions

    def add_proposition(self, proposition: str) -> None:
        """Add proposition to world"""
        self.propositions.add(proposition)

    def remove_proposition(self, proposition: str) -> None:
        """Remove proposition from world"""
        self.propositions.discard(proposition)


@dataclass
class ModalFormula:
    """Modal logic formula representation"""

    operator: Optional[ModalOperator]
    content: Union[str, "ModalFormula", List["ModalFormula"]]
    connective: Optional[LogicalConnective] = None

    def __str__(self) -> str:
        """String representation of formula"""
        if self.operator:
            if isinstance(self.content, str):
                return f"{self.operator.value}{self.content}"
            else:
                return f"{self.operator.value}({self.content})"
        elif self.connective and isinstance(self.content, list):
            content_strs = [str(c) for c in self.content]
            return f"({self.connective.value.join(content_strs)})"
        else:
            return str(self.content)


@dataclass
class KripkeModel:
    """Kripke model for modal logic evaluation"""

    worlds: Dict[str, World]
    accessibility: Dict[Tuple[str, str], bool]  # (world1, world2) -> accessible
    current_world: str

    def __post_init__(self):
        """Initialize accessibility relation"""
        if not self.accessibility:
            # Default: all worlds accessible to each other
            world_ids = list(self.worlds.keys())
            self.accessibility = {
                (w1, w2): True for w1, w2 in itertools.product(world_ids, repeat=2)
            }

    def is_accessible(self, from_world: str, to_world: str) -> bool:
        """Check if to_world is accessible from from_world"""
        return self.accessibility.get((from_world, to_world), False)

    def accessible_worlds(self, from_world: str) -> List[str]:
        """Get all worlds accessible from given world"""
        return [
            to_world
            for to_world in self.worlds.keys()
            if self.is_accessible(from_world, to_world)
        ]

    def add_world(self, world: World) -> None:
        """Add world to model"""
        self.worlds[world.id] = world

        # Update accessibility relation
        for existing_world in self.worlds.keys():
            if existing_world != world.id:
                self.accessibility[(world.id, existing_world)] = True
                self.accessibility[(existing_world, world.id)] = True

    def set_accessibility(
        self, from_world: str, to_world: str, accessible: bool
    ) -> None:
        """Set accessibility relation between worlds"""
        self.accessibility[(from_world, to_world)] = accessible


@dataclass
class ModalValidationResult:
    """Result of modal logic validation"""

    is_valid: bool
    truth_value: bool
    satisfaction_worlds: List[str]
    countermodel_worlds: List[str]
    reasoning_trace: List[str]
    confidence: float
    modal_properties: Dict[str, Any]


class ModalLogicEngine:
    """Enhanced modal logic validation and inference engine"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Default Kripke model
        self.default_model = self._create_default_model()

        # Validation cache
        self._validation_cache: Dict[str, ModalValidationResult] = {}

        self.logger.info("Modal logic engine initialized")

    def _create_default_model(self) -> KripkeModel:
        """Create default Kripke model with common worlds"""

        # Create standard possible worlds
        actual_world = World("w0", {"actual", "existent"})
        possible_world1 = World("w1", {"possible", "contingent"})
        possible_world2 = World("w2", {"possible", "alternative"})
        necessary_world = World("wn", {"necessary", "universal", "actual", "existent"})

        worlds = {
            "w0": actual_world,
            "w1": possible_world1,
            "w2": possible_world2,
            "wn": necessary_world,
        }

        # Default accessibility: S5 system (all worlds accessible to all)
        accessibility = {}
        world_ids = list(worlds.keys())
        for w1, w2 in itertools.product(world_ids, repeat=2):
            accessibility[(w1, w2)] = True

        return KripkeModel(
            worlds=worlds, accessibility=accessibility, current_world="w0"
        )

    def validate_modal_formula(
        self,
        formula: Union[str, ModalFormula],
        model: Optional[KripkeModel] = None,
        world: Optional[str] = None,
    ) -> ModalValidationResult:
        """
        Validate modal logic formula in given model and world

        Args:
            formula: Modal formula to validate
            model: Kripke model (uses default if None)
            world: Evaluation world (uses current if None)

        Returns:
            ModalValidationResult: Comprehensive validation results
        """
        try:
            # Use default model if none provided
            if model is None:
                model = self.default_model

            # Use current world if none specified
            if world is None:
                world = model.current_world

            # Convert string to ModalFormula if needed
            if isinstance(formula, str):
                formula = self._parse_formula(formula)

            # Check cache
            cache_key = f"{formula}_{model.current_world}_{world}"
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]

            # Evaluate formula
            reasoning_trace = []
            truth_value, satisfaction_worlds = self._evaluate_formula(
                formula, model, world, reasoning_trace
            )

            # Find countermodel worlds
            all_worlds = list(model.worlds.keys())
            countermodel_worlds = [
                w for w in all_worlds if w not in satisfaction_worlds
            ]

            # Calculate confidence based on consistency
            confidence = self._calculate_modal_confidence(
                formula, model, satisfaction_worlds, countermodel_worlds
            )

            # Extract modal properties
            modal_properties = self._analyze_modal_properties(
                formula, model, satisfaction_worlds
            )

            # Determine overall validity
            is_valid = len(satisfaction_worlds) > 0 and truth_value

            result = ModalValidationResult(
                is_valid=is_valid,
                truth_value=truth_value,
                satisfaction_worlds=satisfaction_worlds,
                countermodel_worlds=countermodel_worlds,
                reasoning_trace=reasoning_trace,
                confidence=confidence,
                modal_properties=modal_properties,
            )

            # Cache result
            self._validation_cache[cache_key] = result

            self.logger.debug(
                f"Modal validation completed: {is_valid} (confidence: {confidence:.3f})"
            )
            return result

        except Exception as e:
            self.logger.error(f"Modal validation failed: {e}")
            # Return failure result
            return ModalValidationResult(
                is_valid=False,
                truth_value=False,
                satisfaction_worlds=[],
                countermodel_worlds=list(model.worlds.keys()) if model else [],
                reasoning_trace=[f"Validation error: {str(e)}"],
                confidence=0.0,
                modal_properties={},
            )

    def _parse_formula(self, formula_str: str) -> ModalFormula:
        """Parse string representation into ModalFormula"""
        # Simplified parsing - in production, use proper parser
        formula_str = formula_str.strip()

        # Check for modal operators
        if formula_str.startswith("□"):
            content = formula_str[1:].strip()
            return ModalFormula(ModalOperator.NECESSITY, content)
        elif formula_str.startswith("◊"):
            content = formula_str[1:].strip()
            return ModalFormula(ModalOperator.POSSIBILITY, content)
        elif formula_str.startswith("K"):
            content = formula_str[1:].strip()
            return ModalFormula(ModalOperator.KNOWLEDGE, content)
        elif formula_str.startswith("B"):
            content = formula_str[1:].strip()
            return ModalFormula(ModalOperator.BELIEF, content)
        elif formula_str.startswith("¬"):
            content = formula_str[1:].strip()
            return ModalFormula(None, content, LogicalConnective.NOT)
        else:
            # Atomic proposition
            return ModalFormula(None, formula_str)

    def _evaluate_formula(
        self, formula: ModalFormula, model: KripkeModel, world: str, trace: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate modal formula in given world

        Returns:
            (truth_value, satisfaction_worlds)
        """
        trace.append(f"Evaluating {formula} in world {world}")

        if formula.operator is None:
            # Handle atomic propositions and logical connectives
            if formula.connective == LogicalConnective.NOT:
                # Negation
                inner_result, inner_worlds = self._evaluate_formula(
                    ModalFormula(None, formula.content), model, world, trace
                )
                result = not inner_result
                satisfaction_worlds = [world] if result else []

            elif isinstance(formula.content, str):
                # Atomic proposition
                result = model.worlds[world].satisfies(formula.content)
                satisfaction_worlds = [world] if result else []
                trace.append(f"Proposition '{formula.content}' in {world}: {result}")
            else:
                # Complex formula - simplified handling
                result = True
                satisfaction_worlds = [world]

        elif formula.operator == ModalOperator.NECESSITY:
            # □φ is true iff φ is true in all accessible worlds
            accessible_worlds = model.accessible_worlds(world)
            all_true = True
            satisfaction_worlds = []

            for accessible_world in accessible_worlds:
                inner_result, inner_worlds = self._evaluate_formula(
                    ModalFormula(None, formula.content), model, accessible_world, trace
                )
                if not inner_result:
                    all_true = False
                    break
                satisfaction_worlds.extend(inner_worlds)

            result = all_true
            if not result:
                satisfaction_worlds = []

            trace.append(
                f"Necessity check in {world}: {result} (checked {len(accessible_worlds)} worlds)"
            )

        elif formula.operator == ModalOperator.POSSIBILITY:
            # ◊φ is true iff φ is true in at least one accessible world
            accessible_worlds = model.accessible_worlds(world)
            any_true = False
            satisfaction_worlds = []

            for accessible_world in accessible_worlds:
                inner_result, inner_worlds = self._evaluate_formula(
                    ModalFormula(None, formula.content), model, accessible_world, trace
                )
                if inner_result:
                    any_true = True
                    satisfaction_worlds.extend(inner_worlds)

            result = any_true
            trace.append(
                f"Possibility check in {world}: {result} (checked {len(accessible_worlds)} worlds)"
            )

        elif formula.operator in [ModalOperator.KNOWLEDGE, ModalOperator.BELIEF]:
            # Epistemic operators - similar to necessity but with different accessibility
            accessible_worlds = model.accessible_worlds(world)
            all_true = True
            satisfaction_worlds = []

            for accessible_world in accessible_worlds:
                inner_result, inner_worlds = self._evaluate_formula(
                    ModalFormula(None, formula.content), model, accessible_world, trace
                )
                if not inner_result:
                    all_true = False
                    break
                satisfaction_worlds.extend(inner_worlds)

            result = all_true
            if not result:
                satisfaction_worlds = []

            operator_name = (
                "Knowledge" if formula.operator == ModalOperator.KNOWLEDGE else "Belief"
            )
            trace.append(f"{operator_name} check in {world}: {result}")

        else:
            # Other modal operators - default handling
            result = True
            satisfaction_worlds = [world]
            trace.append(
                f"Default handling for {formula.operator} in {world}: {result}"
            )

        return result, list(set(satisfaction_worlds))  # Remove duplicates

    def _calculate_modal_confidence(
        self,
        formula: ModalFormula,
        model: KripkeModel,
        satisfaction_worlds: List[str],
        countermodel_worlds: List[str],
    ) -> float:
        """Calculate confidence in modal validation result"""

        total_worlds = len(model.worlds)
        if total_worlds == 0:
            return 0.0

        # Base confidence from satisfaction ratio
        satisfaction_ratio = len(satisfaction_worlds) / total_worlds

        # Bonus for modal operators (they have stronger semantics)
        modal_bonus = 0.0
        if formula.operator in [ModalOperator.NECESSITY, ModalOperator.KNOWLEDGE]:
            # Necessity requires universal satisfaction
            modal_bonus = 0.2 if satisfaction_ratio == 1.0 else -0.3
        elif formula.operator == ModalOperator.POSSIBILITY:
            # Possibility only needs existential satisfaction
            modal_bonus = 0.2 if satisfaction_ratio > 0 else -0.2

        # Consistency bonus
        consistency_bonus = (
            0.1 if len(countermodel_worlds) == 0 or len(satisfaction_worlds) == 0 else 0
        )

        confidence = satisfaction_ratio + modal_bonus + consistency_bonus
        return max(0.0, min(1.0, confidence))

    def _analyze_modal_properties(
        self, formula: ModalFormula, model: KripkeModel, satisfaction_worlds: List[str]
    ) -> Dict[str, Any]:
        """Analyze modal properties of validated formula"""

        total_worlds = len(model.worlds)
        satisfied_count = len(satisfaction_worlds)

        properties = {
            "modal_operator": formula.operator.value if formula.operator else None,
            "satisfaction_degree": (
                satisfied_count / total_worlds if total_worlds > 0 else 0
            ),
            "is_tautology": satisfied_count == total_worlds,
            "is_contradiction": satisfied_count == 0,
            "is_contingent": 0 < satisfied_count < total_worlds,
            "accessibility_dependencies": self._analyze_accessibility_deps(
                model, satisfaction_worlds
            ),
            "world_coverage": {
                "satisfied": satisfaction_worlds,
                "total_worlds": total_worlds,
                "coverage_percentage": (
                    (satisfied_count / total_worlds * 100) if total_worlds > 0 else 0
                ),
            },
        }

        # Modal system properties
        if formula.operator:
            properties["modal_system_properties"] = {
                "requires_reflexivity": formula.operator
                in [ModalOperator.KNOWLEDGE, ModalOperator.NECESSITY],
                "requires_transitivity": formula.operator == ModalOperator.KNOWLEDGE,
                "requires_symmetry": formula.operator == ModalOperator.BELIEF,
            }

        return properties

    def _analyze_accessibility_deps(
        self, model: KripkeModel, satisfaction_worlds: List[str]
    ) -> Dict[str, Any]:
        """Analyze accessibility relation dependencies"""

        # Count accessibility relations involving satisfaction worlds
        relevant_relations = 0
        total_relations = 0

        for (w1, w2), accessible in model.accessibility.items():
            total_relations += 1
            if accessible and (w1 in satisfaction_worlds or w2 in satisfaction_worlds):
                relevant_relations += 1

        return {
            "relevant_accessibility_ratio": (
                relevant_relations / total_relations if total_relations > 0 else 0
            ),
            "satisfied_world_connectivity": len(satisfaction_worlds) > 1,
            "isolated_satisfied_worlds": [
                w
                for w in satisfaction_worlds
                if not any(
                    model.is_accessible(w, other)
                    for other in satisfaction_worlds
                    if other != w
                )
            ],
        }

    def create_custom_model(
        self,
        worlds: List[World],
        accessibility_rules: Optional[Dict[Tuple[str, str], bool]] = None,
    ) -> KripkeModel:
        """Create custom Kripke model with specified worlds and accessibility"""

        world_dict = {world.id: world for world in worlds}

        # Default accessibility if none provided
        if accessibility_rules is None:
            world_ids = [world.id for world in worlds]
            accessibility_rules = {
                (w1, w2): True for w1, w2 in itertools.product(world_ids, repeat=2)
            }

        current_world = worlds[0].id if worlds else "w0"

        model = KripkeModel(
            worlds=world_dict,
            accessibility=accessibility_rules,
            current_world=current_world,
        )

        self.logger.info(f"Created custom model with {len(worlds)} worlds")
        return model

    def validate_modal_consistency(
        self, formulas: List[Union[str, ModalFormula]]
    ) -> Dict[str, Any]:
        """Check consistency of multiple modal formulas"""

        results = []
        overall_consistent = True

        for i, formula in enumerate(formulas):
            result = self.validate_modal_formula(formula)
            results.append(result)

            if not result.is_valid:
                overall_consistent = False

        # Check for mutual consistency
        mutual_consistency = self._check_mutual_consistency(results)

        return {
            "overall_consistent": overall_consistent and mutual_consistency,
            "individual_results": results,
            "mutual_consistency": mutual_consistency,
            "consistency_score": (
                sum(r.confidence for r in results) / len(results) if results else 0
            ),
        }

    def _check_mutual_consistency(self, results: List[ModalValidationResult]) -> bool:
        """Check if validation results are mutually consistent"""

        if len(results) <= 1:
            return True

        # Check for contradictory satisfaction patterns
        all_satisfaction_worlds = set()
        for result in results:
            if result.is_valid:
                all_satisfaction_worlds.update(result.satisfaction_worlds)

        # If there are common satisfaction worlds, formulas are potentially consistent
        common_worlds = all_satisfaction_worlds
        for result in results:
            if result.is_valid:
                common_worlds = common_worlds.intersection(
                    set(result.satisfaction_worlds)
                )

        return len(common_worlds) > 0


# Global modal logic engine instance
modal_logic_engine = ModalLogicEngine()


__all__ = [
    "ModalOperator",
    "LogicalConnective",
    "World",
    "ModalFormula",
    "KripkeModel",
    "ModalValidationResult",
    "ModalLogicEngine",
    "modal_logic_engine",
]
