"""
Advanced NLP Processing Engine - UIP Step 1 Component
====================================================

Comprehensive natural language processing with entity extraction, intent classification,
semantic parsing, and theological/philosophical concept recognition.

Enhanced with: Trinity vector semantic mapping, modal logic parsing, theological NLP, ontological entity recognition
"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from system_utillities.shared.system_imports import *


class IntentCategory(Enum):
    """Categories of user intents"""

    THEOLOGICAL_INQUIRY = "theological"
    PHILOSOPHICAL_QUESTION = "philosophical"
    LOGICAL_ANALYSIS = "logical"
    MODAL_REASONING = "modal"
    TRINITY_ANALYSIS = "trinity"
    ONTOLOGICAL_QUERY = "ontological"
    COMPUTATIONAL_REQUEST = "computational"
    DEFINITION_REQUEST = "definition"
    COMPARISON_REQUEST = "comparison"
    VALIDATION_REQUEST = "validation"
    GENERAL_CONVERSATION = "general"


class EntityType(Enum):
    """Types of named entities"""

    THEOLOGICAL_CONCEPT = "theological_concept"
    PHILOSOPHICAL_TERM = "philosophical_term"
    MODAL_OPERATOR = "modal_operator"
    LOGICAL_CONNECTIVE = "logical_connective"
    TRINITY_DIMENSION = "trinity_dimension"
    ONTOLOGICAL_CATEGORY = "ontological_category"
    DIVINE_ATTRIBUTE = "divine_attribute"
    PERSON_NAME = "person_name"
    NUMERICAL_VALUE = "numerical"
    TEMPORAL_REFERENCE = "temporal"


class SemanticRelation(Enum):
    """Semantic relationships between concepts"""

    IMPLIES = "implies"
    CONTRADICTS = "contradicts"
    ENTAILS = "entails"
    PRESUPPOSES = "presupposes"
    EXEMPLIFIES = "exemplifies"
    ANALOGOUS_TO = "analogous_to"
    PART_OF = "part_of"
    INSTANCE_OF = "instance_of"
    CAUSE_OF = "cause_of"
    DEFINED_BY = "defined_by"


@dataclass
class NamedEntity:
    """Extracted named entity"""

    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.text} ({self.entity_type.value})"


@dataclass
class SemanticTriple:
    """Semantic relationship triple"""

    subject: str
    relation: SemanticRelation
    object: str
    confidence: float

    def __str__(self) -> str:
        return f"{self.subject} {self.relation.value} {self.object}"


@dataclass
class IntentClassification:
    """Intent classification result"""

    primary_intent: IntentCategory
    confidence: float
    secondary_intents: List[Tuple[IntentCategory, float]] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class SemanticParse:
    """Semantic parsing result"""

    logical_form: str
    confidence: float
    modal_operators: List[str] = field(default_factory=list)
    quantifiers: List[str] = field(default_factory=list)
    theological_concepts: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.logical_form


@dataclass
class NLPProcessingResult:
    """Comprehensive NLP processing result"""

    original_text: str
    intent_classification: IntentClassification
    named_entities: List[NamedEntity]
    semantic_triples: List[SemanticTriple]
    semantic_parse: Optional[SemanticParse]
    sentiment_analysis: Dict[str, float]
    complexity_metrics: Dict[str, float]
    trinity_vector_mapping: Optional[Tuple[float, float, float]]
    theological_analysis: Dict[str, Any]
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


class TheologicalNLP:
    """Theological and philosophical NLP processing"""

    def __init__(self):
        self.theological_terms = self._initialize_theological_terms()
        self.divine_attributes = self._initialize_divine_attributes()
        self.philosophical_concepts = self._initialize_philosophical_concepts()
        self.trinity_mappings = self._initialize_trinity_mappings()

    def _initialize_theological_terms(self) -> Dict[str, Dict[str, Any]]:
        """Initialize theological terminology database"""
        return {
            # Core theological concepts
            "god": {
                "category": "divine_being",
                "attributes": ["omnipotent", "omniscient", "omnipresent"],
                "trinity_vector": (1.0, 1.0, 1.0),
                "synonyms": ["deity", "divine being", "almighty", "creator"],
            },
            "trinity": {
                "category": "divine_doctrine",
                "attributes": ["triune", "three_persons", "one_essence"],
                "trinity_vector": (1.0, 1.0, 1.0),
                "synonyms": ["triune god", "godhead"],
            },
            "father": {
                "category": "trinity_person",
                "attributes": ["unbegotten", "source", "paternal"],
                "trinity_vector": (1.0, 0.9, 0.9),
                "synonyms": ["first person", "creator"],
            },
            "son": {
                "category": "trinity_person",
                "attributes": ["begotten", "word", "logos"],
                "trinity_vector": (0.9, 1.0, 1.0),
                "synonyms": ["second person", "christ", "jesus", "logos", "word"],
            },
            "holy_spirit": {
                "category": "trinity_person",
                "attributes": ["proceeding", "sanctifier", "spirit"],
                "trinity_vector": (0.9, 1.0, 0.9),
                "synonyms": ["third person", "spirit", "paraclete", "advocate"],
            },
            "incarnation": {
                "category": "theological_doctrine",
                "attributes": ["divine_human", "hypostatic_union"],
                "trinity_vector": (1.0, 0.8, 1.0),
            },
            "salvation": {
                "category": "soteriological",
                "attributes": ["redemption", "grace", "mercy"],
                "trinity_vector": (0.8, 1.0, 0.9),
            },
        }

    def _initialize_divine_attributes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize divine attributes database"""
        return {
            "omnipotent": {
                "definition": "all-powerful",
                "trinity_dimension": "existence",
                "philosophical_analysis": "maximal_power",
            },
            "omniscient": {
                "definition": "all-knowing",
                "trinity_dimension": "truth",
                "philosophical_analysis": "maximal_knowledge",
            },
            "omnipresent": {
                "definition": "present everywhere",
                "trinity_dimension": "existence",
                "philosophical_analysis": "spatial_transcendence",
            },
            "perfect": {
                "definition": "lacking no excellence",
                "trinity_dimension": "goodness",
                "philosophical_analysis": "maximal_excellence",
            },
            "eternal": {
                "definition": "outside of time",
                "trinity_dimension": "existence",
                "philosophical_analysis": "temporal_transcendence",
            },
            "immutable": {
                "definition": "unchanging",
                "trinity_dimension": "truth",
                "philosophical_analysis": "metaphysical_stability",
            },
        }

    def _initialize_philosophical_concepts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize philosophical concepts database"""
        return {
            "substance": {
                "tradition": "aristotelian",
                "category": "metaphysics",
                "definition": "that which exists in itself",
            },
            "essence": {
                "tradition": "scholastic",
                "category": "metaphysics",
                "definition": "what something is",
            },
            "existence": {
                "tradition": "thomistic",
                "category": "metaphysics",
                "definition": "that something is",
            },
            "necessary": {
                "tradition": "modal_logic",
                "category": "logic",
                "definition": "true in all possible worlds",
            },
            "contingent": {
                "tradition": "modal_logic",
                "category": "logic",
                "definition": "true in some but not all possible worlds",
            },
        }

    def _initialize_trinity_mappings(self) -> Dict[str, Tuple[float, float, float]]:
        """Initialize Trinity vector mappings for concepts"""
        return {
            "creation": (0.9, 0.8, 0.7),
            "redemption": (0.7, 1.0, 0.9),
            "sanctification": (0.8, 0.9, 1.0),
            "revelation": (0.6, 0.8, 1.0),
            "providence": (0.9, 0.9, 0.8),
            "judgment": (0.8, 1.0, 1.0),
            "mercy": (0.7, 1.0, 0.8),
            "justice": (0.9, 1.0, 1.0),
        }

    def extract_theological_concepts(self, text: str) -> List[NamedEntity]:
        """Extract theological concepts from text"""
        entities = []
        text_lower = text.lower()

        # Check for theological terms
        for term, info in self.theological_terms.items():
            # Direct match
            if term in text_lower:
                start_pos = text_lower.find(term)
                entities.append(
                    NamedEntity(
                        text=term,
                        entity_type=EntityType.THEOLOGICAL_CONCEPT,
                        start_pos=start_pos,
                        end_pos=start_pos + len(term),
                        confidence=0.9,
                        attributes=info,
                    )
                )

            # Synonym matches
            for synonym in info.get("synonyms", []):
                if synonym in text_lower:
                    start_pos = text_lower.find(synonym)
                    entities.append(
                        NamedEntity(
                            text=synonym,
                            entity_type=EntityType.THEOLOGICAL_CONCEPT,
                            start_pos=start_pos,
                            end_pos=start_pos + len(synonym),
                            confidence=0.8,
                            attributes=info,
                        )
                    )

        # Check for divine attributes
        for attr, info in self.divine_attributes.items():
            if attr in text_lower:
                start_pos = text_lower.find(attr)
                entities.append(
                    NamedEntity(
                        text=attr,
                        entity_type=EntityType.DIVINE_ATTRIBUTE,
                        start_pos=start_pos,
                        end_pos=start_pos + len(attr),
                        confidence=0.85,
                        attributes=info,
                    )
                )

        return entities

    def map_to_trinity_vector(self, text: str) -> Optional[Tuple[float, float, float]]:
        """Map text concepts to Trinity vector representation"""

        # Check direct mappings
        text_lower = text.lower()
        for concept, vector in self.trinity_mappings.items():
            if concept in text_lower:
                return vector

        # Check theological terms
        for term, info in self.theological_terms.items():
            if term in text_lower and "trinity_vector" in info:
                return info["trinity_vector"]

        # Analyze semantic content for implicit Trinity mapping
        existence_indicators = ["exists", "being", "reality", "actual", "present"]
        goodness_indicators = ["good", "perfect", "beautiful", "love", "mercy", "kind"]
        truth_indicators = [
            "true",
            "truth",
            "knowledge",
            "wisdom",
            "revelation",
            "logos",
        ]

        e_score = sum(
            1 for indicator in existence_indicators if indicator in text_lower
        )
        g_score = sum(1 for indicator in goodness_indicators if indicator in text_lower)
        t_score = sum(1 for indicator in truth_indicators if indicator in text_lower)

        if e_score + g_score + t_score > 0:
            total = e_score + g_score + t_score
            return (e_score / total, g_score / total, t_score / total)

        return None


class IntentClassifier:
    """Intent classification engine"""

    def __init__(self):
        self.intent_patterns = self._initialize_intent_patterns()
        self.intent_keywords = self._initialize_intent_keywords()

    def _initialize_intent_patterns(self) -> Dict[IntentCategory, List[str]]:
        """Initialize regex patterns for intent recognition"""
        return {
            IntentCategory.THEOLOGICAL_INQUIRY: [
                r"\b(what|who|how) is (god|trinity|father|son|spirit)\b",
                r"\b(divine|theological|doctrine|faith|belief)\b",
                r"\b(salvation|redemption|incarnation|atonement)\b",
            ],
            IntentCategory.PHILOSOPHICAL_QUESTION: [
                r"\b(metaphysics|ontology|epistemology|ethics)\b",
                r"\b(substance|essence|existence|being)\b",
                r"\b(what is the nature of|philosophical|philosophy)\b",
            ],
            IntentCategory.MODAL_REASONING: [
                r"\b(necessarily|possibly|must|might|could)\b",
                r"\b(necessary|contingent|possible|impossible)\b",
                r"\b(modal logic|possible world)\b",
            ],
            IntentCategory.LOGICAL_ANALYSIS: [
                r"\b(if.*then|implies|entails|follows)\b",
                r"\b(valid|sound|argument|premise|conclusion)\b",
                r"\b(logic|logical|reasoning)\b",
            ],
            IntentCategory.TRINITY_ANALYSIS: [
                r"\b(trinity|triune|three persons|consubstantial)\b",
                r"\b(father.*son.*spirit|divine persons)\b",
                r"\b(perichoresis|circumincession|hypostasis)\b",
            ],
            IntentCategory.DEFINITION_REQUEST: [
                r"\b(what (is|are|does)|define|definition|meaning)\b",
                r"\b(explain|describe|tell me about)\b",
            ],
            IntentCategory.COMPARISON_REQUEST: [
                r"\b(compare|difference|similar|versus|vs)\b",
                r"\b(better|worse|same|different)\b",
            ],
            IntentCategory.VALIDATION_REQUEST: [
                r"\b(is.*true|verify|check|validate|correct)\b",
                r"\b(consistent|contradictory|valid)\b",
            ],
        }

    def _initialize_intent_keywords(self) -> Dict[IntentCategory, List[str]]:
        """Initialize keyword lists for intent classification"""
        return {
            IntentCategory.THEOLOGICAL_INQUIRY: [
                "god",
                "divine",
                "trinity",
                "father",
                "son",
                "spirit",
                "salvation",
                "church",
                "doctrine",
                "faith",
                "belief",
                "prayer",
                "worship",
            ],
            IntentCategory.PHILOSOPHICAL_QUESTION: [
                "philosophy",
                "metaphysics",
                "ontology",
                "epistemology",
                "ethics",
                "substance",
                "essence",
                "existence",
                "being",
                "reality",
                "truth",
            ],
            IntentCategory.MODAL_REASONING: [
                "necessary",
                "contingent",
                "possible",
                "impossible",
                "modal",
                "necessarily",
                "possibly",
                "must",
                "might",
                "could",
            ],
            IntentCategory.LOGICAL_ANALYSIS: [
                "logic",
                "argument",
                "premise",
                "conclusion",
                "valid",
                "sound",
                "implies",
                "entails",
                "follows",
                "reasoning",
            ],
            IntentCategory.TRINITY_ANALYSIS: [
                "trinity",
                "triune",
                "consubstantial",
                "perichoresis",
                "hypostasis",
                "three persons",
                "divine persons",
                "godhead",
            ],
        }

    def classify_intent(self, text: str) -> IntentClassification:
        """Classify user intent from text"""
        text_lower = text.lower()

        # Pattern matching scores
        pattern_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            pattern_scores[intent] = score

        # Keyword matching scores
        keyword_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            keyword_scores[intent] = score

        # Combine scores
        combined_scores = {}
        all_intents = set(pattern_scores.keys()) | set(keyword_scores.keys())

        for intent in all_intents:
            pattern_score = pattern_scores.get(intent, 0)
            keyword_score = keyword_scores.get(intent, 0)

            # Weight pattern matches higher than keyword matches
            combined_scores[intent] = pattern_score * 2 + keyword_score

        # Find primary intent
        if combined_scores:
            primary_intent = max(
                combined_scores.keys(), key=lambda x: combined_scores[x]
            )
            max_score = combined_scores[primary_intent]

            # Calculate confidence
            total_score = sum(combined_scores.values())
            confidence = max_score / max(total_score, 1)

            # Find secondary intents
            secondary_intents = [
                (intent, score / max(total_score, 1))
                for intent, score in combined_scores.items()
                if intent != primary_intent and score > 0
            ]
            secondary_intents.sort(key=lambda x: x[1], reverse=True)

            # Generate reasoning
            reasoning = (
                f"Pattern matches: {pattern_scores.get(primary_intent, 0)}, "
                f"Keyword matches: {keyword_scores.get(primary_intent, 0)}"
            )

            return IntentClassification(
                primary_intent=primary_intent,
                confidence=confidence,
                secondary_intents=secondary_intents[:3],  # Top 3
                reasoning=reasoning,
            )

        # Default to general conversation
        return IntentClassification(
            primary_intent=IntentCategory.GENERAL_CONVERSATION,
            confidence=0.5,
            reasoning="No specific patterns detected",
        )


class EntityExtractor:
    """Named entity extraction engine"""

    def __init__(self, theological_nlp: TheologicalNLP):
        self.theological_nlp = theological_nlp
        self.entity_patterns = self._initialize_entity_patterns()

    def _initialize_entity_patterns(self) -> Dict[EntityType, List[str]]:
        """Initialize entity extraction patterns"""
        return {
            EntityType.MODAL_OPERATOR: [
                r"\b(necessarily|possibly|must|might|could|may)\b"
            ],
            EntityType.LOGICAL_CONNECTIVE: [
                r"\b(and|or|not|if|then|implies|entails|because|therefore)\b"
            ],
            EntityType.TRINITY_DIMENSION: [
                r"\b(existence|goodness|truth|being|good|true)\b"
            ],
            EntityType.ONTOLOGICAL_CATEGORY: [
                r"\b(substance|attribute|relation|mode|essence|accident)\b"
            ],
            EntityType.NUMERICAL_VALUE: [r"\b(\d+(?:\.\d+)?)\b"],
            EntityType.TEMPORAL_REFERENCE: [
                r"\b(eternal|temporal|always|never|sometimes|now|then)\b"
            ],
        }

    def extract_entities(self, text: str) -> List[NamedEntity]:
        """Extract all named entities from text"""
        entities = []

        # Extract theological concepts
        theological_entities = self.theological_nlp.extract_theological_concepts(text)
        entities.extend(theological_entities)

        # Extract other entity types
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Check if already covered by theological entities
                    overlapping = any(
                        e.start_pos <= match.start() < e.end_pos
                        or match.start() <= e.start_pos < match.end()
                        for e in entities
                    )

                    if not overlapping:
                        entities.append(
                            NamedEntity(
                                text=match.group(),
                                entity_type=entity_type,
                                start_pos=match.start(),
                                end_pos=match.end(),
                                confidence=0.8,
                            )
                        )

        # Sort by position
        entities.sort(key=lambda e: e.start_pos)
        return entities


class SemanticParser:
    """Semantic parsing and logical form generation"""

    def __init__(self):
        self.modal_operators = {
            "necessarily": "□",
            "possibly": "◊",
            "must": "□",
            "might": "◊",
        }
        self.quantifiers = {"all": "∀", "every": "∀", "some": "∃", "there exists": "∃"}
        self.logical_connectives = {"and": "∧", "or": "∨", "not": "¬", "implies": "→"}

    def parse_to_logical_form(
        self, text: str, entities: List[NamedEntity]
    ) -> Optional[SemanticParse]:
        """Parse text into logical form representation"""

        try:
            logical_form = text.lower()
            modal_ops = []
            quantifs = []
            theological_concepts = []

            # Replace modal operators
            for word, symbol in self.modal_operators.items():
                if word in logical_form:
                    logical_form = logical_form.replace(word, symbol)
                    modal_ops.append(symbol)

            # Replace quantifiers
            for word, symbol in self.quantifiers.items():
                if word in logical_form:
                    logical_form = logical_form.replace(word, symbol)
                    quantifs.append(symbol)

            # Replace logical connectives
            for word, symbol in self.logical_connectives.items():
                if word in logical_form:
                    logical_form = logical_form.replace(word, symbol)

            # Extract theological concepts
            for entity in entities:
                if entity.entity_type == EntityType.THEOLOGICAL_CONCEPT:
                    theological_concepts.append(entity.text)

            # Simple confidence calculation
            complexity_indicators = (
                len(modal_ops) + len(quantifs) + len(theological_concepts)
            )
            confidence = min(0.9, 0.5 + complexity_indicators * 0.1)

            return SemanticParse(
                logical_form=logical_form,
                confidence=confidence,
                modal_operators=modal_ops,
                quantifiers=quantifs,
                theological_concepts=theological_concepts,
            )

        except Exception:
            return None


class AdvancedNLPEngine:
    """Main advanced NLP processing engine"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.theological_nlp = TheologicalNLP()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor(self.theological_nlp)
        self.semantic_parser = SemanticParser()

        # Processing cache
        self._processing_cache: Dict[str, NLPProcessingResult] = {}

        self.logger.info("Advanced NLP engine initialized")

    def process_text(self, text: str) -> NLPProcessingResult:
        """
        Comprehensive NLP processing of input text

        Args:
            text: Input text to process

        Returns:
            NLPProcessingResult: Comprehensive processing results
        """
        try:
            # Check cache
            if text in self._processing_cache:
                return self._processing_cache[text]

            # Intent classification
            intent_classification = self.intent_classifier.classify_intent(text)

            # Named entity extraction
            named_entities = self.entity_extractor.extract_entities(text)

            # Semantic parsing
            semantic_parse = self.semantic_parser.parse_to_logical_form(
                text, named_entities
            )

            # Semantic triple extraction
            semantic_triples = self._extract_semantic_triples(text, named_entities)

            # Sentiment analysis
            sentiment_analysis = self._analyze_sentiment(text)

            # Complexity metrics
            complexity_metrics = self._calculate_complexity_metrics(
                text, named_entities
            )

            # Trinity vector mapping
            trinity_vector = self.theological_nlp.map_to_trinity_vector(text)

            # Theological analysis
            theological_analysis = self._perform_theological_analysis(
                text, named_entities
            )

            # Processing metadata
            processing_metadata = {
                "text_length": len(text),
                "entity_count": len(named_entities),
                "processing_timestamp": time.time(),
                "confidence_overall": self._calculate_overall_confidence(
                    intent_classification, named_entities, semantic_parse
                ),
            }

            result = NLPProcessingResult(
                original_text=text,
                intent_classification=intent_classification,
                named_entities=named_entities,
                semantic_triples=semantic_triples,
                semantic_parse=semantic_parse,
                sentiment_analysis=sentiment_analysis,
                complexity_metrics=complexity_metrics,
                trinity_vector_mapping=trinity_vector,
                theological_analysis=theological_analysis,
                processing_metadata=processing_metadata,
            )

            # Cache result
            self._processing_cache[text] = result

            self.logger.debug(
                f"NLP processing completed: {intent_classification.primary_intent.value}"
            )
            return result

        except Exception as e:
            self.logger.error(f"NLP processing failed: {e}")

            # Return minimal result
            return NLPProcessingResult(
                original_text=text,
                intent_classification=IntentClassification(
                    IntentCategory.GENERAL_CONVERSATION, 0.0, [], f"Error: {str(e)}"
                ),
                named_entities=[],
                semantic_triples=[],
                semantic_parse=None,
                sentiment_analysis={},
                complexity_metrics={},
                trinity_vector_mapping=None,
                theological_analysis={},
                processing_metadata={"error": str(e)},
            )

    def _extract_semantic_triples(
        self, text: str, entities: List[NamedEntity]
    ) -> List[SemanticTriple]:
        """Extract semantic relationship triples"""
        triples = []
        text_lower = text.lower()

        # Simple pattern-based triple extraction
        implication_patterns = [
            r"(\w+) implies (\w+)",
            r"if (\w+) then (\w+)",
            r"(\w+) entails (\w+)",
        ]

        for pattern in implication_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                triples.append(
                    SemanticTriple(
                        subject=match.group(1),
                        relation=SemanticRelation.IMPLIES,
                        object=match.group(2),
                        confidence=0.7,
                    )
                )

        # Entity-based relationships
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i != j:
                    # Check for definitional relationships
                    if (
                        "is" in text_lower
                        and entity1.text in text_lower
                        and entity2.text in text_lower
                    ):
                        # Simple heuristic for definitional relationships
                        triples.append(
                            SemanticTriple(
                                subject=entity1.text,
                                relation=SemanticRelation.DEFINED_BY,
                                object=entity2.text,
                                confidence=0.6,
                            )
                        )

        return triples

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment/emotional content of text"""

        positive_words = [
            "good",
            "perfect",
            "beautiful",
            "love",
            "mercy",
            "grace",
            "joy",
            "peace",
            "hope",
            "salvation",
            "blessed",
            "holy",
            "sacred",
        ]

        negative_words = [
            "evil",
            "sin",
            "suffering",
            "pain",
            "hell",
            "damnation",
            "wrath",
            "judgment",
            "punishment",
            "death",
            "corruption",
            "fallen",
        ]

        neutral_words = [
            "truth",
            "knowledge",
            "being",
            "existence",
            "essence",
            "substance",
            "logic",
            "reason",
            "analysis",
            "study",
            "doctrine",
            "theology",
        ]

        text_lower = text.lower()

        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        neutral_score = sum(1 for word in neutral_words if word in text_lower)

        total_score = positive_score + negative_score + neutral_score

        if total_score > 0:
            return {
                "positive": positive_score / total_score,
                "negative": negative_score / total_score,
                "neutral": neutral_score / total_score,
            }
        else:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    def _calculate_complexity_metrics(
        self, text: str, entities: List[NamedEntity]
    ) -> Dict[str, float]:
        """Calculate text complexity metrics"""

        words = text.split()
        sentences = re.split(r"[.!?]+", text)

        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": sum(len(word) for word in words) / max(len(words), 1),
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "entity_density": len(entities) / max(len(words), 1),
            "theological_complexity": self._calculate_theological_complexity(entities),
            "modal_complexity": self._calculate_modal_complexity(text),
            "logical_complexity": self._calculate_logical_complexity(text),
        }

    def _calculate_theological_complexity(self, entities: List[NamedEntity]) -> float:
        """Calculate theological complexity based on entities"""
        theological_entities = [
            e
            for e in entities
            if e.entity_type
            in [EntityType.THEOLOGICAL_CONCEPT, EntityType.DIVINE_ATTRIBUTE]
        ]

        # More theological entities = higher complexity
        base_complexity = len(theological_entities) / 10.0  # Normalize

        # Trinity-related concepts add complexity
        trinity_entities = [
            e for e in theological_entities if "trinity" in e.text.lower()
        ]
        trinity_bonus = len(trinity_entities) * 0.2

        return min(1.0, base_complexity + trinity_bonus)

    def _calculate_modal_complexity(self, text: str) -> float:
        """Calculate modal logic complexity"""
        modal_indicators = ["necessarily", "possibly", "must", "might", "could", "may"]
        text_lower = text.lower()

        modal_count = sum(
            1 for indicator in modal_indicators if indicator in text_lower
        )
        return min(1.0, modal_count / 5.0)  # Normalize to [0,1]

    def _calculate_logical_complexity(self, text: str) -> float:
        """Calculate logical complexity"""
        logical_indicators = [
            "if",
            "then",
            "implies",
            "entails",
            "therefore",
            "because",
            "and",
            "or",
            "not",
        ]
        text_lower = text.lower()

        logical_count = sum(
            1 for indicator in logical_indicators if indicator in text_lower
        )
        return min(1.0, logical_count / 8.0)  # Normalize to [0,1]

    def _perform_theological_analysis(
        self, text: str, entities: List[NamedEntity]
    ) -> Dict[str, Any]:
        """Perform comprehensive theological analysis"""

        analysis = {
            "theological_domain": "general",
            "primary_concepts": [],
            "divine_attributes_mentioned": [],
            "trinity_references": [],
            "doctrinal_category": None,
            "scholastic_elements": [],
            "modal_theological_claims": [],
        }

        # Identify primary theological concepts
        theological_entities = [
            e for e in entities if e.entity_type == EntityType.THEOLOGICAL_CONCEPT
        ]
        analysis["primary_concepts"] = [e.text for e in theological_entities]

        # Identify divine attributes
        divine_entities = [
            e for e in entities if e.entity_type == EntityType.DIVINE_ATTRIBUTE
        ]
        analysis["divine_attributes_mentioned"] = [e.text for e in divine_entities]

        # Check for Trinity references
        trinity_terms = ["trinity", "father", "son", "spirit", "triune", "persons"]
        text_lower = text.lower()
        analysis["trinity_references"] = [
            term for term in trinity_terms if term in text_lower
        ]

        # Determine doctrinal category
        if analysis["trinity_references"]:
            analysis["doctrinal_category"] = "trinitarian"
        elif analysis["divine_attributes_mentioned"]:
            analysis["doctrinal_category"] = "theological_attributes"
        elif "salvation" in text_lower or "redemption" in text_lower:
            analysis["doctrinal_category"] = "soteriological"
        elif "creation" in text_lower or "creator" in text_lower:
            analysis["doctrinal_category"] = "cosmological"

        # Check for scholastic elements
        scholastic_terms = [
            "substance",
            "essence",
            "existence",
            "accident",
            "form",
            "matter",
        ]
        analysis["scholastic_elements"] = [
            term for term in scholastic_terms if term in text_lower
        ]

        return analysis

    def _calculate_overall_confidence(
        self,
        intent_classification: IntentClassification,
        entities: List[NamedEntity],
        semantic_parse: Optional[SemanticParse],
    ) -> float:
        """Calculate overall processing confidence"""

        # Intent classification confidence
        intent_confidence = intent_classification.confidence

        # Entity extraction confidence (average of entity confidences)
        if entities:
            entity_confidence = sum(e.confidence for e in entities) / len(entities)
        else:
            entity_confidence = 0.5

        # Semantic parsing confidence
        parse_confidence = semantic_parse.confidence if semantic_parse else 0.3

        # Weighted average
        overall_confidence = (
            intent_confidence * 0.4 + entity_confidence * 0.3 + parse_confidence * 0.3
        )

        return overall_confidence


# Global NLP engine instance
advanced_nlp_engine = AdvancedNLPEngine()


__all__ = [
    "IntentCategory",
    "EntityType",
    "SemanticRelation",
    "NamedEntity",
    "SemanticTriple",
    "IntentClassification",
    "SemanticParse",
    "NLPProcessingResult",
    "TheologicalNLP",
    "IntentClassifier",
    "EntityExtractor",
    "SemanticParser",
    "AdvancedNLPEngine",
    "advanced_nlp_engine",
]
