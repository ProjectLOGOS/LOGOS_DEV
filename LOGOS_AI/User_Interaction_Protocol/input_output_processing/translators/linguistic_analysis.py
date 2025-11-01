"""
Linguistic Analysis - UIP Step 1
=================================

Disambiguates meaning, infers user intent, detects malformed input.
Core linguistic processing that transforms raw user input into structured
semantic representations for downstream processing.
"""

from system_utillities.shared.system_imports import *
from system_utillities.user_interaction.uip_registry import (
    UIPContext,
    UIPStep,
    register_uip_handler,
)


@dataclass
class LinguisticAnalysisResult:
    """Result structure for linguistic analysis"""

    intent_classification: Dict[str, float]  # intent -> confidence
    entity_extraction: List[Dict[str, Any]]  # extracted entities
    semantic_representation: Dict[str, Any]  # semantic structure
    confidence: float  # overall analysis confidence
    language_detected: str
    malformed_input_flags: List[str]  # detected issues
    clarification_needed: bool
    disambiguation_candidates: List[Dict[str, Any]]
    contextual_metadata: Dict[str, Any]


class LinguisticAnalyzer:
    """Core linguistic analysis engine"""

    def __init__(self):
        """Initialize linguistic analyzer"""
        self.logger = logging.getLogger(__name__)
        self.intent_classifier = (
            None  # Will be loaded from interfaces/natural_language/
        )
        self.entity_extractor = None  # Will be loaded from interfaces/natural_language/
        self.semantic_inferencer = (
            None  # Will be loaded from intelligence/reasoning_engines/
        )
        self.initialize_components()

    def initialize_components(self):
        """Initialize linguistic processing components"""
        try:
            # These will import the actual implementations once they exist
            # from interfaces.natural_language.intent_classifier import IntentClassifier
            # from interfaces.natural_language.entity_extractor import EntityExtractor
            # from intelligence.reasoning_engines.semantic_inferencer import SemanticInferencer

            # Placeholder initialization
            self.logger.info(
                "Linguistic analyzer components initialized (placeholder mode)"
            )

        except ImportError as e:
            self.logger.warning(f"Some linguistic components not available: {e}")

    async def analyze(self, context: UIPContext) -> LinguisticAnalysisResult:
        """
        Perform complete linguistic analysis on user input

        Args:
            context: UIP processing context

        Returns:
            LinguisticAnalysisResult: Structured analysis results
        """
        user_input = context.user_input
        metadata = context.metadata

        self.logger.info(f"Starting linguistic analysis for {context.correlation_id}")

        try:
            # Step 1.1: Language detection
            detected_language = await self._detect_language(user_input, metadata)

            # Step 1.2: Input validation and malformation detection
            malformed_flags = await self._check_malformed_input(user_input)

            # Step 1.3: Intent classification
            intent_classification = await self._classify_intent(
                user_input, detected_language
            )

            # Step 1.4: Entity extraction
            entities = await self._extract_entities(user_input, detected_language)

            # Step 1.5: Semantic representation generation
            semantic_repr = await self._generate_semantic_representation(
                user_input, intent_classification, entities
            )

            # Step 1.6: Disambiguation analysis
            disambiguation_candidates = await self._analyze_disambiguation(
                user_input, intent_classification, entities
            )

            # Step 1.7: Confidence calculation
            overall_confidence = self._calculate_overall_confidence(
                intent_classification, entities, semantic_repr, malformed_flags
            )

            # Step 1.8: Clarification need assessment
            needs_clarification = self._assess_clarification_need(
                overall_confidence, disambiguation_candidates, malformed_flags
            )

            # Create result
            result = LinguisticAnalysisResult(
                intent_classification=intent_classification,
                entity_extraction=entities,
                semantic_representation=semantic_repr,
                confidence=overall_confidence,
                language_detected=detected_language,
                malformed_input_flags=malformed_flags,
                clarification_needed=needs_clarification,
                disambiguation_candidates=disambiguation_candidates,
                contextual_metadata={
                    "input_length": len(user_input),
                    "analysis_timestamp": time.time(),
                    "processing_components": [
                        "intent",
                        "entity",
                        "semantic",
                        "disambiguation",
                    ],
                },
            )

            self.logger.info(
                f"Linguistic analysis completed for {context.correlation_id} "
                f"(confidence: {overall_confidence:.3f})"
            )

            return result

        except Exception as e:
            self.logger.error(
                f"Linguistic analysis failed for {context.correlation_id}: {e}"
            )
            raise

    async def _detect_language(self, user_input: str, metadata: Dict) -> str:
        """Detect input language"""
        # Check if language is specified in metadata
        if "language" in metadata and metadata["language"]:
            return metadata["language"]

        # Placeholder language detection logic
        # In real implementation, would use langdetect or similar
        if any(char for char in user_input if ord(char) > 127):
            # Contains non-ASCII characters, might be non-English
            return "unknown"
        else:
            return "en"  # Default to English

    async def _check_malformed_input(self, user_input: str) -> List[str]:
        """Check for malformed input patterns"""
        flags = []

        # Check for empty or whitespace-only input
        if not user_input.strip():
            flags.append("empty_input")

        # Check for excessive length
        if len(user_input) > 10000:
            flags.append("excessive_length")

        # Check for potential injection patterns
        suspicious_patterns = ["<script>", "javascript:", "eval(", "exec("]
        if any(pattern in user_input.lower() for pattern in suspicious_patterns):
            flags.append("potential_injection")

        # Check for excessive repetition
        words = user_input.split()
        if len(words) > 5:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:
                flags.append("excessive_repetition")

        # Check for binary or encoded content
        if any(
            ord(char) < 32 and char not in ["\n", "\r", "\t"] for char in user_input
        ):
            flags.append("binary_content")

        return flags

    async def _classify_intent(
        self, user_input: str, language: str
    ) -> Dict[str, float]:
        """Classify user intent"""
        # Placeholder intent classification
        # In real implementation, would use trained models

        # Simple keyword-based classification for now
        intent_scores = {
            "question": 0.0,
            "request": 0.0,
            "command": 0.0,
            "conversation": 0.0,
            "information_seeking": 0.0,
            "problem_solving": 0.0,
            "creative_task": 0.0,
            "analysis_task": 0.0,
        }

        lower_input = user_input.lower()

        # Question patterns
        if any(
            word in lower_input
            for word in ["what", "why", "how", "when", "where", "who", "?"]
        ):
            intent_scores["question"] = 0.8
            intent_scores["information_seeking"] = 0.6

        # Request patterns
        if any(
            word in lower_input
            for word in ["please", "can you", "could you", "help me"]
        ):
            intent_scores["request"] = 0.7

        # Command patterns
        if any(
            word in lower_input
            for word in ["do", "create", "make", "generate", "build"]
        ):
            intent_scores["command"] = 0.6
            intent_scores["creative_task"] = 0.5

        # Analysis patterns
        if any(
            word in lower_input for word in ["analyze", "compare", "evaluate", "assess"]
        ):
            intent_scores["analysis_task"] = 0.7

        # Conversation patterns
        if any(word in lower_input for word in ["hello", "hi", "thanks", "goodbye"]):
            intent_scores["conversation"] = 0.8

        # Normalize scores so highest is primary intent
        max_score = max(intent_scores.values()) if intent_scores.values() else 0.1
        if max_score < 0.3:
            intent_scores["information_seeking"] = 0.5  # Default fallback

        return intent_scores

    async def _extract_entities(
        self, user_input: str, language: str
    ) -> List[Dict[str, Any]]:
        """Extract named entities from input"""
        # Placeholder entity extraction
        entities = []

        # Simple pattern-based extraction for now
        words = user_input.split()

        for i, word in enumerate(words):
            # Capitalized words might be proper nouns
            if word[0].isupper() and len(word) > 2:
                entities.append(
                    {
                        "text": word,
                        "label": "PROPER_NOUN",
                        "start": user_input.find(word),
                        "end": user_input.find(word) + len(word),
                        "confidence": 0.6,
                    }
                )

        return entities

    async def _generate_semantic_representation(
        self,
        user_input: str,
        intent_classification: Dict[str, float],
        entities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate semantic representation of the input"""

        primary_intent = max(intent_classification.items(), key=lambda x: x[1])

        semantic_repr = {
            "primary_intent": primary_intent[0],
            "intent_confidence": primary_intent[1],
            "secondary_intents": [
                intent
                for intent, score in intent_classification.items()
                if score > 0.3 and intent != primary_intent[0]
            ],
            "key_entities": [
                entity for entity in entities if entity["confidence"] > 0.5
            ],
            "semantic_features": {
                "complexity": len(user_input.split())
                / 10.0,  # Rough complexity measure
                "formality": 0.5,  # Placeholder formality score
                "specificity": len(entities) / max(len(user_input.split()), 1),
            },
            "discourse_markers": self._extract_discourse_markers(user_input),
        }

        return semantic_repr

    def _extract_discourse_markers(self, user_input: str) -> List[str]:
        """Extract discourse markers that indicate structure"""
        markers = []
        lower_input = user_input.lower()

        discourse_patterns = [
            "first",
            "second",
            "third",
            "finally",
            "however",
            "therefore",
            "because",
            "although",
            "moreover",
            "furthermore",
            "in conclusion",
        ]

        for pattern in discourse_patterns:
            if pattern in lower_input:
                markers.append(pattern)

        return markers

    async def _analyze_disambiguation(
        self,
        user_input: str,
        intent_classification: Dict[str, float],
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Analyze potential ambiguities requiring disambiguation"""

        candidates = []

        # Check for multiple high-confidence intents
        high_confidence_intents = [
            intent for intent, score in intent_classification.items() if score > 0.6
        ]

        if len(high_confidence_intents) > 1:
            candidates.append(
                {
                    "type": "intent_ambiguity",
                    "description": f"Multiple possible intents: {', '.join(high_confidence_intents)}",
                    "candidates": high_confidence_intents,
                    "requires_clarification": True,
                }
            )

        # Check for ambiguous pronouns
        pronouns = ["it", "this", "that", "they", "them"]
        found_pronouns = [p for p in pronouns if p in user_input.lower().split()]

        if found_pronouns and len(entities) > 1:
            candidates.append(
                {
                    "type": "pronoun_reference",
                    "description": f"Ambiguous pronoun reference: {', '.join(found_pronouns)}",
                    "candidates": [entity["text"] for entity in entities],
                    "requires_clarification": True,
                }
            )

        return candidates

    def _calculate_overall_confidence(
        self,
        intent_classification: Dict[str, float],
        entities: List[Dict[str, Any]],
        semantic_repr: Dict[str, Any],
        malformed_flags: List[str],
    ) -> float:
        """Calculate overall confidence in linguistic analysis"""

        # Base confidence from intent classification
        max_intent_confidence = (
            max(intent_classification.values()) if intent_classification else 0.0
        )

        # Penalty for malformed input
        malformed_penalty = len(malformed_flags) * 0.2

        # Bonus for successful entity extraction
        entity_bonus = min(len(entities) * 0.1, 0.3)

        # Calculate final confidence
        confidence = max_intent_confidence + entity_bonus - malformed_penalty
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]

        return confidence

    def _assess_clarification_need(
        self,
        confidence: float,
        disambiguation_candidates: List[Dict[str, Any]],
        malformed_flags: List[str],
    ) -> bool:
        """Assess whether clarification is needed"""

        # Need clarification if confidence is low
        if confidence < 0.4:
            return True

        # Need clarification if there are disambiguation issues
        if any(
            candidate["requires_clarification"]
            for candidate in disambiguation_candidates
        ):
            return True

        # Need clarification if input is malformed
        if malformed_flags:
            return True

        return False


# Global analyzer instance
linguistic_analyzer = LinguisticAnalyzer()


@register_uip_handler(
    UIPStep.STEP_1_LINGUISTIC, dependencies=[UIPStep.STEP_0_PREPROCESSING], timeout=30
)
async def handle_linguistic_analysis(context: UIPContext) -> Dict[str, Any]:
    """
    UIP Step 1: Linguistic Analysis Handler

    Performs complete linguistic analysis including intent classification,
    entity extraction, semantic representation, and disambiguation analysis.
    """
    logger = logging.getLogger(__name__)

    try:
        # Perform linguistic analysis
        analysis_result = await linguistic_analyzer.analyze(context)

        # Convert result to dictionary for context storage
        result_dict = {
            "step": "linguistic_analysis_complete",
            "intent_classification": analysis_result.intent_classification,
            "entity_extraction": analysis_result.entity_extraction,
            "semantic_representation": analysis_result.semantic_representation,
            "confidence": analysis_result.confidence,
            "language_detected": analysis_result.language_detected,
            "malformed_input_flags": analysis_result.malformed_input_flags,
            "clarification_needed": analysis_result.clarification_needed,
            "disambiguation_candidates": analysis_result.disambiguation_candidates,
            "contextual_metadata": analysis_result.contextual_metadata,
        }

        # Add disclaimers if needed
        disclaimers = []
        if analysis_result.malformed_input_flags:
            disclaimers.append(
                "Input may contain formatting issues that could affect processing."
            )
        if analysis_result.clarification_needed:
            disclaimers.append(
                "Input may be ambiguous and could benefit from clarification."
            )
        if analysis_result.confidence < 0.5:
            disclaimers.append("Linguistic analysis confidence is low.")

        result_dict["disclaimers"] = disclaimers

        logger.info(
            f"Linguistic analysis completed for {context.correlation_id} "
            f"(confidence: {analysis_result.confidence:.3f}, "
            f"clarification_needed: {analysis_result.clarification_needed})"
        )

        return result_dict

    except Exception as e:
        logger.error(f"Linguistic analysis failed for {context.correlation_id}: {e}")
        raise


__all__ = [
    "LinguisticAnalysisResult",
    "LinguisticAnalyzer",
    "linguistic_analyzer",
    "handle_linguistic_analysis",
]
