"""
Advanced Language Translation Engine - UIP Step 1 Component
==========================================================

Multi-domain translation engine for natural language, formal logic, and semantic representations.
Supports hierarchical translation with domain-specific vocabularies and inference rules.

Adapted from: V2_Possible_Gap_Fillers/uipie/uipie_translator.py
Enhanced with: Trinity vector integration, modal logic translation, ontological mapping
"""

from protocols.shared.system_imports import *
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import json


class TranslationDomain(Enum):
    """Supported translation domains"""
    NATURAL_LANGUAGE = "natural"
    FORMAL_LOGIC = "logic"
    MODAL_LOGIC = "modal"
    TRINITY_VECTORS = "trinity"
    ONTOLOGICAL = "ontology"
    THEOLOGICAL = "theology"
    PHILOSOPHICAL = "philosophy"
    MATHEMATICAL = "math"
    PROCEDURAL = "procedure"


class TranslationMode(Enum):
    """Translation processing modes"""
    DIRECT = "direct"           # Direct mapping
    INFERENTIAL = "inferential" # With logical inference
    CONTEXTUAL = "contextual"   # Context-aware
    HIERARCHICAL = "hierarchical" # Multi-level analysis


@dataclass
class TranslationRule:
    """Individual translation rule"""
    source_pattern: str
    target_pattern: str
    domain: TranslationDomain
    confidence: float
    conditions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationContext:
    """Translation context and state"""
    source_domain: TranslationDomain
    target_domain: TranslationDomain
    mode: TranslationMode
    context_vars: Dict[str, Any] = field(default_factory=dict)
    inference_chain: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.5


@dataclass
class TranslationResult:
    """Translation operation result"""
    original_text: str
    translated_text: str
    source_domain: TranslationDomain
    target_domain: TranslationDomain
    confidence: float
    applied_rules: List[str]
    inference_steps: List[str]
    semantic_annotations: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class VocabularyManager:
    """Domain-specific vocabulary and concept manager"""
    
    def __init__(self):
        self.vocabularies = self._initialize_vocabularies()
        self.concept_mappings = self._initialize_concept_mappings()
        
    def _initialize_vocabularies(self) -> Dict[TranslationDomain, Dict[str, str]]:
        """Initialize domain-specific vocabularies"""
        
        return {
            TranslationDomain.NATURAL_LANGUAGE: {
                # Common natural language constructs
                "exists": "there exists",
                "forall": "for all", 
                "implies": "implies that",
                "necessary": "must be",
                "possible": "might be",
                "always": "in every case",
                "sometimes": "in some cases"
            },
            
            TranslationDomain.FORMAL_LOGIC: {
                # Logical operators and quantifiers
                "exists": "∃",
                "forall": "∀",
                "implies": "→",
                "and": "∧",
                "or": "∨", 
                "not": "¬",
                "equivalent": "↔"
            },
            
            TranslationDomain.MODAL_LOGIC: {
                # Modal operators
                "necessary": "□",
                "possible": "◊",
                "knows": "K",
                "believes": "B",
                "obligated": "O",
                "permitted": "P"
            },
            
            TranslationDomain.TRINITY_VECTORS: {
                # Trinity-specific terminology
                "existence": "E-dimension",
                "goodness": "G-dimension", 
                "truth": "T-dimension",
                "divine_attribute": "Trinity vector",
                "perfection": "Trinity unity",
                "coherence": "dimensional harmony"
            },
            
            TranslationDomain.THEOLOGICAL: {
                # Theological concepts
                "omnipotent": "all-powerful",
                "omniscient": "all-knowing",
                "omnipresent": "everywhere present",
                "transcendent": "beyond finite",
                "immanent": "within creation",
                "eternal": "outside time"
            },
            
            TranslationDomain.ONTOLOGICAL: {
                # Ontological categories
                "substance": "fundamental being",
                "attribute": "property of being",
                "relation": "connection between beings",
                "mode": "way of being",
                "essence": "what something is",
                "existence": "that something is"
            }
        }
    
    def _initialize_concept_mappings(self) -> Dict[Tuple[TranslationDomain, TranslationDomain], Dict[str, str]]:
        """Initialize cross-domain concept mappings"""
        
        mappings = {}
        
        # Natural Language <-> Formal Logic
        mappings[(TranslationDomain.NATURAL_LANGUAGE, TranslationDomain.FORMAL_LOGIC)] = {
            "there exists": "∃",
            "for all": "∀",
            "if then": "→",
            "and": "∧",
            "or": "∨",
            "not": "¬"
        }
        
        # Natural Language <-> Modal Logic
        mappings[(TranslationDomain.NATURAL_LANGUAGE, TranslationDomain.MODAL_LOGIC)] = {
            "necessarily": "□",
            "possibly": "◊", 
            "must be": "□",
            "might be": "◊",
            "knows that": "K",
            "believes that": "B"
        }
        
        # Trinity Vectors <-> Theological
        mappings[(TranslationDomain.TRINITY_VECTORS, TranslationDomain.THEOLOGICAL)] = {
            "E-dimension": "divine existence",
            "G-dimension": "divine goodness",
            "T-dimension": "divine truth",
            "Trinity vector": "divine nature",
            "dimensional harmony": "theological coherence"
        }
        
        # Formal Logic <-> Modal Logic
        mappings[(TranslationDomain.FORMAL_LOGIC, TranslationDomain.MODAL_LOGIC)] = {
            "∀x": "□",  # Universal quantification as necessity
            "∃x": "◊",  # Existential quantification as possibility
        }
        
        return mappings
    
    def get_vocabulary(self, domain: TranslationDomain) -> Dict[str, str]:
        """Get vocabulary for specific domain"""
        return self.vocabularies.get(domain, {})
    
    def get_mapping(self, source: TranslationDomain, target: TranslationDomain) -> Dict[str, str]:
        """Get concept mapping between domains"""
        return self.concept_mappings.get((source, target), {})


class TranslationRuleEngine:
    """Rule-based translation engine"""
    
    def __init__(self, vocabulary_manager: VocabularyManager):
        self.vocab_manager = vocabulary_manager
        self.rules = self._initialize_rules()
        
    def _initialize_rules(self) -> List[TranslationRule]:
        """Initialize translation rules"""
        
        rules = []
        
        # Natural Language -> Formal Logic rules
        rules.extend([
            TranslationRule(
                source_pattern=r"there exists (\w+) such that (.+)",
                target_pattern=r"∃\1(\2)",
                domain=TranslationDomain.FORMAL_LOGIC,
                confidence=0.8
            ),
            TranslationRule(
                source_pattern=r"for all (\w+), (.+)",
                target_pattern=r"∀\1(\2)",
                domain=TranslationDomain.FORMAL_LOGIC,
                confidence=0.8
            ),
            TranslationRule(
                source_pattern=r"if (.+) then (.+)",
                target_pattern=r"(\1) → (\2)",
                domain=TranslationDomain.FORMAL_LOGIC,
                confidence=0.9
            )
        ])
        
        # Natural Language -> Modal Logic rules
        rules.extend([
            TranslationRule(
                source_pattern=r"necessarily (.+)",
                target_pattern=r"□(\1)",
                domain=TranslationDomain.MODAL_LOGIC,
                confidence=0.9
            ),
            TranslationRule(
                source_pattern=r"possibly (.+)",
                target_pattern=r"◊(\1)",
                domain=TranslationDomain.MODAL_LOGIC,
                confidence=0.9
            ),
            TranslationRule(
                source_pattern=r"(\w+) knows that (.+)",
                target_pattern=r"K_\1(\2)",
                domain=TranslationDomain.MODAL_LOGIC,
                confidence=0.8
            )
        ])
        
        # Trinity Vector rules
        rules.extend([
            TranslationRule(
                source_pattern=r"existence level (\d+\.?\d*)",
                target_pattern=r"E=\1",
                domain=TranslationDomain.TRINITY_VECTORS,
                confidence=0.9
            ),
            TranslationRule(
                source_pattern=r"goodness level (\d+\.?\d*)",
                target_pattern=r"G=\1",
                domain=TranslationDomain.TRINITY_VECTORS,
                confidence=0.9
            ),
            TranslationRule(
                source_pattern=r"truth level (\d+\.?\d*)",
                target_pattern=r"T=\1",
                domain=TranslationDomain.TRINITY_VECTORS,
                confidence=0.9
            )
        ])
        
        return rules
    
    def find_applicable_rules(
        self, 
        text: str, 
        context: TranslationContext
    ) -> List[Tuple[TranslationRule, re.Match]]:
        """Find translation rules applicable to text"""
        
        applicable = []
        
        for rule in self.rules:
            # Check if rule applies to target domain
            if rule.domain != context.target_domain:
                continue
            
            # Check pattern match
            match = re.search(rule.source_pattern, text, re.IGNORECASE)
            if match:
                # Check conditions if any
                if self._check_rule_conditions(rule, context):
                    applicable.append((rule, match))
        
        # Sort by confidence
        applicable.sort(key=lambda x: x[0].confidence, reverse=True)
        
        return applicable
    
    def _check_rule_conditions(self, rule: TranslationRule, context: TranslationContext) -> bool:
        """Check if rule conditions are satisfied"""
        
        for condition in rule.conditions:
            # Simple condition checking - can be enhanced
            if condition.startswith("domain="):
                required_domain = condition.split("=")[1]
                if context.source_domain.value != required_domain:
                    return False
            elif condition.startswith("mode="):
                required_mode = condition.split("=")[1]
                if context.mode.value != required_mode:
                    return False
        
        return True


class SemanticAnalyzer:
    """Semantic analysis and annotation engine"""
    
    def __init__(self):
        self.semantic_patterns = self._initialize_semantic_patterns()
        
    def _initialize_semantic_patterns(self) -> Dict[str, List[str]]:
        """Initialize semantic analysis patterns"""
        
        return {
            'quantifiers': [
                r'\b(all|every|each|any)\b',
                r'\b(some|exists?|there\s+(?:is|are|exists?))\b',
                r'\b(no|none|nothing|never)\b'
            ],
            'modal_expressions': [
                r'\b(necessarily|must|required|essential)\b',
                r'\b(possibly|might|could|maybe|perhaps)\b',
                r'\b(certainly|definitely|surely)\b'
            ],
            'theological_concepts': [
                r'\b(divine|god|deity|sacred|holy)\b',
                r'\b(omnipotent|omniscient|omnipresent|perfect)\b',
                r'\b(eternal|infinite|transcendent|immanent)\b'
            ],
            'logical_connectives': [
                r'\b(and|but|also|moreover|furthermore)\b',
                r'\b(or|either|alternatively)\b',
                r'\b(if|when|whenever|provided|given)\b',
                r'\b(then|therefore|thus|hence|consequently)\b'
            ]
        }
    
    def analyze_semantics(self, text: str) -> Dict[str, Any]:
        """Perform semantic analysis on text"""
        
        annotations = {
            'semantic_categories': {},
            'complexity_metrics': {},
            'structural_analysis': {}
        }
        
        # Categorize semantic elements
        for category, patterns in self.semantic_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(re.findall(pattern, text, re.IGNORECASE))
            annotations['semantic_categories'][category] = matches
        
        # Calculate complexity metrics
        annotations['complexity_metrics'] = {
            'word_count': len(text.split()),
            'unique_words': len(set(text.lower().split())),
            'avg_word_length': sum(len(word) for word in text.split()) / max(1, len(text.split())),
            'modal_density': len(annotations['semantic_categories'].get('modal_expressions', [])),
            'logical_density': len(annotations['semantic_categories'].get('logical_connectives', []))
        }
        
        # Structural analysis
        annotations['structural_analysis'] = {
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'clause_complexity': self._analyze_clause_complexity(text),
            'nested_structures': self._detect_nested_structures(text)
        }
        
        return annotations
    
    def _analyze_clause_complexity(self, text: str) -> float:
        """Analyze complexity of clause structure"""
        
        # Count subordinating conjunctions and relative pronouns
        subordinating_patterns = [
            r'\b(because|since|although|while|whereas|if|unless|until)\b',
            r'\b(who|whom|whose|which|that|where|when)\b'
        ]
        
        subordinate_count = 0
        for pattern in subordinating_patterns:
            subordinate_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Normalize by sentence count
        sentence_count = max(1, len(re.split(r'[.!?]+', text)))
        complexity = subordinate_count / sentence_count
        
        return min(1.0, complexity)  # Cap at 1.0
    
    def _detect_nested_structures(self, text: str) -> List[str]:
        """Detect nested logical/grammatical structures"""
        
        nested = []
        
        # Parenthetical expressions
        if '(' in text and ')' in text:
            nested.append('parenthetical')
        
        # Nested quantification
        if re.search(r'(all|some|every).*?(all|some|every)', text, re.IGNORECASE):
            nested.append('nested_quantification')
        
        # Modal embedding
        if re.search(r'(necessarily|possibly).*?(necessarily|possibly)', text, re.IGNORECASE):
            nested.append('modal_embedding')
        
        return nested


class AdvancedTranslationEngine:
    """Main advanced translation engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.vocab_manager = VocabularyManager()
        self.rule_engine = TranslationRuleEngine(self.vocab_manager)
        self.semantic_analyzer = SemanticAnalyzer()
        
        # Translation cache
        self._translation_cache: Dict[str, TranslationResult] = {}
        
        # Configuration
        self.default_confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.max_inference_depth = self.config.get('max_inference_depth', 5)
        
        self.logger.info("Advanced translation engine initialized")
    
    def translate(
        self,
        text: str,
        source_domain: TranslationDomain,
        target_domain: TranslationDomain,
        mode: TranslationMode = TranslationMode.CONTEXTUAL,
        context_vars: Optional[Dict[str, Any]] = None
    ) -> TranslationResult:
        """
        Translate text between domains with specified mode
        
        Args:
            text: Text to translate
            source_domain: Source domain
            target_domain: Target domain  
            mode: Translation mode
            context_vars: Additional context variables
            
        Returns:
            TranslationResult: Comprehensive translation results
        """
        try:
            # Create translation context
            context = TranslationContext(
                source_domain=source_domain,
                target_domain=target_domain,
                mode=mode,
                context_vars=context_vars or {},
                confidence_threshold=self.default_confidence_threshold
            )
            
            # Check cache
            cache_key = f"{text}_{source_domain.value}_{target_domain.value}_{mode.value}"
            if cache_key in self._translation_cache:
                return self._translation_cache[cache_key]
            
            # Perform semantic analysis
            semantic_annotations = self.semantic_analyzer.analyze_semantics(text)
            
            # Execute translation based on mode
            if mode == TranslationMode.DIRECT:
                result = self._translate_direct(text, context, semantic_annotations)
            elif mode == TranslationMode.INFERENTIAL:
                result = self._translate_inferential(text, context, semantic_annotations)
            elif mode == TranslationMode.CONTEXTUAL:
                result = self._translate_contextual(text, context, semantic_annotations)
            elif mode == TranslationMode.HIERARCHICAL:
                result = self._translate_hierarchical(text, context, semantic_annotations)
            else:
                result = self._translate_direct(text, context, semantic_annotations)
            
            # Cache result
            self._translation_cache[cache_key] = result
            
            self.logger.debug(f"Translation completed: {source_domain.value} -> {target_domain.value} (confidence: {result.confidence:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            
            # Return error result
            return TranslationResult(
                original_text=text,
                translated_text=f"[Translation Error: {str(e)}]",
                source_domain=source_domain,
                target_domain=target_domain,
                confidence=0.0,
                applied_rules=[],
                inference_steps=[f"Error: {str(e)}"],
                semantic_annotations={}
            )
    
    def _translate_direct(
        self, 
        text: str, 
        context: TranslationContext, 
        semantic_annotations: Dict[str, Any]
    ) -> TranslationResult:
        """Direct translation using vocabulary mappings and rules"""
        
        translated_text = text
        applied_rules = []
        inference_steps = ["Direct translation mode"]
        
        # Apply vocabulary mappings
        mapping = self.vocab_manager.get_mapping(context.source_domain, context.target_domain)
        for source_term, target_term in mapping.items():
            if source_term in translated_text:
                translated_text = translated_text.replace(source_term, target_term)
                applied_rules.append(f"vocab:{source_term}->{target_term}")
        
        # Apply translation rules
        applicable_rules = self.rule_engine.find_applicable_rules(translated_text, context)
        
        for rule, match in applicable_rules[:3]:  # Limit to top 3 rules
            try:
                # Apply rule transformation
                new_text = re.sub(rule.source_pattern, rule.target_pattern, translated_text)
                if new_text != translated_text:
                    translated_text = new_text
                    applied_rules.append(f"rule:{rule.source_pattern}")
                    inference_steps.append(f"Applied rule: {rule.source_pattern} -> {rule.target_pattern}")
            except Exception as e:
                self.logger.warning(f"Rule application failed: {e}")
        
        # Calculate confidence
        confidence = self._calculate_translation_confidence(
            text, translated_text, applicable_rules, semantic_annotations
        )
        
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_domain=context.source_domain,
            target_domain=context.target_domain,
            confidence=confidence,
            applied_rules=applied_rules,
            inference_steps=inference_steps,
            semantic_annotations=semantic_annotations
        )
    
    def _translate_inferential(
        self, 
        text: str, 
        context: TranslationContext, 
        semantic_annotations: Dict[str, Any]
    ) -> TranslationResult:
        """Inferential translation with logical reasoning"""
        
        # Start with direct translation
        result = self._translate_direct(text, context, semantic_annotations)
        
        # Add inferential steps
        inference_steps = result.inference_steps + ["Inferential analysis"]
        
        # Analyze logical structure for additional inferences
        if context.target_domain == TranslationDomain.FORMAL_LOGIC:
            # Add logical structure inference
            if 'quantifiers' in semantic_annotations['semantic_categories']:
                quantifiers = semantic_annotations['semantic_categories']['quantifiers']
                if quantifiers:
                    inference_steps.append(f"Inferred quantifier structure: {quantifiers}")
                    result.confidence = min(1.0, result.confidence + 0.1)
        
        elif context.target_domain == TranslationDomain.MODAL_LOGIC:
            # Add modal inference
            if 'modal_expressions' in semantic_annotations['semantic_categories']:
                modals = semantic_annotations['semantic_categories']['modal_expressions']
                if modals:
                    inference_steps.append(f"Inferred modal structure: {modals}")
                    result.confidence = min(1.0, result.confidence + 0.1)
        
        result.inference_steps = inference_steps
        return result
    
    def _translate_contextual(
        self, 
        text: str, 
        context: TranslationContext, 
        semantic_annotations: Dict[str, Any]
    ) -> TranslationResult:
        """Context-aware translation with adaptive processing"""
        
        # Start with inferential translation
        result = self._translate_inferential(text, context, semantic_annotations)
        
        # Add contextual adjustments
        inference_steps = result.inference_steps + ["Contextual adaptation"]
        
        # Adjust based on complexity
        complexity = semantic_annotations['complexity_metrics']
        if complexity['modal_density'] > 2:
            inference_steps.append("High modal density detected - enhanced modal processing")
            result.confidence = min(1.0, result.confidence + 0.05)
        
        if complexity['logical_density'] > 3:
            inference_steps.append("High logical density detected - enhanced logical processing")
            result.confidence = min(1.0, result.confidence + 0.05)
        
        # Context variable integration
        if context.context_vars:
            inference_steps.append(f"Applied context variables: {list(context.context_vars.keys())}")
            result.confidence = min(1.0, result.confidence + 0.1)
        
        result.inference_steps = inference_steps
        return result
    
    def _translate_hierarchical(
        self, 
        text: str, 
        context: TranslationContext, 
        semantic_annotations: Dict[str, Any]
    ) -> TranslationResult:
        """Hierarchical translation with multi-level analysis"""
        
        inference_steps = ["Hierarchical translation mode"]
        
        # Level 1: Lexical translation
        level1_context = TranslationContext(
            context.source_domain, context.target_domain, 
            TranslationMode.DIRECT, context.context_vars
        )
        level1_result = self._translate_direct(text, level1_context, semantic_annotations)
        inference_steps.append("Level 1: Lexical translation completed")
        
        # Level 2: Syntactic translation  
        level2_context = TranslationContext(
            context.source_domain, context.target_domain,
            TranslationMode.INFERENTIAL, context.context_vars
        )
        level2_result = self._translate_inferential(level1_result.translated_text, level2_context, semantic_annotations)
        inference_steps.append("Level 2: Syntactic translation completed")
        
        # Level 3: Semantic translation
        level3_context = TranslationContext(
            context.source_domain, context.target_domain,
            TranslationMode.CONTEXTUAL, context.context_vars
        )
        level3_result = self._translate_contextual(level2_result.translated_text, level3_context, semantic_annotations)
        inference_steps.append("Level 3: Semantic translation completed")
        
        # Combine results
        all_applied_rules = level1_result.applied_rules + level2_result.applied_rules + level3_result.applied_rules
        all_inference_steps = inference_steps + level1_result.inference_steps + level2_result.inference_steps + level3_result.inference_steps
        
        # Calculate hierarchical confidence
        hierarchical_confidence = (level1_result.confidence + level2_result.confidence + level3_result.confidence) / 3.0
        
        return TranslationResult(
            original_text=text,
            translated_text=level3_result.translated_text,
            source_domain=context.source_domain,
            target_domain=context.target_domain,
            confidence=hierarchical_confidence,
            applied_rules=all_applied_rules,
            inference_steps=all_inference_steps,
            semantic_annotations=semantic_annotations
        )
    
    def _calculate_translation_confidence(
        self,
        original: str,
        translated: str,
        applicable_rules: List[Tuple[TranslationRule, re.Match]],
        semantic_annotations: Dict[str, Any]
    ) -> float:
        """Calculate confidence in translation result"""
        
        # Base confidence from rule quality
        if applicable_rules:
            rule_confidence = sum(rule.confidence for rule, _ in applicable_rules) / len(applicable_rules)
        else:
            rule_confidence = 0.3  # Low confidence without rules
        
        # Coverage bonus (how much of text was translated)
        coverage = min(1.0, len(translated.split()) / max(1, len(original.split())))
        
        # Semantic complexity penalty (more complex = potentially less reliable)
        complexity_metrics = semantic_annotations.get('complexity_metrics', {})
        complexity_penalty = min(0.3, complexity_metrics.get('modal_density', 0) * 0.05 + 
                                complexity_metrics.get('logical_density', 0) * 0.03)
        
        confidence = rule_confidence * 0.6 + coverage * 0.3 - complexity_penalty
        return max(0.0, min(1.0, confidence))


# Global translation engine instance
advanced_translation_engine = AdvancedTranslationEngine()


__all__ = [
    'TranslationDomain',
    'TranslationMode', 
    'TranslationRule',
    'TranslationContext',
    'TranslationResult',
    'VocabularyManager',
    'TranslationRuleEngine',
    'SemanticAnalyzer',
    'AdvancedTranslationEngine',
    'advanced_translation_engine'
]