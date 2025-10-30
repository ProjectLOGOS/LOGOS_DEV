"""
Translation Engine - UIP Step 2 IEL Ontological Synthesis Gateway

Unified entrypoint for translating normalized lambda structures into
natural language representations for downstream processing.
Replaces legacy t_engine.py and semantic_bridge.py with unified approach.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("IEL_ONTO_KIT")


def convert_to_nl(normalized_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert normalized lambda structures to natural language representation

    Args:
        normalized_data: Normalized structure data from lambda_core

    Returns:
        Dict containing natural language translations and metadata
    """
    try:
        logger.info("Starting lambda structure to NL translation")

        # Extract normalized payload
        payload = normalized_data.get("payload", {})
        lambda_structures = payload.get("normalized_structures", {})
        normalization_metadata = payload.get("normalization_metadata", {})

        # Initialize translation context
        translation_context = {
            "timestamp": datetime.utcnow().isoformat(),
            "source_complexity": normalization_metadata.get("complexity_score", 0.0),
            "translation_mode": _determine_translation_mode(lambda_structures),
            "natural_language_outputs": {},
            "semantic_mappings": {},
            "translation_quality": {},
        }

        # Process each normalized structure type
        for structure_type, structure_data in lambda_structures.items():
            nl_output = _translate_structure(
                structure_type, structure_data, translation_context
            )
            translation_context["natural_language_outputs"][structure_type] = nl_output

            # Generate semantic mappings
            semantic_mapping = _generate_semantic_mapping(
                structure_type, structure_data, nl_output
            )
            translation_context["semantic_mappings"][structure_type] = semantic_mapping

        # Synthesize unified natural language representation
        unified_nl = _synthesize_unified_representation(translation_context)

        # Calculate translation quality metrics
        quality_metrics = _calculate_translation_quality(
            translation_context, unified_nl
        )
        translation_context["translation_quality"] = quality_metrics

        logger.info(
            f"NL translation completed: quality={quality_metrics.get('overall_quality', 0.0):.3f}"
        )

        return {
            "status": "ok",
            "payload": {
                "unified_natural_language": unified_nl,
                "structured_translations": translation_context[
                    "natural_language_outputs"
                ],
                "semantic_mappings": translation_context["semantic_mappings"],
                "translation_metadata": {
                    "source_complexity": translation_context["source_complexity"],
                    "translation_mode": translation_context["translation_mode"],
                    "quality_metrics": quality_metrics,
                    "processing_timestamp": translation_context["timestamp"],
                },
            },
            "metadata": {
                "stage": "translation",
                "structures_translated": len(lambda_structures),
                "quality_score": quality_metrics.get("overall_quality", 0.0),
            },
        }

    except Exception as e:
        logger.error(f"NL translation failed: {e}")
        raise


def _generate_semantic_mapping(
    structure_type: str, structure_data: Dict[str, Any], nl_output: str
) -> Dict[str, Any]:
    """Generate semantic mapping between formal structure and natural language"""
    # Handle various nl_output types
    if isinstance(nl_output, dict):
        text_elements = list(nl_output.keys())[:5]
    elif isinstance(nl_output, str):
        text_elements = nl_output.split()[:5]
    else:
        text_elements = ["conceptual", "representation"]

    return {
        "formal_elements": (
            list(structure_data.keys()) if isinstance(structure_data, dict) else []
        ),
        "natural_elements": text_elements,
        "mapping_confidence": 0.8,
        "semantic_relations": ["conceptual_bridge", "linguistic_mapping"],
    }


def _apply_grammar_rules(semantic_mapping: Dict[str, Any]) -> List[str]:
    """Apply grammar rules to semantic mapping"""
    return semantic_mapping.get(
        "natural_elements", ["structured", "conceptual", "representation"]
    )


def _optimize_readability(nl_elements: List[str]) -> str:
    """Optimize natural language elements for readability"""
    return " ".join(nl_elements) + " with enhanced semantic clarity"


def _determine_translation_mode(lambda_structures: Dict[str, Any]) -> str:
    """Determine optimal translation mode based on structure complexity"""
    try:
        structure_count = len(lambda_structures)

        # Analyze structure types
        has_formal_logic = any(
            "logic" in key.lower() for key in lambda_structures.keys()
        )
        has_natural_lang = any(
            "natural" in key.lower() for key in lambda_structures.keys()
        )
        has_combinators = any(
            "combinator" in key.lower() for key in lambda_structures.keys()
        )

        # Determine mode based on content
        if structure_count <= 2 and not has_formal_logic:
            return "simple"
        elif has_formal_logic and has_combinators:
            return "formal"
        elif has_natural_lang:
            return "hybrid"
        else:
            return "standard"

    except Exception:
        return "standard"


def _translate_structure(
    structure_type: str, structure_data: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    """Translate individual lambda structure to natural language"""
    try:
        # Extract structure components
        lambda_expressions = structure_data.get("lambda_expressions", [])
        reduction_steps = structure_data.get("reduction_steps", [])
        semantic_content = structure_data.get("semantic_content", {})

        # Choose translation strategy based on structure type
        if "combinator" in structure_type.lower():
            nl_text = _translate_combinator_structure(
                lambda_expressions, reduction_steps
            )
        elif "logic" in structure_type.lower():
            nl_text = _translate_logical_structure(lambda_expressions, semantic_content)
        elif "natural" in structure_type.lower():
            nl_text = _translate_natural_structure(semantic_content)
        else:
            nl_text = _translate_standard_structure(
                lambda_expressions, semantic_content
            )

        # Generate structural description
        structural_description = _generate_structural_description(structure_data)

        # Create readability metrics
        readability = _assess_readability(nl_text)

        return {
            "natural_language_text": nl_text,
            "structural_description": structural_description,
            "readability_metrics": readability,
            "translation_confidence": _calculate_translation_confidence(
                structure_data, nl_text
            ),
        }

    except Exception as e:
        logger.warning(f"Structure translation failed for {structure_type}: {e}")
        return {
            "natural_language_text": f"[Translation unavailable for {structure_type}]",
            "structural_description": "Error in translation",
            "readability_metrics": {"score": 0.0, "complexity": "unknown"},
            "translation_confidence": 0.0,
        }


def _translate_combinator_structure(
    expressions: List[Dict[str, Any]], reduction_steps: List[Dict[str, Any]]
) -> str:
    """Translate combinator-based lambda structures"""
    try:
        if not expressions:
            return "No combinator expressions found."

        # Extract combinator types and patterns
        combinator_descriptions = []

        for expr in expressions:
            expr_type = expr.get("type", "unknown")
            combinator_name = expr.get("combinator", "")

            if combinator_name in ["S", "K", "I"]:
                # Standard SKI combinators
                descriptions = {
                    "S": "applies function composition with argument distribution",
                    "K": "creates constant function, discarding second argument",
                    "I": "returns input unchanged (identity function)",
                }
                desc = descriptions.get(
                    combinator_name, "performs combinator operation"
                )
                combinator_descriptions.append(
                    f"The {combinator_name} combinator {desc}"
                )
            else:
                combinator_descriptions.append(
                    f"A {expr_type} expression performs logical transformation"
                )

        # Describe reduction process
        if reduction_steps:
            reduction_desc = f"through {len(reduction_steps)} reduction steps"
        else:
            reduction_desc = "via direct evaluation"

        # Combine descriptions
        if len(combinator_descriptions) == 1:
            return f"{combinator_descriptions[0]} {reduction_desc}."
        else:
            return f"The expression combines {len(combinator_descriptions)} operations {reduction_desc}."

    except Exception:
        return "Complex combinator expression with multiple transformation steps."


def _translate_logical_structure(
    expressions: List[Dict[str, Any]], semantic_content: Dict[str, Any]
) -> str:
    """Translate logical lambda structures"""
    try:
        # Extract logical components
        predicates = semantic_content.get("predicates", [])
        quantifiers = semantic_content.get("quantifiers", [])
        logical_operators = semantic_content.get("logical_operators", [])

        # Build natural language description
        components = []

        # Describe quantifiers
        if quantifiers:
            quant_desc = _describe_quantifiers(quantifiers)
            if quant_desc:
                components.append(quant_desc)

        # Describe predicates
        if predicates:
            pred_desc = _describe_predicates(predicates)
            if pred_desc:
                components.append(pred_desc)

        # Describe logical structure
        if logical_operators:
            logic_desc = _describe_logical_operators(logical_operators)
            if logic_desc:
                components.append(logic_desc)

        # Combine components
        if components:
            return " ".join(components) + "."
        else:
            return "A logical structure with formal reasoning patterns."

    except Exception:
        return "Complex logical expression with formal semantic content."


def _translate_natural_structure(semantic_content: Dict[str, Any]) -> str:
    """Translate natural language oriented structures"""
    try:
        # Extract natural language elements
        entities = semantic_content.get("entities", [])
        relations = semantic_content.get("relations", [])
        intentions = semantic_content.get("intentions", [])

        # Build natural description
        parts = []

        if entities:
            entity_names = [entity.get("name", "entity") for entity in entities]
            if len(entity_names) == 1:
                parts.append(f"concerning {entity_names[0]}")
            elif len(entity_names) > 1:
                parts.append(
                    f"involving {', '.join(entity_names[:-1])} and {entity_names[-1]}"
                )

        if relations:
            rel_descriptions = [rel.get("description", "relates") for rel in relations]
            if rel_descriptions:
                parts.append(f"where {' and '.join(rel_descriptions)}")

        if intentions:
            intent_descriptions = [
                intent.get("description", "intends") for intent in intentions
            ]
            if intent_descriptions:
                parts.append(f"with intention to {' and '.join(intent_descriptions)}")

        if parts:
            return "This expression describes a situation " + " ".join(parts) + "."
        else:
            return "A natural language expression with semantic content."

    except Exception:
        return "Natural language expression with structured semantic meaning."


def _translate_standard_structure(
    expressions: List[Dict[str, Any]], semantic_content: Dict[str, Any]
) -> str:
    """Translate standard lambda structures"""
    try:
        # Extract key information
        expr_count = len(expressions)
        has_semantic = bool(semantic_content)

        # Build description based on available information
        if expr_count == 0:
            return "An abstract expression structure."
        elif expr_count == 1:
            expr = expressions[0]
            expr_type = expr.get("type", "lambda")
            return f"A {expr_type} expression that transforms input according to defined rules."
        else:
            return f"A compound expression with {expr_count} lambda terms performing sequential transformations."

    except Exception:
        return "A lambda calculus expression with computational content."


def _describe_quantifiers(quantifiers: List[Dict[str, Any]]) -> str:
    """Generate natural language description of quantifiers"""
    try:
        descriptions = []
        for quant in quantifiers:
            quant_type = quant.get("type", "")
            if quant_type == "universal":
                descriptions.append("for all cases")
            elif quant_type == "existential":
                descriptions.append("there exists a case where")
            else:
                descriptions.append("under certain conditions")

        if descriptions:
            return "The expression states that " + " and ".join(descriptions)
        return ""

    except Exception:
        return ""


def _describe_predicates(predicates: List[Dict[str, Any]]) -> str:
    """Generate natural language description of predicates"""
    try:
        pred_names = [pred.get("name", "property") for pred in predicates]
        if len(pred_names) == 1:
            return f"the {pred_names[0]} property holds"
        elif len(pred_names) > 1:
            return f"the properties {', '.join(pred_names[:-1])} and {pred_names[-1]} are satisfied"
        return ""

    except Exception:
        return ""


def _describe_logical_operators(operators: List[Dict[str, Any]]) -> str:
    """Generate natural language description of logical operators"""
    try:
        op_descriptions = []
        for op in operators:
            op_type = op.get("type", "")
            if op_type == "and":
                op_descriptions.append("conjunction")
            elif op_type == "or":
                op_descriptions.append("disjunction")
            elif op_type == "implies":
                op_descriptions.append("implication")
            elif op_type == "not":
                op_descriptions.append("negation")
            else:
                op_descriptions.append("logical operation")

        if op_descriptions:
            if len(op_descriptions) == 1:
                return f"using {op_descriptions[0]}"
            else:
                return f"combining through {' and '.join(op_descriptions)}"
        return ""

    except Exception:
        return ""


def _generate_structural_description(structure_data: Dict[str, Any]) -> str:
    """Generate description of lambda structure organization"""
    try:
        expr_count = len(structure_data.get("lambda_expressions", []))
        reduction_count = len(structure_data.get("reduction_steps", []))

        parts = []

        if expr_count > 0:
            parts.append(f"{expr_count} expression{'s' if expr_count > 1 else ''}")

        if reduction_count > 0:
            parts.append(
                f"{reduction_count} reduction step{'s' if reduction_count > 1 else ''}"
            )

        if parts:
            return f"Structure contains {' and '.join(parts)}."
        else:
            return "Basic structural organization."

    except Exception:
        return "Standard lambda structure."


def _synthesize_unified_representation(translation_context: Dict[str, Any]) -> str:
    """Synthesize unified natural language from all structure translations"""
    try:
        nl_outputs = translation_context.get("natural_language_outputs", {})

        if not nl_outputs:
            return "No translatable content found."

        # Extract individual translations
        translations = []
        for structure_type, output in nl_outputs.items():
            nl_text = output.get("natural_language_text", "")
            if nl_text and not nl_text.startswith("[Translation unavailable"):
                translations.append(nl_text)

        if not translations:
            return "Translation content is not available in natural language form."
        elif len(translations) == 1:
            return translations[0]
        else:
            # Combine multiple translations
            return (
                "This represents a complex expression that "
                + " Additionally, it ".join([t.rstrip(".") for t in translations])
                + "."
            )

    except Exception:
        return "Complex multi-part expression with structured logical content."


def _assess_readability(text: str) -> Dict[str, Any]:
    """Assess readability metrics of natural language text"""
    try:
        if not text:
            return {"score": 0.0, "complexity": "unknown", "word_count": 0}

        # Basic readability metrics
        words = text.split()
        sentences = text.count(".") + text.count("!") + text.count("?")
        sentences = max(sentences, 1)  # Avoid division by zero

        avg_words_per_sentence = len(words) / sentences

        # Simple readability score (inverse of complexity)
        if avg_words_per_sentence <= 10:
            complexity = "simple"
            score = 0.9
        elif avg_words_per_sentence <= 20:
            complexity = "moderate"
            score = 0.7
        else:
            complexity = "complex"
            score = 0.5

        return {
            "score": score,
            "complexity": complexity,
            "word_count": len(words),
            "sentence_count": sentences,
            "avg_words_per_sentence": avg_words_per_sentence,
        }

    except Exception:
        return {"score": 0.5, "complexity": "unknown", "word_count": 0}


def _calculate_translation_confidence(
    structure_data: Dict[str, Any], nl_text: str
) -> float:
    """Calculate confidence score for translation quality"""
    try:
        # Factors affecting confidence
        has_expressions = len(structure_data.get("lambda_expressions", [])) > 0
        has_semantic_content = bool(structure_data.get("semantic_content", {}))
        has_reductions = len(structure_data.get("reduction_steps", [])) > 0

        text_quality = (
            1.0
            if nl_text and not nl_text.startswith("[Translation unavailable")
            else 0.0
        )
        text_length = min(
            len(nl_text.split()) / 20.0, 1.0
        )  # Normalize to reasonable length

        # Calculate confidence
        structure_score = (
            (0.4 if has_expressions else 0.0)
            + (0.3 if has_semantic_content else 0.0)
            + (0.3 if has_reductions else 0.0)
        )

        text_score = text_quality * 0.7 + text_length * 0.3

        overall_confidence = structure_score * 0.6 + text_score * 0.4

        return min(max(overall_confidence, 0.0), 1.0)

    except Exception:
        return 0.5


def _calculate_translation_quality(
    translation_context: Dict[str, Any], unified_nl: str
) -> Dict[str, float]:
    """Calculate overall translation quality metrics"""
    try:
        nl_outputs = translation_context.get("natural_language_outputs", {})

        # Individual translation confidences
        confidences = []
        for output in nl_outputs.values():
            conf = output.get("translation_confidence", 0.0)
            confidences.append(conf)

        # Average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Unified text quality
        unified_readability = _assess_readability(unified_nl)
        text_quality = unified_readability.get("score", 0.0)

        # Coverage (proportion of structures successfully translated)
        successful_translations = sum(
            1
            for output in nl_outputs.values()
            if not output.get("natural_language_text", "").startswith(
                "[Translation unavailable"
            )
        )
        coverage = successful_translations / len(nl_outputs) if nl_outputs else 0.0

        # Overall quality
        overall_quality = avg_confidence * 0.4 + text_quality * 0.3 + coverage * 0.3

        return {
            "overall_quality": overall_quality,
            "average_confidence": avg_confidence,
            "text_quality": text_quality,
            "translation_coverage": coverage,
            "structures_translated": len(nl_outputs),
        }

    except Exception:
        return {
            "overall_quality": 0.0,
            "average_confidence": 0.0,
            "text_quality": 0.0,
            "translation_coverage": 0.0,
            "structures_translated": 0,
        }
