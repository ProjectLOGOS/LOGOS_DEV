"""
Lambda Core - UIP Step 2 IEL Ontological Synthesis Gateway

Core lambda calculus normalization logic for IEL ontological processing.
Handles lambda expression normalization, reduction, and structural optimization.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("IEL_ONTO_KIT")


def normalize_structure(recombined_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize lambda structures from recombined IEL outputs

    Args:
        recombined_data: Recombined IEL outputs from recombine_core

    Returns:
        Dict containing normalized lambda structures ready for translation
    """
    try:
        logger.info("Starting lambda structure normalization")

        # Extract recombined payload
        payload = recombined_data.get("payload", {})
        merged_vectors = payload.get("merged_vectors", {})
        ontological_alignments = payload.get("ontological_alignments", {})
        synthesis_metadata = payload.get("synthesis_metadata", {})

        # Initialize normalization context
        normalization_context = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_complexity": synthesis_metadata.get("quality_score", 0.0),
            "normalization_strategy": _determine_normalization_strategy(
                merged_vectors, ontological_alignments
            ),
            "normalized_structures": {},
            "lambda_expressions": {},
            "reduction_metadata": {},
        }

        # Generate lambda expressions from IEL structures
        lambda_expressions = _generate_lambda_expressions(
            merged_vectors, ontological_alignments
        )
        normalization_context["lambda_expressions"] = lambda_expressions

        # Process each expression type
        for expr_type, expressions in lambda_expressions.items():
            normalized_structure = _normalize_expression_type(
                expr_type, expressions, normalization_context
            )
            normalization_context["normalized_structures"][
                expr_type
            ] = normalized_structure

        # Perform global optimization
        optimized_structures = _optimize_structures(
            normalization_context["normalized_structures"]
        )
        normalization_context["normalized_structures"] = optimized_structures

        # Calculate complexity and quality metrics
        complexity_metrics = _calculate_complexity_metrics(normalization_context)
        normalization_context["normalization_metadata"] = {
            "complexity_score": complexity_metrics["overall_complexity"],
            "reduction_efficiency": complexity_metrics["reduction_efficiency"],
            "structural_coherence": complexity_metrics["structural_coherence"],
            "optimization_quality": complexity_metrics["optimization_quality"],
        }

        logger.info(
            f"Lambda normalization completed: complexity={complexity_metrics['overall_complexity']:.3f}"
        )

        return {
            "status": "ok",
            "payload": normalization_context,
            "metadata": {
                "stage": "lambda_normalization",
                "structures_processed": len(lambda_expressions),
                "complexity_score": complexity_metrics["overall_complexity"],
            },
        }

    except Exception as e:
        logger.error(f"Lambda structure normalization failed: {e}")
        raise


def _determine_normalization_strategy(
    merged_vectors: Dict[str, Any], ontological_alignments: Dict[str, Any]
) -> str:
    """Determine optimal normalization strategy based on input characteristics"""
    try:
        # Analyze input characteristics
        vector_coherence = merged_vectors.get("coherence_score", 0.0)
        synthesis_confidence = merged_vectors.get("synthesis_confidence", 0.0)
        domain_count = len(ontological_alignments)

        # Strategy selection logic
        if domain_count <= 3 and vector_coherence > 0.8:
            return "simple_reduction"
        elif domain_count > 10 or vector_coherence < 0.3:
            return "complex_normalization"
        elif synthesis_confidence > 0.7:
            return "confidence_driven"
        else:
            return "balanced_approach"

    except Exception:
        return "balanced_approach"


def _generate_lambda_expressions(
    merged_vectors: Dict[str, Any], ontological_alignments: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate lambda expressions from IEL merged data"""
    try:
        expressions = {
            "vector_combinators": [],
            "ontological_logic": [],
            "natural_language_bridge": [],
            "domain_abstractions": [],
        }

        # Generate vector combinator expressions
        weighted_avg = merged_vectors.get("weighted_average", [])
        if weighted_avg:
            vector_expr = _create_vector_combinator_expression(weighted_avg)
            expressions["vector_combinators"].append(vector_expr)

        # Generate ontological logic expressions
        for domain_name, alignment_data in ontological_alignments.items():
            ontological_expr = _create_ontological_expression(
                domain_name, alignment_data
            )
            expressions["ontological_logic"].append(ontological_expr)

        # Generate natural language bridge expressions
        nl_bridge_expr = _create_natural_language_bridge(
            merged_vectors, ontological_alignments
        )
        expressions["natural_language_bridge"].append(nl_bridge_expr)

        # Generate domain abstraction expressions
        domain_abstractions = _create_domain_abstractions(ontological_alignments)
        expressions["domain_abstractions"].extend(domain_abstractions)

        return expressions

    except Exception as e:
        logger.warning(f"Lambda expression generation failed: {e}")
        return {
            "vector_combinators": [],
            "ontological_logic": [],
            "natural_language_bridge": [],
            "domain_abstractions": [],
        }


def _create_vector_combinator_expression(vector: List[float]) -> Dict[str, Any]:
    """Create lambda expression for vector combinator operations"""
    try:
        # Convert vector to combinator representation
        combinator_terms = []

        for i, component in enumerate(vector):
            if abs(component) > 0.1:  # Threshold for significant components
                if component > 0:
                    combinator_terms.append(f"S({i})")
                else:
                    combinator_terms.append(f"K({i})")

        if not combinator_terms:
            combinator_terms = ["I"]  # Identity combinator for zero vector

        return {
            "type": "vector_combinator",
            "combinator": "composite",
            "terms": combinator_terms,
            "lambda_expression": f"λx.({' '.join(combinator_terms)}) x",
            "semantic_content": {
                "vector_dimension": len(vector),
                "significant_components": len(combinator_terms),
                "vector_norm": sum(abs(x) for x in vector),
            },
        }

    except Exception:
        return {
            "type": "vector_combinator",
            "combinator": "I",
            "terms": ["I"],
            "lambda_expression": "λx.x",
            "semantic_content": {},
        }


def _create_ontological_expression(
    domain_name: str, alignment_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Create lambda expression for ontological domain alignment"""
    try:
        ontological_property = alignment_data.get("property", "unknown")
        activation_strength = alignment_data.get("activation_strength", 0.0)
        trinity_weights = alignment_data.get("trinity_weights", {})

        # Create predicate-based lambda expression
        predicates = []

        # Domain predicate
        predicates.append(f"Domain({domain_name})")

        # Ontological property predicate
        predicates.append(f"OntoProp({ontological_property})")

        # Trinity weight predicates
        for trinity_dim, weight in trinity_weights.items():
            if weight > 0.3:  # Significant Trinity alignment
                predicates.append(f"Trinity_{trinity_dim}({weight:.2f})")

        # Activation predicate
        if activation_strength > 0.5:
            predicates.append(f"Activated({activation_strength:.2f})")

        lambda_expr = f"λx.∀y.({' ∧ '.join(predicates)})(x,y)"

        return {
            "type": "ontological_logic",
            "domain": domain_name,
            "property": ontological_property,
            "lambda_expression": lambda_expr,
            "predicates": predicates,
            "quantifiers": [{"type": "universal", "variable": "y"}],
            "logical_operators": [{"type": "and", "arity": len(predicates)}],
            "semantic_content": {
                "activation_strength": activation_strength,
                "trinity_alignment": trinity_weights,
            },
        }

    except Exception:
        return {
            "type": "ontological_logic",
            "domain": domain_name,
            "lambda_expression": "λx.True(x)",
            "semantic_content": {},
        }


def _create_natural_language_bridge(
    merged_vectors: Dict[str, Any], ontological_alignments: Dict[str, Any]
) -> Dict[str, Any]:
    """Create lambda expression bridging formal and natural language"""
    try:
        # Extract natural language elements
        entities = []
        relations = []

        # Generate entities from domains
        for domain_name in ontological_alignments.keys():
            entities.append(
                {
                    "name": domain_name.lower(),
                    "type": "domain_entity",
                    "properties": ["ontological", "formal"],
                }
            )

        # Generate relations from vector coherence
        coherence_score = merged_vectors.get("coherence_score", 0.0)
        if coherence_score > 0.5:
            relations.append(
                {
                    "type": "coherence_relation",
                    "description": "maintains structural coherence",
                    "strength": coherence_score,
                }
            )

        # Create natural language lambda expression
        if entities and relations:
            lambda_expr = f"λx.∃e∃r.(Entity(e) ∧ Relation(r) ∧ Coherent(e,r,x))"
        else:
            lambda_expr = "λx.NaturalLanguage(x)"

        return {
            "type": "natural_language_bridge",
            "lambda_expression": lambda_expr,
            "semantic_content": {
                "entities": entities,
                "relations": relations,
                "intentions": [{"description": "bridge formal to natural language"}],
            },
            "quantifiers": [
                {"type": "existential", "variable": "e"},
                {"type": "existential", "variable": "r"},
            ],
        }

    except Exception:
        return {
            "type": "natural_language_bridge",
            "lambda_expression": "λx.NaturalLanguage(x)",
            "semantic_content": {},
        }


def _create_domain_abstractions(
    ontological_alignments: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Create lambda expressions for domain-level abstractions"""
    try:
        abstractions = []

        # Group domains by ontological property
        property_groups = {}
        for domain_name, alignment_data in ontological_alignments.items():
            prop = alignment_data.get("property", "unknown")
            if prop not in property_groups:
                property_groups[prop] = []
            property_groups[prop].append(domain_name)

        # Create abstraction for each property group
        for ontological_prop, domains in property_groups.items():
            if len(domains) > 1:
                # Multiple domains share this property
                lambda_expr = f"λx.∃d∈{{{','.join(domains)}}}.(Domain(d) ∧ Property({ontological_prop}, d, x))"

                abstractions.append(
                    {
                        "type": "domain_abstraction",
                        "abstraction_level": "property_group",
                        "ontological_property": ontological_prop,
                        "domains": domains,
                        "lambda_expression": lambda_expr,
                        "semantic_content": {
                            "property": ontological_prop,
                            "domain_count": len(domains),
                            "abstraction_type": "property_based",
                        },
                    }
                )

        # Create general domain abstraction if no property groups
        if not abstractions:
            all_domains = list(ontological_alignments.keys())
            lambda_expr = f"λx.∃d.(Domain(d) ∧ ProcessedBy(d, x))"

            abstractions.append(
                {
                    "type": "domain_abstraction",
                    "abstraction_level": "general",
                    "domains": all_domains,
                    "lambda_expression": lambda_expr,
                    "semantic_content": {
                        "domain_count": len(all_domains),
                        "abstraction_type": "general",
                    },
                }
            )

        return abstractions

    except Exception:
        return []


def _normalize_expression_type(
    expr_type: str, expressions: List[Dict[str, Any]], context: Dict[str, Any]
) -> Dict[str, Any]:
    """Normalize expressions of a specific type"""
    try:
        if not expressions:
            return {
                "normalized_expressions": [],
                "reduction_steps": [],
                "optimization_applied": False,
            }

        normalized_exprs = []
        all_reduction_steps = []

        for expr in expressions:
            # Apply normalization based on expression type
            if expr_type == "vector_combinators":
                normalized_expr, reduction_steps = _normalize_combinator_expression(
                    expr
                )
            elif expr_type == "ontological_logic":
                normalized_expr, reduction_steps = _normalize_logical_expression(expr)
            elif expr_type == "natural_language_bridge":
                normalized_expr, reduction_steps = _normalize_natural_expression(expr)
            elif expr_type == "domain_abstractions":
                normalized_expr, reduction_steps = _normalize_abstraction_expression(
                    expr
                )
            else:
                normalized_expr, reduction_steps = _normalize_generic_expression(expr)

            normalized_exprs.append(normalized_expr)
            all_reduction_steps.extend(reduction_steps)

        # Apply optimization if beneficial
        optimization_applied = False
        if len(normalized_exprs) > 1:
            optimized_exprs = _optimize_expression_group(normalized_exprs, expr_type)
            if optimized_exprs != normalized_exprs:
                normalized_exprs = optimized_exprs
                optimization_applied = True

        return {
            "normalized_expressions": normalized_exprs,
            "reduction_steps": all_reduction_steps,
            "optimization_applied": optimization_applied,
            "expression_count": len(normalized_exprs),
        }

    except Exception as e:
        logger.warning(f"Expression normalization failed for {expr_type}: {e}")
        return {
            "normalized_expressions": [],
            "reduction_steps": [],
            "optimization_applied": False,
            "error": str(e),
        }


def _normalize_combinator_expression(
    expr: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Normalize combinator-based lambda expression"""
    try:
        lambda_expr = expr.get("lambda_expression", "λx.x")
        combinator_terms = expr.get("terms", ["I"])

        # Reduction steps for combinator simplification
        reduction_steps = []

        # Step 1: Combine adjacent K combinators
        simplified_terms = _combine_k_combinators(combinator_terms)
        if simplified_terms != combinator_terms:
            reduction_steps.append(
                {
                    "step": "K_combinator_reduction",
                    "before": combinator_terms,
                    "after": simplified_terms,
                    "rule": "K(K(x)) → K(x)",
                }
            )

        # Step 2: Apply S combinator rules if applicable
        further_simplified = _apply_s_combinator_rules(simplified_terms)
        if further_simplified != simplified_terms:
            reduction_steps.append(
                {
                    "step": "S_combinator_application",
                    "before": simplified_terms,
                    "after": further_simplified,
                    "rule": "S(K(x))(K(y)) → K(xy)",
                }
            )

        # Generate normalized expression
        if len(further_simplified) == 1 and further_simplified[0] == "I":
            normalized_lambda = "λx.x"
        else:
            normalized_lambda = f"λx.({' '.join(further_simplified)}) x"

        normalized_expr = expr.copy()
        normalized_expr["lambda_expression"] = normalized_lambda
        normalized_expr["terms"] = further_simplified
        normalized_expr["normalized"] = True

        return normalized_expr, reduction_steps

    except Exception:
        return expr, []


def _normalize_logical_expression(
    expr: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Normalize logical lambda expression"""
    try:
        lambda_expr = expr.get("lambda_expression", "λx.True(x)")
        predicates = expr.get("predicates", [])

        reduction_steps = []

        # Step 1: Remove redundant predicates
        unique_predicates = list(set(predicates))
        if len(unique_predicates) != len(predicates):
            reduction_steps.append(
                {
                    "step": "predicate_deduplication",
                    "before": predicates,
                    "after": unique_predicates,
                    "rule": "P ∧ P → P",
                }
            )

        # Step 2: Simplify quantifier scope
        simplified_expr = _simplify_quantifier_scope(lambda_expr)
        if simplified_expr != lambda_expr:
            reduction_steps.append(
                {
                    "step": "quantifier_simplification",
                    "before": lambda_expr,
                    "after": simplified_expr,
                    "rule": "∀x.(P(x) ∧ Q(x)) → ∀x.P(x) ∧ ∀x.Q(x)",
                }
            )

        normalized_expr = expr.copy()
        normalized_expr["lambda_expression"] = simplified_expr
        normalized_expr["predicates"] = unique_predicates
        normalized_expr["normalized"] = True

        return normalized_expr, reduction_steps

    except Exception:
        return expr, []


def _normalize_natural_expression(
    expr: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Normalize natural language bridge expression"""
    try:
        # Natural language expressions typically don't need complex reduction
        # Focus on semantic consistency and clarity

        lambda_expr = expr.get("lambda_expression", "λx.NaturalLanguage(x)")

        # Simple normalization: ensure proper quantifier binding
        normalized_expr = _ensure_proper_binding(lambda_expr)

        reduction_steps = []
        if normalized_expr != lambda_expr:
            reduction_steps.append(
                {
                    "step": "variable_binding_normalization",
                    "before": lambda_expr,
                    "after": normalized_expr,
                    "rule": "proper variable scoping",
                }
            )

        result_expr = expr.copy()
        result_expr["lambda_expression"] = normalized_expr
        result_expr["normalized"] = True

        return result_expr, reduction_steps

    except Exception:
        return expr, []


def _normalize_abstraction_expression(
    expr: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Normalize domain abstraction expression"""
    try:
        lambda_expr = expr.get("lambda_expression", "λx.Domain(x)")
        domains = expr.get("domains", [])

        reduction_steps = []

        # Simplify domain sets if possible
        if len(domains) > 5:  # Too many domains, create abstraction
            simplified_expr = "λx.∃d.(AbstractDomain(d) ∧ ProcessedBy(d, x))"
            reduction_steps.append(
                {
                    "step": "domain_abstraction",
                    "before": lambda_expr,
                    "after": simplified_expr,
                    "rule": "large domain set abstraction",
                }
            )
            lambda_expr = simplified_expr

        normalized_expr = expr.copy()
        normalized_expr["lambda_expression"] = lambda_expr
        normalized_expr["normalized"] = True

        return normalized_expr, reduction_steps

    except Exception:
        return expr, []


def _normalize_generic_expression(
    expr: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Normalize generic lambda expression"""
    try:
        lambda_expr = expr.get("lambda_expression", "λx.x")

        # Basic alpha-conversion and beta-reduction
        normalized_expr = _apply_basic_lambda_rules(lambda_expr)

        reduction_steps = []
        if normalized_expr != lambda_expr:
            reduction_steps.append(
                {
                    "step": "basic_lambda_normalization",
                    "before": lambda_expr,
                    "after": normalized_expr,
                    "rule": "alpha/beta reduction",
                }
            )

        result_expr = expr.copy()
        result_expr["lambda_expression"] = normalized_expr
        result_expr["normalized"] = True

        return result_expr, reduction_steps

    except Exception:
        return expr, []


def _combine_k_combinators(terms: List[str]) -> List[str]:
    """Combine adjacent K combinators"""
    try:
        if len(terms) <= 1:
            return terms

        result = []
        i = 0
        while i < len(terms):
            current = terms[i]
            # Look for K combinator patterns
            if (
                current.startswith("K(")
                and i + 1 < len(terms)
                and terms[i + 1].startswith("K(")
            ):
                # Combine K(x) K(y) → K(x) (K combinator property)
                result.append(current)
                i += 2  # Skip the next K combinator
            else:
                result.append(current)
                i += 1

        return result

    except Exception:
        return terms


def _apply_s_combinator_rules(terms: List[str]) -> List[str]:
    """Apply S combinator reduction rules"""
    try:
        # Simple S combinator rules
        # This is a simplified version - full combinator calculus is complex

        result = terms.copy()

        # Look for S K K pattern which reduces to I
        for i in range(len(result) - 2):
            if (
                result[i].startswith("S(")
                and result[i + 1].startswith("K(")
                and result[i + 2].startswith("K(")
            ):
                # S K K → I
                result = result[:i] + ["I"] + result[i + 3 :]
                break

        return result

    except Exception:
        return terms


def _simplify_quantifier_scope(lambda_expr: str) -> str:
    """Simplify quantifier scope in logical expressions"""
    try:
        # Basic quantifier simplification patterns
        # This is a simplified version of quantifier manipulation

        expr = lambda_expr

        # Pattern: ∀x.(P(x) ∧ Q(x)) can sometimes be simplified
        # For now, just clean up whitespace and normalize symbols
        expr = re.sub(r"\s+", " ", expr)
        expr = expr.replace("∧", " ∧ ").replace("∨", " ∨ ")
        expr = re.sub(r"\s+", " ", expr)

        return expr.strip()

    except Exception:
        return lambda_expr


def _ensure_proper_binding(lambda_expr: str) -> str:
    """Ensure proper variable binding in lambda expressions"""
    try:
        # Basic variable binding checks
        # This is a simplified version - full lambda calculus binding is complex

        expr = lambda_expr

        # Ensure lambda variables are properly scoped
        # For now, just normalize the expression format
        if not expr.startswith("λ"):
            expr = "λx." + expr

        return expr

    except Exception:
        return lambda_expr


def _apply_basic_lambda_rules(lambda_expr: str) -> str:
    """Apply basic lambda calculus reduction rules"""
    try:
        expr = lambda_expr

        # Beta reduction: (λx.M) N → M[x:=N]
        # Alpha conversion: λx.M → λy.M[x:=y] if y is fresh
        # For now, just normalize basic patterns

        # Normalize identity function
        if "λx.x" in expr or "λy.y" in expr:
            expr = "λx.x"

        # Clean up expression format
        expr = re.sub(r"\s+", " ", expr)

        return expr.strip()

    except Exception:
        return lambda_expr


def _optimize_expression_group(
    expressions: List[Dict[str, Any]], expr_type: str
) -> List[Dict[str, Any]]:
    """Optimize a group of expressions of the same type"""
    try:
        if len(expressions) <= 1:
            return expressions

        # Type-specific optimizations
        if expr_type == "vector_combinators":
            return _optimize_combinator_group(expressions)
        elif expr_type == "ontological_logic":
            return _optimize_logical_group(expressions)
        elif expr_type == "domain_abstractions":
            return _optimize_abstraction_group(expressions)
        else:
            return expressions

    except Exception:
        return expressions


def _optimize_combinator_group(
    expressions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Optimize group of combinator expressions"""
    try:
        # Combine compatible combinator expressions
        if len(expressions) == 1:
            return expressions

        # For simplicity, merge similar combinator terms
        merged_terms = []
        for expr in expressions:
            terms = expr.get("terms", [])
            merged_terms.extend(terms)

        # Remove duplicates while preserving order
        unique_terms = []
        for term in merged_terms:
            if term not in unique_terms:
                unique_terms.append(term)

        # Create optimized expression
        optimized_expr = expressions[0].copy()
        optimized_expr["terms"] = unique_terms
        optimized_expr["lambda_expression"] = f"λx.({' '.join(unique_terms)}) x"
        optimized_expr["optimized"] = True

        return [optimized_expr]

    except Exception:
        return expressions


def _optimize_logical_group(expressions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Optimize group of logical expressions"""
    try:
        # Combine predicates from multiple logical expressions
        all_predicates = []
        all_quantifiers = []

        for expr in expressions:
            predicates = expr.get("predicates", [])
            quantifiers = expr.get("quantifiers", [])
            all_predicates.extend(predicates)
            all_quantifiers.extend(quantifiers)

        # Remove duplicate predicates
        unique_predicates = list(set(all_predicates))

        # Create optimized combined expression
        if len(unique_predicates) > 0:
            combined_lambda = f"λx.∀y.({' ∧ '.join(unique_predicates)})(x,y)"
        else:
            combined_lambda = "λx.True(x)"

        optimized_expr = expressions[0].copy()
        optimized_expr["lambda_expression"] = combined_lambda
        optimized_expr["predicates"] = unique_predicates
        optimized_expr["quantifiers"] = all_quantifiers
        optimized_expr["optimized"] = True

        return [optimized_expr]

    except Exception:
        return expressions


def _optimize_abstraction_group(
    expressions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Optimize group of abstraction expressions"""
    try:
        # Combine domain abstractions into higher-level abstraction
        all_domains = []
        abstraction_levels = []

        for expr in expressions:
            domains = expr.get("domains", [])
            level = expr.get("abstraction_level", "general")
            all_domains.extend(domains)
            abstraction_levels.append(level)

        # Create unified abstraction
        unique_domains = list(set(all_domains))

        if len(unique_domains) > 8:  # Many domains, create high-level abstraction
            abstraction_lambda = "λx.∃d.(HighLevelAbstraction(d) ∧ ProcessedBy(d, x))"
            level = "high_level"
        else:
            domain_set = "{" + ",".join(unique_domains) + "}"
            abstraction_lambda = f"λx.∃d∈{domain_set}.(Domain(d) ∧ ProcessedBy(d, x))"
            level = "unified"

        optimized_expr = expressions[0].copy()
        optimized_expr["lambda_expression"] = abstraction_lambda
        optimized_expr["domains"] = unique_domains
        optimized_expr["abstraction_level"] = level
        optimized_expr["optimized"] = True

        return [optimized_expr]

    except Exception:
        return expressions


def _optimize_structures(
    normalized_structures: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Apply global optimization across all normalized structures"""
    try:
        # Cross-structure optimization opportunities
        optimized = normalized_structures.copy()

        # Count total expressions across all types
        total_expressions = sum(
            len(struct.get("normalized_expressions", []))
            for struct in normalized_structures.values()
        )

        # If too many expressions, apply higher-level optimization
        if total_expressions > 20:
            # Create meta-abstraction
            meta_expr = {
                "type": "meta_abstraction",
                "lambda_expression": "λx.∃T∈{vector,ontological,natural,domain}.(Type(T) ∧ ProcessedBy(T, x))",
                "semantic_content": {
                    "abstraction_type": "meta_level",
                    "total_expressions": total_expressions,
                    "structure_types": list(normalized_structures.keys()),
                },
            }

            # Add meta-abstraction to structures
            optimized["meta_abstraction"] = {
                "normalized_expressions": [meta_expr],
                "reduction_steps": [],
                "optimization_applied": True,
                "meta_level": True,
            }

        return optimized

    except Exception:
        return normalized_structures


def _calculate_complexity_metrics(
    normalization_context: Dict[str, Any],
) -> Dict[str, float]:
    """Calculate complexity and quality metrics for normalization"""
    try:
        normalized_structures = normalization_context.get("normalized_structures", {})
        lambda_expressions = normalization_context.get("lambda_expressions", {})

        # Expression count metrics
        total_expressions = sum(
            len(struct.get("normalized_expressions", []))
            for struct in normalized_structures.values()
        )

        original_expressions = sum(len(exprs) for exprs in lambda_expressions.values())

        # Reduction efficiency
        if original_expressions > 0:
            reduction_efficiency = 1.0 - (total_expressions / original_expressions)
        else:
            reduction_efficiency = 0.0

        # Structural coherence (based on optimization success)
        optimization_count = sum(
            1
            for struct in normalized_structures.values()
            if struct.get("optimization_applied", False)
        )

        if len(normalized_structures) > 0:
            structural_coherence = optimization_count / len(normalized_structures)
        else:
            structural_coherence = 0.0

        # Overall complexity (inverse of simplification success)
        if total_expressions == 0:
            overall_complexity = 0.0
        elif total_expressions <= 5:
            overall_complexity = 0.2
        elif total_expressions <= 15:
            overall_complexity = 0.5
        else:
            overall_complexity = 0.8

        # Optimization quality
        reduction_steps_total = sum(
            len(struct.get("reduction_steps", []))
            for struct in normalized_structures.values()
        )

        optimization_quality = min(reduction_steps_total / 10.0, 1.0)

        return {
            "overall_complexity": overall_complexity,
            "reduction_efficiency": max(0.0, reduction_efficiency),
            "structural_coherence": structural_coherence,
            "optimization_quality": optimization_quality,
            "total_expressions": total_expressions,
            "reduction_steps": reduction_steps_total,
        }

    except Exception:
        return {
            "overall_complexity": 0.5,
            "reduction_efficiency": 0.0,
            "structural_coherence": 0.0,
            "optimization_quality": 0.0,
            "total_expressions": 0,
            "reduction_steps": 0,
        }
