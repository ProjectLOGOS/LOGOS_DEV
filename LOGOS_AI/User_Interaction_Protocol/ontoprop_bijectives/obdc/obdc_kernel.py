"""
OBDC Kernel - UIP Step 2 IEL Ontological Synthesis Gateway

Output Buffer and Data Communication kernel for tokenization and streaming
of processed IEL synthesis results. Handles final output formatting and 
communication protocols for Step 2 → Step 3 transition.
"""

import logging
from typing import Dict, List, Any, Optional, Generator, Iterator
import json
import time
from datetime import datetime
from enum import Enum

logger = logging.getLogger("IEL_ONTO_KIT")


class TokenType(Enum):
    """Token types for OBDC output stream"""
    HEADER = "header"
    CONTENT = "content"
    METADATA = "metadata"
    STRUCTURE = "structure"
    SEMANTIC = "semantic"
    FOOTER = "footer"
    ERROR = "error"


def emit_tokens(translated_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Emit tokenized output stream from translated IEL synthesis data
    
    Args:
        translated_data: Translated natural language data from translation_engine
        
    Returns:
        Dict containing tokenized output stream ready for Step 3 processing
    """
    try:
        logger.info("Starting OBDC token emission")
        
        # Extract translated payload
        payload = translated_data.get("payload", {})
        unified_nl = payload.get("unified_natural_language", "")
        structured_translations = payload.get("structured_translations", {})
        translation_metadata = payload.get("translation_metadata", {})
        
        # Initialize OBDC emission context
        emission_context = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": _generate_session_id(),
            "token_stream": [],
            "buffer_segments": {},
            "communication_protocol": "IEL_STEP2_TO_STEP3",
            "output_format": "structured_tokens"
        }
        
        # Generate token stream
        token_stream = _generate_token_stream(
            unified_nl, 
            structured_translations, 
            translation_metadata,
            emission_context
        )
        
        # Buffer management
        buffer_segments = _create_buffer_segments(token_stream, translation_metadata)
        
        # Communication protocol setup
        protocol_headers = _generate_protocol_headers(emission_context, translation_metadata)
        
        # Stream optimization
        optimized_stream = _optimize_token_stream(token_stream, buffer_segments)
        
        # Quality validation
        validation_results = _validate_token_stream(optimized_stream)
        
        # Final emission package
        emission_package = {
            "token_stream": optimized_stream,
            "buffer_segments": buffer_segments,
            "protocol_headers": protocol_headers,
            "validation_results": validation_results,
            "emission_metadata": {
                "total_tokens": len(optimized_stream),
                "buffer_count": len(buffer_segments),
                "protocol_version": "IEL_OBDC_v2.0",
                "quality_score": validation_results.get("quality_score", 0.0),
                "emission_timestamp": emission_context["timestamp"]
            }
        }
        
        logger.info(f"OBDC emission completed: {len(optimized_stream)} tokens, "
                   f"quality={validation_results.get('quality_score', 0.0):.3f}")
        
        return {
            "status": "ok",
            "payload": emission_package,
            "metadata": {
                "stage": "obdc_emission",
                "tokens_emitted": len(optimized_stream),
                "step3_ready": True
            }
        }
        
    except Exception as e:
        logger.error(f"OBDC token emission failed: {e}")
        raise


def _generate_session_id() -> str:
    """Generate unique session ID for OBDC emission"""
    timestamp = int(time.time() * 1000)
    return f"OBDC_{timestamp}"


def _generate_token_stream(unified_nl: str, 
                          structured_translations: Dict[str, Any],
                          translation_metadata: Dict[str, Any],
                          context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate structured token stream from translation results"""
    try:
        tokens = []
        
        # Header token
        header_token = _create_token(
            TokenType.HEADER,
            {
                "session_id": context["session_id"],
                "protocol": context["communication_protocol"],
                "content_type": "IEL_synthesis",
                "format_version": "2.0"
            }
        )
        tokens.append(header_token)
        
        # Metadata tokens
        metadata_token = _create_token(
            TokenType.METADATA,
            {
                "source_complexity": translation_metadata.get("source_complexity", 0.0),
                "translation_mode": translation_metadata.get("translation_mode", "standard"),
                "quality_metrics": translation_metadata.get("quality_metrics", {}),
                "processing_timestamp": translation_metadata.get("processing_timestamp", "")
            }
        )
        tokens.append(metadata_token)
        
        # Content tokens from unified natural language
        content_tokens = _tokenize_unified_content(unified_nl)
        tokens.extend(content_tokens)
        
        # Structure tokens from structured translations
        structure_tokens = _create_structure_tokens(structured_translations)
        tokens.extend(structure_tokens)
        
        # Semantic tokens for ontological mappings
        semantic_tokens = _create_semantic_tokens(structured_translations)
        tokens.extend(semantic_tokens)
        
        # Footer token
        footer_token = _create_token(
            TokenType.FOOTER,
            {
                "session_id": context["session_id"],
                "total_content_tokens": len(content_tokens),
                "total_structure_tokens": len(structure_tokens),
                "total_semantic_tokens": len(semantic_tokens),
                "completion_timestamp": datetime.utcnow().isoformat()
            }
        )
        tokens.append(footer_token)
        
        return tokens
        
    except Exception as e:
        logger.warning(f"Token stream generation failed: {e}")
        # Return minimal error stream
        return [_create_token(TokenType.ERROR, {"error": str(e)})]


def _create_token(token_type: TokenType, data: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized token structure"""
    return {
        "token_id": f"{token_type.value}_{int(time.time() * 1000000)}",
        "token_type": token_type.value,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
        "size": len(json.dumps(data)) if data else 0
    }


def _tokenize_unified_content(unified_nl: str) -> List[Dict[str, Any]]:
    """Tokenize unified natural language content"""
    try:
        tokens = []
        
        if not unified_nl or unified_nl.strip() == "":
            return tokens
        
        # Split content into semantic chunks
        sentences = _split_into_sentences(unified_nl)
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                content_token = _create_token(
                    TokenType.CONTENT,
                    {
                        "sequence_id": i,
                        "content": sentence.strip(),
                        "content_type": "natural_language",
                        "word_count": len(sentence.split()),
                        "char_count": len(sentence)
                    }
                )
                tokens.append(content_token)
        
        return tokens
        
    except Exception:
        return []


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for tokenization"""
    try:
        # Simple sentence splitting (could be enhanced with NLP library)
        import re
        
        # Split on sentence ending punctuation
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:  # Non-empty after stripping
                # Ensure sentence ends with period if it doesn't have punctuation
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
        
    except Exception:
        # Fallback: return original text as single sentence
        return [text] if text else []


def _create_structure_tokens(structured_translations: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create tokens from structured translation data"""
    try:
        tokens = []
        
        for structure_type, translation_data in structured_translations.items():
            structure_token = _create_token(
                TokenType.STRUCTURE,
                {
                    "structure_type": structure_type,
                    "natural_language_text": translation_data.get("natural_language_text", ""),
                    "structural_description": translation_data.get("structural_description", ""),
                    "readability_metrics": translation_data.get("readability_metrics", {}),
                    "translation_confidence": translation_data.get("translation_confidence", 0.0)
                }
            )
            tokens.append(structure_token)
        
        return tokens
        
    except Exception:
        return []


def _create_semantic_tokens(structured_translations: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create semantic tokens for ontological information"""
    try:
        tokens = []
        
        # Extract semantic information from structured translations
        for structure_type, translation_data in structured_translations.items():
            nl_text = translation_data.get("natural_language_text", "")
            
            if nl_text and not nl_text.startswith("[Translation unavailable"):
                # Create semantic token for meaningful content
                semantic_token = _create_token(
                    TokenType.SEMANTIC,
                    {
                        "source_structure": structure_type,
                        "semantic_content": nl_text,
                        "ontological_relevance": _assess_ontological_relevance(structure_type),
                        "conceptual_weight": _calculate_conceptual_weight(translation_data),
                        "iel_alignment": _extract_iel_alignment_info(structure_type)
                    }
                )
                tokens.append(semantic_token)
        
        return tokens
        
    except Exception:
        return []


def _assess_ontological_relevance(structure_type: str) -> float:
    """Assess ontological relevance of structure type"""
    try:
        relevance_weights = {
            "vector_combinators": 0.3,
            "ontological_logic": 0.9,
            "natural_language_bridge": 0.7,
            "domain_abstractions": 0.8
        }
        
        return relevance_weights.get(structure_type, 0.5)
        
    except Exception:
        return 0.5


def _calculate_conceptual_weight(translation_data: Dict[str, Any]) -> float:
    """Calculate conceptual weight of translation data"""
    try:
        confidence = translation_data.get("translation_confidence", 0.0)
        readability = translation_data.get("readability_metrics", {}).get("score", 0.0)
        
        # Combine confidence and readability
        conceptual_weight = (confidence * 0.6 + readability * 0.4)
        
        return min(max(conceptual_weight, 0.0), 1.0)
        
    except Exception:
        return 0.0


def _extract_iel_alignment_info(structure_type: str) -> Dict[str, Any]:
    """Extract IEL alignment information for structure type"""
    try:
        alignment_info = {
            "structure_type": structure_type,
            "iel_domains": [],
            "ontological_mapping": "unknown"
        }
        
        # Map structure types to IEL domain categories
        if structure_type == "ontological_logic":
            alignment_info["iel_domains"] = ["ModalPraxis", "GnosiPraxis"]
            alignment_info["ontological_mapping"] = "logical_structures"
        elif structure_type == "domain_abstractions":
            alignment_info["iel_domains"] = ["ThemiPraxis", "DynaPraxis"]
            alignment_info["ontological_mapping"] = "domain_hierarchies"
        elif structure_type == "natural_language_bridge":
            alignment_info["iel_domains"] = ["HexiPraxis"]
            alignment_info["ontological_mapping"] = "linguistic_bridges"
        elif structure_type == "vector_combinators":
            alignment_info["iel_domains"] = ["ChremaPraxis"]
            alignment_info["ontological_mapping"] = "mathematical_structures"
        
        return alignment_info
        
    except Exception:
        return {"structure_type": structure_type, "iel_domains": [], "ontological_mapping": "unknown"}


def _create_buffer_segments(token_stream: List[Dict[str, Any]], 
                           translation_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Create buffer segments for stream optimization"""
    try:
        segments = {
            "header_segment": [],
            "content_segment": [],
            "structure_segment": [],
            "semantic_segment": [],
            "metadata_segment": [],
            "footer_segment": []
        }
        
        # Categorize tokens into segments
        for token in token_stream:
            token_type = token.get("token_type", "")
            
            if token_type == "header":
                segments["header_segment"].append(token)
            elif token_type == "content":
                segments["content_segment"].append(token)
            elif token_type == "structure":
                segments["structure_segment"].append(token)
            elif token_type == "semantic":
                segments["semantic_segment"].append(token)
            elif token_type == "metadata":
                segments["metadata_segment"].append(token)
            elif token_type == "footer":
                segments["footer_segment"].append(token)
        
        # Add segment metadata
        for segment_name, tokens in segments.items():
            segments[f"{segment_name}_info"] = {
                "token_count": len(tokens),
                "total_size": sum(token.get("size", 0) for token in tokens),
                "priority": _get_segment_priority(segment_name)
            }
        
        return segments
        
    except Exception:
        return {}


def _get_segment_priority(segment_name: str) -> int:
    """Get processing priority for buffer segment"""
    priorities = {
        "header_segment": 1,      # Highest priority
        "metadata_segment": 2,
        "semantic_segment": 3,
        "structure_segment": 4,
        "content_segment": 5,
        "footer_segment": 6       # Lowest priority
    }
    
    return priorities.get(segment_name.replace("_info", ""), 5)


def _generate_protocol_headers(emission_context: Dict[str, Any], 
                             translation_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate communication protocol headers"""
    try:
        headers = {
            "protocol_version": "IEL_OBDC_v2.0",
            "communication_type": "STEP2_TO_STEP3",
            "session_id": emission_context.get("session_id", ""),
            "timestamp": emission_context.get("timestamp", ""),
            "content_type": "application/iel-synthesis+json",
            "encoding": "utf-8",
            "compression": "none",
            "authentication": {
                "method": "internal_step_transition",
                "credentials": "UIP_STEP_AUTHORIZATION"
            },
            "routing": {
                "source": "UIP_STEP_2_IEL_SYNTHESIS",
                "destination": "UIP_STEP_3_IEL_OVERLAY",
                "priority": "high",
                "delivery_mode": "guaranteed"
            },
            "quality_of_service": {
                "reliability": "high",
                "latency": "low",
                "throughput": "standard"
            },
            "content_metadata": {
                "source_complexity": translation_metadata.get("source_complexity", 0.0),
                "translation_quality": translation_metadata.get("quality_metrics", {}).get("overall_quality", 0.0),
                "processing_mode": translation_metadata.get("translation_mode", "standard")
            }
        }
        
        return headers
        
    except Exception:
        return {
            "protocol_version": "IEL_OBDC_v2.0",
            "communication_type": "STEP2_TO_STEP3",
            "content_type": "application/iel-synthesis+json"
        }


def _optimize_token_stream(token_stream: List[Dict[str, Any]], 
                          buffer_segments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Optimize token stream for efficient processing"""
    try:
        if not token_stream:
            return token_stream
        
        optimized_stream = []
        
        # Priority-based ordering
        segment_priorities = [
            ("header_segment", 1),
            ("metadata_segment", 2),
            ("semantic_segment", 3),
            ("structure_segment", 4),
            ("content_segment", 5),
            ("footer_segment", 6)
        ]
        
        # Process segments in priority order
        for segment_name, priority in segment_priorities:
            segment_tokens = buffer_segments.get(segment_name, [])
            
            # Sort tokens within segment by timestamp
            sorted_tokens = sorted(segment_tokens, key=lambda t: t.get("timestamp", ""))
            
            # Add optimized tokens to stream
            optimized_stream.extend(sorted_tokens)
        
        # Add sequence numbers for stream integrity
        for i, token in enumerate(optimized_stream):
            token["sequence_number"] = i
            token["stream_position"] = i / len(optimized_stream) if optimized_stream else 0.0
        
        return optimized_stream
        
    except Exception:
        return token_stream


def _validate_token_stream(token_stream: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate token stream quality and completeness"""
    try:
        validation_results = {
            "is_valid": True,
            "quality_score": 0.0,
            "completeness_check": {},
            "integrity_check": {},
            "format_validation": {},
            "issues": []
        }
        
        if not token_stream:
            validation_results["is_valid"] = False
            validation_results["issues"].append("Empty token stream")
            return validation_results
        
        # Completeness check
        token_types = [token.get("token_type", "") for token in token_stream]
        required_types = ["header", "metadata", "footer"]
        
        completeness_score = 0.0
        for req_type in required_types:
            if req_type in token_types:
                completeness_score += 1.0
        
        completeness_score /= len(required_types)
        validation_results["completeness_check"] = {
            "score": completeness_score,
            "required_types": required_types,
            "found_types": list(set(token_types)),
            "missing_types": [t for t in required_types if t not in token_types]
        }
        
        # Integrity check
        sequence_numbers = [token.get("sequence_number", -1) for token in token_stream]
        expected_sequence = list(range(len(token_stream)))
        
        integrity_score = 1.0 if sequence_numbers == expected_sequence else 0.0
        validation_results["integrity_check"] = {
            "score": integrity_score,
            "sequence_valid": sequence_numbers == expected_sequence,
            "token_count": len(token_stream)
        }
        
        # Format validation
        format_issues = []
        for i, token in enumerate(token_stream):
            if not isinstance(token.get("data"), dict):
                format_issues.append(f"Token {i}: invalid data format")
            if not token.get("token_id"):
                format_issues.append(f"Token {i}: missing token_id")
            if not token.get("timestamp"):
                format_issues.append(f"Token {i}: missing timestamp")
        
        format_score = max(0.0, 1.0 - len(format_issues) / len(token_stream))
        validation_results["format_validation"] = {
            "score": format_score,
            "issues": format_issues
        }
        
        # Overall quality score
        validation_results["quality_score"] = (
            completeness_score * 0.4 +
            integrity_score * 0.4 +
            format_score * 0.2
        )
        
        # Overall validity
        validation_results["is_valid"] = (
            validation_results["quality_score"] >= 0.7 and
            len(format_issues) == 0
        )
        
        return validation_results
        
    except Exception as e:
        return {
            "is_valid": False,
            "quality_score": 0.0,
            "issues": [f"Validation error: {str(e)}"]
        }