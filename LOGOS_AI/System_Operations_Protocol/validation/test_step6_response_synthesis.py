"""
UIP Step 6 — Resolution & Response Synthesis Validation Tests
===========================================================

Comprehensive test suite for the UIP Step 6 response synthesis system.
Tests response generation, conflict resolution, format selection, and 
integration with Trinity + Adaptive Inference systems.

Tests follow PXL ⧟⇌⟹⩪ constraint validation.
"""

import pytest
import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Import UIP Step 6 components
try:
    from ..protocols.user_interaction.response_formatter import (
        ResponseSynthesizer,
        ResponseFormat,
        SynthesisMethod, 
        SynthesizedResponse,
        ConflictResolver,
        ResponseFormatSelector
    )
    RESPONSE_FORMATTER_AVAILABLE = True
except ImportError:
    RESPONSE_FORMATTER_AVAILABLE = False
    pytest.skip("Response formatter not available", allow_module_level=True)

# Mock data for testing
@dataclass
class MockAdaptiveProfile:
    """Mock adaptive profile for testing."""
    confidence_level: float
    learning_context: Dict[str, Any]
    temporal_state: Dict[str, Any]
    adaptation_metrics: Dict[str, Any]

@dataclass 
class MockTrinityVector:
    """Mock Trinity integration results."""
    trinity_coherence: float
    trinity_components: Dict[str, Any]
    integration_confidence: float
    validation_status: str

@dataclass
class MockIELBundle:
    """Mock IEL reasoning results."""
    modal_analysis: Dict[str, Any]
    empirical_evidence: Dict[str, Any]
    temporal_context: Dict[str, Any]
    confidence_metrics: Dict[str, Any]

class TestResponseSynthesizer:
    """Test cases for the ResponseSynthesizer orchestrator."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.synthesizer = ResponseSynthesizer()
        
        # Create mock input data
        self.mock_adaptive_profile = {
            "confidence_level": 0.85,
            "learning_context": {"domain": "philosophical", "complexity": "high"},
            "temporal_state": {"phase": "analysis", "duration": 120},
            "adaptation_metrics": {"coherence": 0.9, "consistency": 0.87}
        }
        
        self.mock_trinity_vector = {
            "trinity_coherence": 0.92,
            "trinity_components": {
                "thesis": {"strength": 0.88, "content": "Base proposition"},
                "antithesis": {"strength": 0.85, "content": "Counter-argument"},
                "synthesis": {"strength": 0.91, "content": "Resolved position"}
            },
            "integration_confidence": 0.89,
            "validation_status": "VALIDATED"
        }
        
        self.mock_iel_bundle = {
            "modal_analysis": {"necessity": 0.83, "possibility": 0.95, "contingency": 0.72},
            "empirical_evidence": {"observational_support": 0.78, "experimental_validation": 0.82},
            "temporal_context": {"past_relevance": 0.75, "future_implications": 0.88},
            "confidence_metrics": {"overall_confidence": 0.84, "uncertainty_bounds": [0.78, 0.90]}
        }

    def test_synthesis_initialization(self):
        """Test that ResponseSynthesizer initializes correctly."""
        assert self.synthesizer is not None
        assert hasattr(self.synthesizer, 'conflict_resolver')
        assert hasattr(self.synthesizer, 'format_selector')
        assert isinstance(self.synthesizer.conflict_resolver, ConflictResolver)
        assert isinstance(self.synthesizer.format_selector, ResponseFormatSelector)

    def test_trinity_weighted_synthesis(self):
        """Test Trinity-weighted synthesis method."""
        result = self.synthesizer.synthesize_response(
            adaptive_profile=self.mock_adaptive_profile,
            trinity_vector=self.mock_trinity_vector,
            iel_bundle=self.mock_iel_bundle,
            synthesis_method=SynthesisMethod.TRINITY_WEIGHTED
        )
        
        assert isinstance(result, SynthesizedResponse)
        assert result.synthesis_method == SynthesisMethod.TRINITY_WEIGHTED
        assert result.confidence_score >= 0.0 and result.confidence_score <= 1.0
        assert len(result.supporting_evidence) > 0
        assert "Trinity-weighted" in result.synthesis_rationale

    def test_confidence_based_synthesis(self):
        """Test confidence-based synthesis method."""
        result = self.synthesizer.synthesize_response(
            adaptive_profile=self.mock_adaptive_profile,
            trinity_vector=self.mock_trinity_vector,
            iel_bundle=self.mock_iel_bundle,
            synthesis_method=SynthesisMethod.CONFIDENCE_BASED
        )
        
        assert isinstance(result, SynthesizedResponse)
        assert result.synthesis_method == SynthesisMethod.CONFIDENCE_BASED
        assert "confidence" in result.synthesis_rationale.lower()
        assert result.confidence_score > 0.5  # Should be reasonable confidence

    def test_consensus_driven_synthesis(self):
        """Test consensus-driven synthesis method."""
        result = self.synthesizer.synthesize_response(
            adaptive_profile=self.mock_adaptive_profile,
            trinity_vector=self.mock_trinity_vector,
            iel_bundle=self.mock_iel_bundle,
            synthesis_method=SynthesisMethod.CONSENSUS_DRIVEN
        )
        
        assert isinstance(result, SynthesizedResponse)
        assert result.synthesis_method == SynthesisMethod.CONSENSUS_DRIVEN
        assert "consensus" in result.synthesis_rationale.lower()
        assert len(result.supporting_evidence) >= 2  # Should show multiple sources

    def test_hierarchical_synthesis(self):
        """Test hierarchical synthesis method."""
        result = self.synthesizer.synthesize_response(
            adaptive_profile=self.mock_adaptive_profile,
            trinity_vector=self.mock_trinity_vector,
            iel_bundle=self.mock_iel_bundle,
            synthesis_method=SynthesisMethod.HIERARCHICAL
        )
        
        assert isinstance(result, SynthesizedResponse)
        assert result.synthesis_method == SynthesisMethod.HIERARCHICAL
        assert "hierarchical" in result.synthesis_rationale.lower()
        assert hasattr(result, 'temporal_context')

    def test_adaptive_priority_synthesis(self):
        """Test adaptive priority synthesis method."""
        result = self.synthesizer.synthesize_response(
            adaptive_profile=self.mock_adaptive_profile,
            trinity_vector=self.mock_trinity_vector,
            iel_bundle=self.mock_iel_bundle,
            synthesis_method=SynthesisMethod.ADAPTIVE_PRIORITY
        )
        
        assert isinstance(result, SynthesizedResponse)
        assert result.synthesis_method == SynthesisMethod.ADAPTIVE_PRIORITY
        assert "adaptive" in result.synthesis_rationale.lower()
        assert result.confidence_score >= self.mock_adaptive_profile["confidence_level"] * 0.8

    def test_conflict_detection_and_resolution(self):
        """Test conflict detection and resolution system."""
        # Create conflicting evidence
        conflicting_iel = self.mock_iel_bundle.copy()
        conflicting_iel["modal_analysis"]["necessity"] = 0.2  # Low necessity conflicts with high confidence
        
        result = self.synthesizer.synthesize_response(
            adaptive_profile=self.mock_adaptive_profile,
            trinity_vector=self.mock_trinity_vector,
            iel_bundle=conflicting_iel,
            synthesis_method=SynthesisMethod.CONSENSUS_DRIVEN
        )
        
        # Should still produce valid result with conflict resolution
        assert isinstance(result, SynthesizedResponse)
        assert result.confidence_score < 0.9  # Should be lower due to conflicts
        assert len(result.conflicts_resolved) > 0

    def test_multiple_response_formats(self):
        """Test generation of multiple response formats."""
        base_result = self.synthesizer.synthesize_response(
            adaptive_profile=self.mock_adaptive_profile,
            trinity_vector=self.mock_trinity_vector,
            iel_bundle=self.mock_iel_bundle
        )
        
        # Test each response format
        for response_format in ResponseFormat:
            formatted_response = self.synthesizer.format_response(base_result, response_format)
            assert isinstance(formatted_response, dict)
            assert "content" in formatted_response
            assert formatted_response["format"] == response_format.value
            
            # Format-specific validations
            if response_format == ResponseFormat.STRUCTURED_JSON:
                assert "metadata" in formatted_response
                assert "confidence" in formatted_response["metadata"]
            elif response_format == ResponseFormat.FORMAL_PROOF:
                assert "proof_steps" in formatted_response
            elif response_format == ResponseFormat.TECHNICAL_REPORT:
                assert "executive_summary" in formatted_response

    def test_temporal_integration(self):
        """Test temporal context integration in synthesis."""
        # Add temporal context to inputs
        temporal_context = {
            "temporal_horizon": "30_days", 
            "causal_dependencies": ["past_event_1", "past_event_2"],
            "future_implications": ["predicted_outcome_1"]
        }
        
        result = self.synthesizer.synthesize_response(
            adaptive_profile=self.mock_adaptive_profile,
            trinity_vector=self.mock_trinity_vector,
            iel_bundle=self.mock_iel_bundle,
            temporal_context=temporal_context
        )
        
        assert hasattr(result, 'temporal_context')
        assert result.temporal_context is not None
        assert "temporal" in str(result.temporal_context).lower()

    def test_pxl_constraint_compliance(self):
        """Test PXL ⧟⇌⟹⩪ constraint validation throughout synthesis."""
        result = self.synthesizer.synthesize_response(
            adaptive_profile=self.mock_adaptive_profile,
            trinity_vector=self.mock_trinity_vector,
            iel_bundle=self.mock_iel_bundle
        )
        
        # Verify PXL constraint symbols are properly handled
        assert hasattr(result, 'pxl_constraints_validated')
        assert result.pxl_constraints_validated is True
        
        # Check for constraint validation in rationale
        constraint_symbols = ['⧟', '⇌', '⟹', '⩪']
        rationale_text = str(result.synthesis_rationale)
        # At least one constraint should be mentioned or validated
        assert any(symbol in rationale_text or "PXL" in rationale_text or "constraint" in rationale_text 
                  for symbol in constraint_symbols + ["PXL", "constraint"])

    def test_low_confidence_handling(self):
        """Test handling of low confidence scenarios."""
        low_confidence_adaptive = self.mock_adaptive_profile.copy()
        low_confidence_adaptive["confidence_level"] = 0.3
        
        low_confidence_trinity = self.mock_trinity_vector.copy()
        low_confidence_trinity["trinity_coherence"] = 0.4
        
        result = self.synthesizer.synthesize_response(
            adaptive_profile=low_confidence_adaptive,
            trinity_vector=low_confidence_trinity,
            iel_bundle=self.mock_iel_bundle
        )
        
        assert isinstance(result, SynthesizedResponse)
        assert result.confidence_score < 0.6  # Should reflect low input confidence
        assert len(result.uncertainty_qualifiers) > 0
        assert "uncertainty" in result.synthesis_rationale.lower() or "low confidence" in result.synthesis_rationale.lower()

    def test_response_quality_metrics(self):
        """Test response quality measurement and validation."""
        result = self.synthesizer.synthesize_response(
            adaptive_profile=self.mock_adaptive_profile,
            trinity_vector=self.mock_trinity_vector,
            iel_bundle=self.mock_iel_bundle
        )
        
        # Validate quality metrics
        assert hasattr(result, 'quality_metrics')
        quality_metrics = result.quality_metrics
        
        assert "coherence_score" in quality_metrics
        assert "completeness_score" in quality_metrics
        assert "reliability_score" in quality_metrics
        
        # All scores should be between 0 and 1
        for metric_name, score in quality_metrics.items():
            if isinstance(score, (int, float)):
                assert 0 <= score <= 1, f"Quality metric {metric_name} out of range: {score}"

    def test_synthesis_performance(self):
        """Test synthesis performance and resource usage."""
        import time
        
        start_time = time.time()
        
        result = self.synthesizer.synthesize_response(
            adaptive_profile=self.mock_adaptive_profile,
            trinity_vector=self.mock_trinity_vector,
            iel_bundle=self.mock_iel_bundle
        )
        
        end_time = time.time()
        synthesis_duration = end_time - start_time
        
        # Synthesis should complete within reasonable time
        assert synthesis_duration < 5.0, f"Synthesis took too long: {synthesis_duration}s"
        assert isinstance(result, SynthesizedResponse)

class TestConflictResolver:
    """Test cases for conflict resolution system."""
    
    def setup_method(self):
        """Set up conflict resolver tests."""
        self.conflict_resolver = ConflictResolver()

    def test_confidence_conflict_resolution(self):
        """Test resolution of confidence-based conflicts."""
        conflicting_inputs = [
            {"source": "trinity", "confidence": 0.9, "content": "High confidence conclusion"},
            {"source": "adaptive", "confidence": 0.3, "content": "Low confidence conclusion"},
            {"source": "iel", "confidence": 0.7, "content": "Medium confidence conclusion"}
        ]
        
        resolution = self.conflict_resolver.resolve_conflicts(conflicting_inputs)
        
        assert "resolved_content" in resolution
        assert "resolution_method" in resolution
        assert resolution["primary_source_confidence"] >= 0.7  # Should favor high confidence

    def test_temporal_conflict_resolution(self):
        """Test resolution of temporal ordering conflicts."""
        temporal_conflicts = [
            {"timestamp": datetime.now() - timedelta(hours=1), "content": "Earlier conclusion"},
            {"timestamp": datetime.now(), "content": "Latest conclusion"},
            {"timestamp": datetime.now() - timedelta(minutes=30), "content": "Middle conclusion"}
        ]
        
        resolution = self.conflict_resolver.resolve_temporal_conflicts(temporal_conflicts)
        
        assert "resolved_content" in resolution
        assert "temporal_ordering" in resolution
        assert len(resolution["temporal_ordering"]) == 3

class TestResponseFormatSelector:
    """Test cases for response format selection system."""
    
    def setup_method(self):
        """Set up format selector tests."""
        self.format_selector = ResponseFormatSelector()

    def test_context_based_format_selection(self):
        """Test format selection based on context."""
        technical_context = {"audience": "technical", "complexity": "high", "purpose": "analysis"}
        general_context = {"audience": "general", "complexity": "low", "purpose": "explanation"}
        
        technical_format = self.format_selector.select_optimal_format(
            context=technical_context,
            content_type="analysis"
        )
        
        general_format = self.format_selector.select_optimal_format(
            context=general_context, 
            content_type="explanation"
        )
        
        assert technical_format in [ResponseFormat.TECHNICAL_REPORT, ResponseFormat.STRUCTURED_JSON, ResponseFormat.FORMAL_PROOF]
        assert general_format in [ResponseFormat.NATURAL_LANGUAGE, ResponseFormat.CONVERSATIONAL]

    def test_content_type_format_mapping(self):
        """Test format selection based on content type."""
        proof_format = self.format_selector.select_optimal_format(
            context={"purpose": "verification"},
            content_type="proof"
        )
        
        assert proof_format == ResponseFormat.FORMAL_PROOF
        
        api_format = self.format_selector.select_optimal_format(
            context={"purpose": "integration"},
            content_type="api_response"
        )
        
        assert api_format == ResponseFormat.API_RESPONSE

if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])