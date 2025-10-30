"""
UIP Step 5 Adaptive Inference Layer Validation Tests

Minimal test suite to verify that the adaptive inference pipeline
executes successfully with all required components.

⧟ Identity constraint: Test reproducibility via deterministic inputs
⇌ Balance constraint: Test coverage vs execution time
⟹ Causal constraint: Test dependencies follow logical sequence  
⩪ Equivalence constraint: Test results consistent across runs
"""

import json
import os
import sys
import logging
from typing import Dict, Any

# Add LOGOS_V2 to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
logos_v2_dir = os.path.dirname(current_dir)
if logos_v2_dir not in sys.path:
    sys.path.insert(0, logos_v2_dir)

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_step5_executes_minimal_pipeline():
    """Test that Step 5 adaptive inference executes the minimal pipeline."""
    try:
        # Import the main function
        from protocols.user_interaction.adaptive_inference_layer import execute_adaptive_inference
        
        # Create test inputs
        trinity = {
            "existence": 0.8,
            "goodness": 0.7,
            "truth": 0.9,
            "metadata": {"test": "trinity_vector"},
            "reasoning": "Test reasoning content for semantic processing"
        }
        
        iel = {
            "frames": ["test_frame_1", "test_frame_2"],
            "group_coherence": {"coherence_score": 0.85},
            "reasoning_chains": [
                {"content": "Test reasoning chain 1", "keywords": ["test", "reasoning"]},
                {"content": "Test reasoning chain 2", "keywords": ["adaptive", "inference"]}
            ]
        }
        
        session = {
            "user": "test_user",
            "session_id": "test_session_001",
            "timestamp": "2025-01-28T12:00:00Z"
        }
        
        # Execute adaptive inference
        print("Executing adaptive inference pipeline...")
        result = execute_adaptive_inference(trinity, iel, session)
        
        # Verify required fields in output
        assert "meta" in result, "Missing 'meta' field in result"
        assert "step" in result["meta"], "Missing 'step' field in meta"
        assert result["meta"]["step"] == 5, f"Expected step 5, got {result['meta']['step']}"
        assert "status" in result["meta"], "Missing 'status' field in meta"
        
        assert "posterior" in result, "Missing 'posterior' field in result"
        assert "rl" in result, "Missing 'rl' field in result" 
        assert "improvement" in result, "Missing 'improvement' field in result"
        assert "drift" in result, "Missing 'drift' field in result"
        assert "embeddings" in result, "Missing 'embeddings' field in result"
        
        # Verify PXL constraints are reported
        if "pxl_constraints" in result["meta"]:
            constraints = result["meta"]["pxl_constraints"]
            assert "identity_hash" in constraints, "Missing identity_hash constraint"
            assert "coherence_target" in constraints, "Missing coherence_target constraint"
            assert "causal_updates" in constraints, "Missing causal_updates constraint"
            assert "embedding_consistency" in constraints, "Missing embedding_consistency constraint"
        
        print("✓ Step 5 pipeline executed successfully")
        print(f"✓ Result status: {result['meta']['status']}")
        print(f"✓ PXL constraints: {result['meta'].get('pxl_constraints', 'Not available')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_availability():
    """Test availability of all Step 5 components."""
    try:
        from protocols.user_interaction.adaptive_inference_layer import get_adaptive_inference_status
        
        status = get_adaptive_inference_status()
        
        print("Component Availability Status:")
        print(f"  Bayesian Inference: {status.get('bayesian_inference_available', False)}")
        print(f"  Semantic Transformers: {status.get('semantic_transformers_available', False)}")
        print(f"  Autonomous Learning: {status.get('autonomous_learning_available', False)}")
        print(f"  Self Improvement: {status.get('self_improvement_available', False)}")
        print(f"  Persistence: {status.get('persistence_available', False)}")
        print(f"  Audit Logging: {status.get('audit_logging_available', False)}")
        print(f"  System Status: {status.get('system_status', 'unknown')}")
        
        # System is operational if status is available
        assert status.get('system_status') in ['operational', 'degraded'], f"Invalid system status: {status.get('system_status')}"
        
        print("✓ Component availability test passed")
        return True
        
    except Exception as e:
        print(f"✗ Component availability test failed: {e}")
        return False

def test_individual_components():
    """Test individual component functions."""
    success_count = 0
    total_tests = 0
    
    # Test Bayesian inference
    total_tests += 1
    try:
        from intelligence.reasoning_engines.bayesian_inference import update_posteriors
        
        test_trinity = {"existence": 0.5, "goodness": 0.6, "truth": 0.7}
        test_iel = {"reasoning_chains": [{"keywords": ["test"]}]}
        
        result = update_posteriors(test_trinity, test_iel)
        assert "beliefs" in result, "Missing beliefs in Bayesian result"
        assert "confidence" in result, "Missing confidence in Bayesian result"
        assert "variance" in result, "Missing variance in Bayesian result"
        
        print("✓ Bayesian inference component test passed")
        success_count += 1
        
    except Exception as e:
        print(f"✗ Bayesian inference component test failed: {e}")
    
    # Test semantic transformers
    total_tests += 1
    try:
        from intelligence.reasoning_engines.semantic_transformers import encode_semantics, detect_concept_drift
        
        test_trinity = {"reasoning": "test content"}
        test_iel = {"reasoning_chains": [{"content": "test reasoning"}]}
        
        embeddings = encode_semantics(test_trinity, test_iel)
        assert hasattr(embeddings, 'shape'), "Missing shape attribute in embeddings"
        
        drift_result = detect_concept_drift(embeddings)
        assert "drift_detected" in drift_result, "Missing drift_detected in drift result"
        
        print("✓ Semantic transformers component test passed")
        success_count += 1
        
    except Exception as e:
        print(f"✗ Semantic transformers component test failed: {e}")
    
    # Test autonomous learning
    total_tests += 1
    try:
        from intelligence.adaptive.autonomous_learning import run_rl_cycle
        
        test_posterior = {"confidence": 0.8}
        test_embeddings = type('obj', (object,), {'shape': (64,)})()
        test_drift = {"drift_detected": False}
        
        rl_result = run_rl_cycle(test_posterior, test_embeddings, test_drift)
        assert "avg_regret" in rl_result, "Missing avg_regret in RL result"
        assert "updates" in rl_result, "Missing updates in RL result"
        
        print("✓ Autonomous learning component test passed")
        success_count += 1
        
    except Exception as e:
        print(f"✗ Autonomous learning component test failed: {e}")
    
    # Test self improvement
    total_tests += 1
    try:
        from intelligence.adaptive.self_improvement import optimize_models
        
        test_posterior = {"confidence": 0.7, "variance": 0.2}
        test_embeddings = type('obj', (object,), {'shape': (128,)})()
        test_rl = {"avg_regret": 0.1, "updates": 2}
        test_drift = {"drift_detected": False, "delta": 0.0}
        
        improvement_result = optimize_models(test_posterior, test_embeddings, test_rl, test_drift)
        assert "updated" in improvement_result, "Missing updated in improvement result"
        assert "state_hash" in improvement_result, "Missing state_hash in improvement result"
        
        print("✓ Self improvement component test passed")
        success_count += 1
        
    except Exception as e:
        print(f"✗ Self improvement component test failed: {e}")
    
    # Test persistence
    total_tests += 1
    try:
        from protocols.system_operations.persistence_manager import persist_adaptive_state
        
        test_profile = {"test": "data", "meta": {"step": 5}}
        result = persist_adaptive_state(test_profile)
        assert isinstance(result, bool), "Persistence should return boolean"
        
        print("✓ Persistence component test passed")
        success_count += 1
        
    except Exception as e:
        print(f"✗ Persistence component test failed: {e}")
    
    # Test audit logging
    total_tests += 1
    try:
        from audit.audit_logger import log_event
        
        result = log_event("TEST_EVENT", {"test": "data"})
        assert isinstance(result, bool), "Audit logging should return boolean"
        
        print("✓ Audit logging component test passed")
        success_count += 1
        
    except Exception as e:
        print(f"✗ Audit logging component test failed: {e}")
    
    print(f"\nComponent Tests Summary: {success_count}/{total_tests} passed ({success_count/total_tests*100:.1f}%)")
    return success_count == total_tests

def run_all_tests():
    """Run all validation tests."""
    print("UIP Step 5 Adaptive Inference Layer - Validation Tests")
    print("=" * 60)
    
    tests = [
        ("Component Availability", test_component_availability),
        ("Individual Components", test_individual_components), 
        ("Minimal Pipeline Execution", test_step5_executes_minimal_pipeline)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 40)
        try:
            if test_func():
                passed_tests += 1
                print(f"✓ {test_name}: PASSED")
            else:
                print(f"✗ {test_name}: FAILED")
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 All tests PASSED - UIP Step 5 is operational!")
        return 0
    else:
        print("❌ Some tests FAILED - Check logs for details")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)