"""
Test Pipeline - UIP Step 2 IEL Ontological Synthesis Gateway

Comprehensive validation for recombine → λ → translate → OBDC workflow integration.
Tests end-to-end pipeline functionality and quality assurance.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime
import json

# Import IEL_ONTO_KIT modules  
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recombine import recombine_core, entropy_metrics
from lambda_processors import lambda_core
from translators import translation_engine
from obdc import obdc_kernel
from registry import iel_registry
from math_cores import vector_norms

logger = logging.getLogger("IEL_ONTO_KIT")


def run_validation() -> Dict[str, Any]:
    """
    Run comprehensive validation of the complete IEL synthesis pipeline
    
    Returns:
        Dict containing validation results and quality metrics
    """
    try:
        logger.info("Starting IEL_ONTO_KIT pipeline validation")
        
        # Initialize validation context
        validation_context = {
            "start_time": datetime.utcnow().isoformat(),
            "test_scenarios": [],
            "stage_results": {},
            "performance_metrics": {},
            "quality_assessment": {},
            "issues_found": []
        }
        
        # Define test scenarios
        test_scenarios = _create_test_scenarios()
        validation_context["test_scenarios"] = [scenario["name"] for scenario in test_scenarios]
        
        # Run validation for each scenario
        all_results = []
        for scenario in test_scenarios:
            scenario_result = _run_scenario_validation(scenario, validation_context)
            all_results.append(scenario_result)
        
        # Aggregate results
        aggregated_results = _aggregate_validation_results(all_results)
        validation_context["stage_results"] = aggregated_results["stage_results"]
        validation_context["performance_metrics"] = aggregated_results["performance_metrics"]
        validation_context["quality_assessment"] = aggregated_results["quality_assessment"]
        
        # Overall validation assessment
        overall_success = _assess_overall_validation_success(validation_context)
        
        validation_context["end_time"] = datetime.utcnow().isoformat()
        validation_context["overall_success"] = overall_success
        validation_context["validation_summary"] = _generate_validation_summary(validation_context)
        
        logger.info(f"Pipeline validation completed: success={overall_success}")
        
        return {
            "status": "ok",
            "payload": validation_context,
            "metadata": {
                "stage": "pipeline_validation",
                "scenarios_tested": len(test_scenarios),
                "overall_success": overall_success
            }
        }
        
    except Exception as e:
        logger.error(f"Pipeline validation failed: {e}")
        return {
            "status": "error",
            "payload": {"error": str(e)},
            "metadata": {"stage": "pipeline_validation"}
        }


def validate_workflow_integration() -> Dict[str, Any]:
    """
    Validate specific workflow integration points
    
    Returns:
        Dict containing integration validation results
    """
    try:
        logger.info("Starting workflow integration validation")
        
        integration_tests = {
            "registry_to_recombine": _test_registry_recombine_integration(),
            "recombine_to_lambda": _test_recombine_lambda_integration(),
            "lambda_to_translate": _test_lambda_translate_integration(),
            "translate_to_obdc": _test_translate_obdc_integration(),
            "data_flow_integrity": _test_data_flow_integrity(),
            "error_handling": _test_error_handling_integration()
        }
        
        # Assess integration quality
        successful_integrations = sum(1 for result in integration_tests.values() if result.get("success", False))
        integration_success_rate = successful_integrations / len(integration_tests)
        
        return {
            "status": "ok",
            "payload": {
                "integration_tests": integration_tests,
                "integration_success_rate": integration_success_rate,
                "successful_integrations": successful_integrations,
                "total_integrations": len(integration_tests),
                "overall_integration_success": integration_success_rate >= 0.8
            },
            "metadata": {
                "stage": "integration_validation",
                "success_rate": integration_success_rate
            }
        }
        
    except Exception as e:
        logger.error(f"Workflow integration validation failed: {e}")
        return {
            "status": "error",
            "payload": {"error": str(e)},
            "metadata": {"stage": "integration_validation"}
        }


def _create_test_scenarios() -> List[Dict[str, Any]]:
    """Create test scenarios for pipeline validation"""
    scenarios = [
        {
            "name": "basic_synthesis",
            "description": "Basic IEL synthesis with minimal input",
            "input_data": {
                "trinity_vectors": {
                    "ethical": [0.5, 0.3, 0.2],
                    "gnoseological": [0.4, 0.4, 0.2], 
                    "teleological": [0.3, 0.3, 0.4]
                },
                "entities": [{"name": "test_entity", "type": "concept"}],
                "intent_classification": {"analysis": 0.7, "synthesis": 0.3},
                "processing_depth": 1
            }
        },
        {
            "name": "complex_synthesis", 
            "description": "Complex IEL synthesis with rich input",
            "input_data": {
                "trinity_vectors": {
                    "ethical": [0.8, 0.1, 0.1, 0.6, 0.4],
                    "gnoseological": [0.2, 0.7, 0.1, 0.5, 0.3],
                    "teleological": [0.3, 0.2, 0.5, 0.4, 0.6]
                },
                "entities": [
                    {"name": "complex_entity_1", "type": "abstract_concept"},
                    {"name": "complex_entity_2", "type": "concrete_object"},
                    {"name": "relation_1", "type": "relationship"}
                ],
                "intent_classification": {
                    "analysis": 0.4, "synthesis": 0.3, "evaluation": 0.2, "creation": 0.1
                },
                "semantic_relations": [
                    {"source": "complex_entity_1", "target": "complex_entity_2", "relation": "contains"}
                ],
                "processing_depth": 3
            }
        },
        {
            "name": "minimal_synthesis",
            "description": "Minimal IEL synthesis with sparse input",
            "input_data": {
                "trinity_vectors": {
                    "ethical": [0.1],
                    "gnoseological": [0.1],
                    "teleological": [0.1]
                },
                "entities": [],
                "intent_classification": {"unknown": 1.0},
                "processing_depth": 1
            }
        },
        {
            "name": "multi_domain_synthesis",
            "description": "Multi-domain IEL synthesis with diverse elements",
            "input_data": {
                "trinity_vectors": {
                    "ethical": [0.6, 0.8, 0.4, 0.7, 0.5, 0.3, 0.9, 0.2],
                    "gnoseological": [0.7, 0.3, 0.6, 0.4, 0.8, 0.5, 0.2, 0.7],
                    "teleological": [0.5, 0.6, 0.7, 0.8, 0.3, 0.9, 0.4, 0.6]
                },
                "entities": [
                    {"name": f"entity_{i}", "type": "domain_concept"} for i in range(5)
                ],
                "intent_classification": {
                    "modal_analysis": 0.3, "epistemic_processing": 0.25,
                    "normative_reasoning": 0.2, "action_theory": 0.15,
                    "linguistic_synthesis": 0.1
                },
                "processing_depth": 2
            }
        }
    ]
    
    return scenarios


def _run_scenario_validation(scenario: Dict[str, Any], 
                           validation_context: Dict[str, Any]) -> Dict[str, Any]:
    """Run validation for a specific test scenario"""
    try:
        scenario_name = scenario["name"]
        input_data = scenario["input_data"]
        
        logger.info(f"Running validation scenario: {scenario_name}")
        
        scenario_result = {
            "scenario_name": scenario_name,
            "start_time": time.time(),
            "stage_results": {},
            "performance_metrics": {},
            "success": False,
            "issues": []
        }
        
        # Stage 1: Registry → Load active domains
        try:
            stage1_start = time.time()
            active_domains = iel_registry.load_active_domains()
            stage1_time = time.time() - stage1_start
            
            scenario_result["stage_results"]["registry"] = {
                "success": len(active_domains) > 0,
                "domains_loaded": len(active_domains),
                "execution_time": stage1_time
            }
        except Exception as e:
            scenario_result["stage_results"]["registry"] = {
                "success": False, "error": str(e), "execution_time": 0
            }
            scenario_result["issues"].append(f"Registry stage failed: {e}")
        
        # Stage 2: Recombine → Merge IEL outputs
        try:
            stage2_start = time.time()
            recombined_result = recombine_core.merge_outputs(input_data, active_domains)
            stage2_time = time.time() - stage2_start
            
            scenario_result["stage_results"]["recombine"] = {
                "success": recombined_result.get("status") == "ok",
                "quality_score": recombined_result.get("payload", {}).get("synthesis_metadata", {}).get("quality_score", 0.0),
                "execution_time": stage2_time
            }
        except Exception as e:
            scenario_result["stage_results"]["recombine"] = {
                "success": False, "error": str(e), "execution_time": 0
            }
            scenario_result["issues"].append(f"Recombine stage failed: {e}")
            recombined_result = {"status": "error", "payload": {}}
        
        # Stage 3: Lambda → Normalize structures
        try:
            stage3_start = time.time()
            lambda_result = lambda_core.normalize_structure(recombined_result)
            stage3_time = time.time() - stage3_start
            
            scenario_result["stage_results"]["lambda"] = {
                "success": lambda_result.get("status") == "ok",
                "complexity_score": lambda_result.get("payload", {}).get("normalization_metadata", {}).get("complexity_score", 0.0),
                "execution_time": stage3_time
            }
        except Exception as e:
            scenario_result["stage_results"]["lambda"] = {
                "success": False, "error": str(e), "execution_time": 0
            }
            scenario_result["issues"].append(f"Lambda stage failed: {e}")
            lambda_result = {"status": "error", "payload": {}}
        
        # Stage 4: Translate → Convert to natural language
        try:
            stage4_start = time.time()
            translate_result = translation_engine.convert_to_nl(lambda_result)
            stage4_time = time.time() - stage4_start
            
            scenario_result["stage_results"]["translate"] = {
                "success": translate_result.get("status") == "ok",
                "quality_score": translate_result.get("payload", {}).get("translation_metadata", {}).get("quality_metrics", {}).get("overall_quality", 0.0),
                "execution_time": stage4_time
            }
        except Exception as e:
            scenario_result["stage_results"]["translate"] = {
                "success": False, "error": str(e), "execution_time": 0
            }
            scenario_result["issues"].append(f"Translate stage failed: {e}")
            translate_result = {"status": "error", "payload": {}}
        
        # Stage 5: OBDC → Emit tokens
        try:
            stage5_start = time.time()
            obdc_result = obdc_kernel.emit_tokens(translate_result)
            stage5_time = time.time() - stage5_start
            
            scenario_result["stage_results"]["obdc"] = {
                "success": obdc_result.get("status") == "ok",
                "tokens_emitted": obdc_result.get("payload", {}).get("emission_metadata", {}).get("total_tokens", 0),
                "quality_score": obdc_result.get("payload", {}).get("validation_results", {}).get("quality_score", 0.0),
                "execution_time": stage5_time
            }
        except Exception as e:
            scenario_result["stage_results"]["obdc"] = {
                "success": False, "error": str(e), "execution_time": 0
            }
            scenario_result["issues"].append(f"OBDC stage failed: {e}")
        
        # Calculate overall scenario success
        successful_stages = sum(1 for stage in scenario_result["stage_results"].values() if stage.get("success", False))
        total_stages = len(scenario_result["stage_results"])
        scenario_result["success"] = successful_stages == total_stages and len(scenario_result["issues"]) == 0
        
        scenario_result["end_time"] = time.time()
        scenario_result["total_execution_time"] = scenario_result["end_time"] - scenario_result["start_time"]
        
        return scenario_result
        
    except Exception as e:
        logger.error(f"Scenario validation failed for {scenario_name}: {e}")
        return {
            "scenario_name": scenario_name,
            "success": False,
            "error": str(e),
            "stage_results": {},
            "issues": [f"Scenario execution failed: {e}"]
        }


def _aggregate_validation_results(scenario_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results from all validation scenarios"""
    try:
        # Initialize aggregated results
        aggregated = {
            "stage_results": {
                "registry": {"success_count": 0, "total_count": 0, "avg_execution_time": 0.0},
                "recombine": {"success_count": 0, "total_count": 0, "avg_execution_time": 0.0, "avg_quality": 0.0},
                "lambda": {"success_count": 0, "total_count": 0, "avg_execution_time": 0.0, "avg_complexity": 0.0},
                "translate": {"success_count": 0, "total_count": 0, "avg_execution_time": 0.0, "avg_quality": 0.0},
                "obdc": {"success_count": 0, "total_count": 0, "avg_execution_time": 0.0, "avg_tokens": 0.0}
            },
            "performance_metrics": {
                "total_scenarios": len(scenario_results),
                "successful_scenarios": 0,
                "avg_total_execution_time": 0.0,
                "stage_success_rates": {}
            },
            "quality_assessment": {
                "overall_quality_score": 0.0,
                "stage_quality_scores": {},
                "consistency_score": 0.0
            }
        }
        
        # Aggregate stage results
        execution_times = []
        
        for result in scenario_results:
            if result.get("success", False):
                aggregated["performance_metrics"]["successful_scenarios"] += 1
            
            execution_times.append(result.get("total_execution_time", 0.0))
            
            # Process each stage
            for stage_name, stage_data in result.get("stage_results", {}).items():
                if stage_name in aggregated["stage_results"]:
                    agg_stage = aggregated["stage_results"][stage_name]
                    
                    agg_stage["total_count"] += 1
                    if stage_data.get("success", False):
                        agg_stage["success_count"] += 1
                    
                    # Accumulate execution times
                    exec_time = stage_data.get("execution_time", 0.0)
                    agg_stage["avg_execution_time"] += exec_time
                    
                    # Stage-specific metrics
                    if stage_name == "recombine":
                        quality = stage_data.get("quality_score", 0.0)
                        if "quality_sum" not in agg_stage:
                            agg_stage["quality_sum"] = 0.0
                        agg_stage["quality_sum"] += quality
                    elif stage_name == "lambda":
                        complexity = stage_data.get("complexity_score", 0.0)
                        if "complexity_sum" not in agg_stage:
                            agg_stage["complexity_sum"] = 0.0
                        agg_stage["complexity_sum"] += complexity
                    elif stage_name == "translate":
                        quality = stage_data.get("quality_score", 0.0)
                        if "quality_sum" not in agg_stage:
                            agg_stage["quality_sum"] = 0.0
                        agg_stage["quality_sum"] += quality
                    elif stage_name == "obdc":
                        tokens = stage_data.get("tokens_emitted", 0)
                        if "tokens_sum" not in agg_stage:
                            agg_stage["tokens_sum"] = 0.0
                        agg_stage["tokens_sum"] += tokens
        
        # Calculate averages
        for stage_name, stage_data in aggregated["stage_results"].items():
            if stage_data["total_count"] > 0:
                stage_data["avg_execution_time"] /= stage_data["total_count"]
                
                # Stage-specific averages
                if stage_name in ["recombine", "translate"] and "quality_sum" in stage_data:
                    stage_data["avg_quality"] = stage_data["quality_sum"] / stage_data["total_count"]
                elif stage_name == "lambda" and "complexity_sum" in stage_data:
                    stage_data["avg_complexity"] = stage_data["complexity_sum"] / stage_data["total_count"]
                elif stage_name == "obdc" and "tokens_sum" in stage_data:
                    stage_data["avg_tokens"] = stage_data["tokens_sum"] / stage_data["total_count"]
        
        # Performance metrics
        if execution_times:
            aggregated["performance_metrics"]["avg_total_execution_time"] = sum(execution_times) / len(execution_times)
        
        # Stage success rates
        for stage_name, stage_data in aggregated["stage_results"].items():
            if stage_data["total_count"] > 0:
                success_rate = stage_data["success_count"] / stage_data["total_count"]
                aggregated["performance_metrics"]["stage_success_rates"][stage_name] = success_rate
        
        # Quality assessment
        quality_scores = []
        for stage_name, stage_data in aggregated["stage_results"].items():
            if "avg_quality" in stage_data:
                quality_scores.append(stage_data["avg_quality"])
        
        if quality_scores:
            aggregated["quality_assessment"]["overall_quality_score"] = sum(quality_scores) / len(quality_scores)
        
        return aggregated
        
    except Exception as e:
        logger.error(f"Result aggregation failed: {e}")
        return {"stage_results": {}, "performance_metrics": {}, "quality_assessment": {}}


def _test_registry_recombine_integration() -> Dict[str, Any]:
    """Test integration between registry and recombine modules"""
    try:
        # Load domains from registry
        domains = iel_registry.load_active_domains()
        
        # Create test input for recombine
        test_input = {
            "trinity_vectors": {"ethical": [0.5], "gnoseological": [0.5], "teleological": [0.5]},
            "entities": [],
            "intent_classification": {"test": 1.0}
        }
        
        # Test recombine with registry domains
        result = recombine_core.merge_outputs(test_input, domains)
        
        return {
            "success": result.get("status") == "ok",
            "domains_loaded": len(domains),
            "recombine_success": result.get("status") == "ok",
            "integration_quality": "good" if len(domains) > 0 and result.get("status") == "ok" else "poor"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def _test_recombine_lambda_integration() -> Dict[str, Any]:
    """Test integration between recombine and lambda modules"""
    try:
        # Create mock recombined data
        mock_recombined = {
            "status": "ok",
            "payload": {
                "merged_vectors": {"weighted_average": [0.5, 0.5, 0.5], "coherence_score": 0.7},
                "ontological_alignments": {"TestDomain": {"activation_strength": 0.8}},
                "synthesis_metadata": {"quality_score": 0.75}
            }
        }
        
        # Test lambda processing
        lambda_result = lambda_core.normalize_structure(mock_recombined)
        
        return {
            "success": lambda_result.get("status") == "ok",
            "input_format_compatible": True,
            "output_format_valid": "normalized_structures" in lambda_result.get("payload", {}),
            "integration_quality": "good" if lambda_result.get("status") == "ok" else "poor"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def _test_lambda_translate_integration() -> Dict[str, Any]:
    """Test integration between lambda and translate modules"""
    try:
        # Create mock lambda data
        mock_lambda = {
            "status": "ok",
            "payload": {
                "normalized_structures": {
                    "test_structure": {
                        "lambda_expressions": [{"type": "test", "lambda_expression": "λx.x"}],
                        "reduction_steps": []
                    }
                },
                "normalization_metadata": {"complexity_score": 0.5}
            }
        }
        
        # Test translation
        translate_result = translation_engine.convert_to_nl(mock_lambda)
        
        return {
            "success": translate_result.get("status") == "ok",
            "input_format_compatible": True,
            "output_format_valid": "unified_natural_language" in translate_result.get("payload", {}),
            "integration_quality": "good" if translate_result.get("status") == "ok" else "poor"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def _test_translate_obdc_integration() -> Dict[str, Any]:
    """Test integration between translate and OBDC modules"""
    try:
        # Create mock translate data
        mock_translate = {
            "status": "ok",
            "payload": {
                "unified_natural_language": "Test natural language output.",
                "structured_translations": {
                    "test_structure": {"natural_language_text": "Test text"}
                },
                "translation_metadata": {"quality_metrics": {"overall_quality": 0.8}}
            }
        }
        
        # Test OBDC emission
        obdc_result = obdc_kernel.emit_tokens(mock_translate)
        
        return {
            "success": obdc_result.get("status") == "ok",
            "input_format_compatible": True,
            "output_format_valid": "token_stream" in obdc_result.get("payload", {}),
            "integration_quality": "good" if obdc_result.get("status") == "ok" else "poor"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def _test_data_flow_integrity() -> Dict[str, Any]:
    """Test data flow integrity across the entire pipeline"""
    try:
        # Test with minimal but complete data flow
        test_input = {
            "trinity_vectors": {"ethical": [0.3], "gnoseological": [0.3], "teleological": [0.4]},
            "entities": [{"name": "test", "type": "concept"}],
            "intent_classification": {"test": 1.0}
        }
        
        # Step 1: Registry
        domains = iel_registry.load_active_domains()
        
        # Step 2: Recombine 
        recombined = recombine_core.merge_outputs(test_input, domains[:3])  # Limit domains for test
        
        # Step 3: Lambda
        normalized = lambda_core.normalize_structure(recombined)
        
        # Step 4: Translate
        translated = translation_engine.convert_to_nl(normalized)
        
        # Step 5: OBDC
        tokens = obdc_kernel.emit_tokens(translated)
        
        # Check data integrity
        integrity_score = 0.0
        if recombined.get("status") == "ok":
            integrity_score += 0.2
        if normalized.get("status") == "ok":
            integrity_score += 0.2
        if translated.get("status") == "ok":
            integrity_score += 0.3
        if tokens.get("status") == "ok":
            integrity_score += 0.3
        
        return {
            "success": integrity_score >= 0.8,
            "integrity_score": integrity_score,
            "data_flow_complete": tokens.get("status") == "ok",
            "integration_quality": "excellent" if integrity_score >= 0.9 else "good" if integrity_score >= 0.7 else "poor"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def _test_error_handling_integration() -> Dict[str, Any]:
    """Test error handling across pipeline integration"""
    try:
        error_scenarios = []
        
        # Test 1: Invalid input to recombine
        try:
            recombine_core.merge_outputs({}, [])
            error_scenarios.append({"scenario": "empty_input", "handled": True})
        except Exception:
            error_scenarios.append({"scenario": "empty_input", "handled": False})
        
        # Test 2: Invalid input to lambda
        try:
            lambda_core.normalize_structure({"status": "error"})
            error_scenarios.append({"scenario": "error_input_lambda", "handled": True})
        except Exception:
            error_scenarios.append({"scenario": "error_input_lambda", "handled": False})
        
        # Test 3: Invalid input to translation
        try:
            translation_engine.convert_to_nl({"status": "error"})
            error_scenarios.append({"scenario": "error_input_translate", "handled": True})
        except Exception:
            error_scenarios.append({"scenario": "error_input_translate", "handled": False})
        
        # Calculate error handling success rate
        handled_count = sum(1 for scenario in error_scenarios if scenario["handled"])
        success_rate = handled_count / len(error_scenarios) if error_scenarios else 0.0
        
        return {
            "success": success_rate >= 0.8,
            "error_handling_rate": success_rate,
            "scenarios_tested": len(error_scenarios),
            "scenarios_handled": handled_count,
            "integration_quality": "good" if success_rate >= 0.8 else "poor"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def _assess_overall_validation_success(validation_context: Dict[str, Any]) -> bool:
    """Assess overall validation success based on all metrics"""
    try:
        stage_results = validation_context.get("stage_results", {})
        performance_metrics = validation_context.get("performance_metrics", {})
        
        # Success criteria
        criteria = {
            "scenario_success_rate": performance_metrics.get("successful_scenarios", 0) / performance_metrics.get("total_scenarios", 1),
            "stage_success_rates": performance_metrics.get("stage_success_rates", {}),
            "performance_acceptable": performance_metrics.get("avg_total_execution_time", float('inf')) < 10.0,  # Under 10 seconds
            "quality_acceptable": validation_context.get("quality_assessment", {}).get("overall_quality_score", 0.0) > 0.5
        }
        
        # Overall success evaluation
        scenario_success = criteria["scenario_success_rate"] >= 0.75
        stage_success = all(rate >= 0.8 for rate in criteria["stage_success_rates"].values())
        
        overall_success = (
            scenario_success and 
            stage_success and
            criteria["performance_acceptable"] and
            criteria["quality_acceptable"]
        )
        
        return overall_success
        
    except Exception:
        return False


def _generate_validation_summary(validation_context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate validation summary report"""
    try:
        performance_metrics = validation_context.get("performance_metrics", {})
        quality_assessment = validation_context.get("quality_assessment", {})
        
        summary = {
            "validation_status": "PASS" if validation_context.get("overall_success", False) else "FAIL",
            "scenarios_tested": performance_metrics.get("total_scenarios", 0),
            "scenarios_passed": performance_metrics.get("successful_scenarios", 0),
            "overall_success_rate": performance_metrics.get("successful_scenarios", 0) / max(performance_metrics.get("total_scenarios", 1), 1),
            "avg_execution_time": performance_metrics.get("avg_total_execution_time", 0.0),
            "quality_score": quality_assessment.get("overall_quality_score", 0.0),
            "stage_performance": performance_metrics.get("stage_success_rates", {}),
            "issues_count": len(validation_context.get("issues_found", [])),
            "recommendations": _generate_recommendations(validation_context)
        }
        
        return summary
        
    except Exception:
        return {"validation_status": "ERROR", "error": "Failed to generate summary"}


def _generate_recommendations(validation_context: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on validation results"""
    recommendations = []
    
    try:
        performance_metrics = validation_context.get("performance_metrics", {})
        stage_success_rates = performance_metrics.get("stage_success_rates", {})
        
        # Performance recommendations
        avg_time = performance_metrics.get("avg_total_execution_time", 0.0)
        if avg_time > 5.0:
            recommendations.append("Consider optimizing pipeline performance - average execution time exceeds 5 seconds")
        
        # Stage-specific recommendations
        for stage, rate in stage_success_rates.items():
            if rate < 0.9:
                recommendations.append(f"Improve {stage} stage reliability - success rate is {rate:.1%}")
        
        # Quality recommendations
        quality_score = validation_context.get("quality_assessment", {}).get("overall_quality_score", 0.0)
        if quality_score < 0.7:
            recommendations.append("Enhance processing quality - overall quality score is below acceptable threshold")
        
        if not recommendations:
            recommendations.append("Pipeline validation successful - no critical issues identified")
        
    except Exception:
        recommendations.append("Unable to generate specific recommendations due to validation analysis errors")
    
    return recommendations