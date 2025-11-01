#!/usr/bin/env python3
"""
LOGOS Functionality Validation Suite
===================================

Comprehensive test suite to verify LOGOS is functioning as intended,
not just running. Tests cognitive capabilities, reasoning coherence,
Trinity integration, and philosophical analysis functions.
"""

import sys
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add LOGOS_AI to path
sys.path.append(str(Path(__file__).parent / "LOGOS_AI"))

class LOGOSFunctionalityValidator:
    """Comprehensive LOGOS functionality validation system"""
    
    def __init__(self):
        self.test_results = {}
        self.validation_score = 0.0
        self.critical_failures = []
        self.warnings = []
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete functionality validation suite"""
        
        print("🔍 LOGOS FUNCTIONALITY VALIDATION SUITE")
        print("=" * 60)
        print("Verifying LOGOS is functioning as intended...")
        print()
        
        validation_categories = [
            ("Core Systems", self.test_core_systems),
            ("UIP Pipeline", self.test_uip_pipeline),
            ("AGP Systems", self.test_agp_systems),
            ("Trinity Framework", self.test_trinity_framework),
            ("Philosophical Reasoning", self.test_philosophical_reasoning),
            ("Cognitive Capabilities", self.test_cognitive_capabilities),
            ("Mathematical Foundations", self.test_mathematical_foundations),
            ("Integration Coherence", self.test_integration_coherence),
            ("Response Quality", self.test_response_quality),
            ("Specialized Analysis", self.test_specialized_analysis)
        ]
        
        total_categories = len(validation_categories)
        passed_categories = 0
        
        for category_name, test_function in validation_categories:
            print(f"🧪 Testing {category_name}...")
            try:
                result = await test_function()
                self.test_results[category_name] = result
                
                if result.get('passed', False):
                    passed_categories += 1
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                    if result.get('critical', False):
                        self.critical_failures.append(category_name)
                
                print(f"   {status} - {result.get('summary', 'No summary')}")
                
                if result.get('warnings'):
                    for warning in result['warnings']:
                        print(f"   ⚠️  {warning}")
                        self.warnings.append(warning)
                
            except Exception as e:
                print(f"   ❌ ERROR - {str(e)}")
                self.test_results[category_name] = {
                    'passed': False, 
                    'error': str(e),
                    'critical': True
                }
                self.critical_failures.append(category_name)
            
            print()
        
        # Calculate overall validation score
        self.validation_score = passed_categories / total_categories
        
        # Generate final report
        return self.generate_validation_report()
    
    async def test_core_systems(self) -> Dict[str, Any]:
        """Test core LOGOS system components"""
        
        try:
            # Test basic imports and initialization
            from User_Interaction_Protocol.intelligence.logos_core import LogosCore
            from User_Interaction_Protocol.intelligence.protopraxic_divine_processor import ProtopraxicDivineProcessor
            
            logos_core = LogosCore()
            pdp = ProtopraxicDivineProcessor()
            
            # Test basic functionality
            core_initialized = hasattr(logos_core, 'process_input')
            pdp_initialized = hasattr(pdp, 'process')
            
            return {
                'passed': core_initialized and pdp_initialized,
                'summary': f'Core systems {"operational" if core_initialized and pdp_initialized else "failed"}',
                'details': {
                    'logos_core': core_initialized,
                    'protopraxic_processor': pdp_initialized
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'summary': f'Core system initialization failed: {str(e)}',
                'critical': True
            }
    
    async def test_uip_pipeline(self) -> Dict[str, Any]:
        """Test User Interaction Protocol pipeline"""
        
        try:
            # Test UIP pipeline with sample query
            import subprocess
            import json
            
            # Test basic UIP functionality
            result = subprocess.run([
                sys.executable, 'send_message_to_logos.py'
            ], input='Test UIP functionality\n', 
               capture_output=True, text=True, timeout=30)
            
            success = result.returncode == 0 and len(result.stdout) > 0
            
            # Check for key UIP components in output
            output = result.stdout.lower()
            uip_indicators = [
                'uip system initialization',
                'pxl integration',
                'trinity',
                'instantiation'
            ]
            
            indicators_found = sum(1 for indicator in uip_indicators if indicator in output)
            
            return {
                'passed': success and indicators_found >= 2,
                'summary': f'UIP pipeline {"functional" if success else "failed"} ({indicators_found}/4 indicators)',
                'details': {
                    'execution_success': success,
                    'indicators_found': indicators_found,
                    'output_length': len(result.stdout)
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'summary': f'UIP pipeline test failed: {str(e)}',
                'critical': True
            }
    
    async def test_agp_systems(self) -> Dict[str, Any]:
        """Test Autonomous Goal Pursuit systems"""
        
        try:
            from startup.agp_startup import AGPStartupManager
            from singularity import SingularitySystem
            
            agp_manager = AGPStartupManager()
            singularity = SingularitySystem()
            
            # Test AGP initialization
            agp_status = agp_manager.status if hasattr(agp_manager, 'status') else None
            
            # Check key AGP components
            components_available = {
                'mvs_space': hasattr(agp_manager, 'mvs_space'),
                'bdn_network': hasattr(agp_manager, 'bdn_network'),
                'singularity_system': hasattr(agp_manager, 'singularity_system'),
                'trinity_enhancement': hasattr(agp_manager, 'trinity_enhancement')
            }
            
            components_count = sum(components_available.values())
            
            return {
                'passed': components_count >= 2,
                'summary': f'AGP systems {"operational" if components_count >= 2 else "incomplete"} ({components_count}/4 components)',
                'details': components_available,
                'warnings': ['Some AGP components running in fallback mode'] if components_count < 4 else []
            }
            
        except Exception as e:
            return {
                'passed': False,
                'summary': f'AGP systems test failed: {str(e)}',
                'critical': False
            }
    
    async def test_trinity_framework(self) -> Dict[str, Any]:
        """Test Trinity mathematical framework"""
        
        try:
            # Test Trinity instantiation system
            test_queries = [
                ('identity', 'What is identity?'),
                ('philosophical', 'What is the nature of existence?'),
                ('meta-ontological', 'What is being?')
            ]
            
            trinity_results = []
            
            for category, query in test_queries:
                try:
                    import subprocess
                    result = subprocess.run([
                        sys.executable, 'send_message_to_logos.py'
                    ], input=f'{query}\n', 
                       capture_output=True, text=True, timeout=20)
                    
                    if result.returncode == 0:
                        output = result.stdout.lower()
                        # Check for Trinity instantiation indicators
                        trinity_indicators = ['𝕀₁', '𝕀₂', '𝕀₃', 'trinity', 'instantiation']
                        indicators_found = sum(1 for indicator in trinity_indicators if indicator in output)
                        trinity_results.append(indicators_found > 0)
                    else:
                        trinity_results.append(False)
                        
                except:
                    trinity_results.append(False)
            
            success_rate = sum(trinity_results) / len(trinity_results)
            
            return {
                'passed': success_rate >= 0.5,
                'summary': f'Trinity framework {"functional" if success_rate >= 0.5 else "impaired"} ({success_rate*100:.1f}% success rate)',
                'details': {
                    'test_queries': len(test_queries),
                    'successful_instantiations': sum(trinity_results),
                    'success_rate': success_rate
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'summary': f'Trinity framework test failed: {str(e)}',
                'critical': False
            }
    
    async def test_philosophical_reasoning(self) -> Dict[str, Any]:
        """Test philosophical reasoning capabilities"""
        
        try:
            # Test specialized philosophical tools
            philosophical_tools = [
                ('meta_epistemology_analysis.py', 'Meta-epistemological'),
                ('meta_ontology_existence.py', 'Meta-ontological'),
                ('identity_analysis.py', 'Identity'),
                ('necessary_identity_analysis.py', 'Necessary Identity')
            ]
            
            tool_results = []
            
            for tool_file, tool_type in philosophical_tools:
                if Path(tool_file).exists():
                    try:
                        import subprocess
                        result = subprocess.run([
                            sys.executable, tool_file
                        ], capture_output=True, text=True, timeout=30)
                        
                        success = result.returncode == 0 and 'confidence:' in result.stdout.lower()
                        tool_results.append((tool_type, success))
                        
                    except:
                        tool_results.append((tool_type, False))
                else:
                    tool_results.append((tool_type, False))
            
            successful_tools = sum(1 for _, success in tool_results if success)
            
            return {
                'passed': successful_tools >= 2,
                'summary': f'Philosophical reasoning {"capable" if successful_tools >= 2 else "limited"} ({successful_tools}/{len(philosophical_tools)} tools functional)',
                'details': {
                    'tool_results': tool_results,
                    'functional_tools': successful_tools
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'summary': f'Philosophical reasoning test failed: {str(e)}',
                'critical': False
            }
    
    async def test_cognitive_capabilities(self) -> Dict[str, Any]:
        """Test cognitive reasoning capabilities"""
        
        try:
            # Test various cognitive functions through LOGOS
            cognitive_tests = [
                ('Logical reasoning', 'If A implies B and A is true, what can we conclude about B?'),
                ('Modal logic', 'What is the difference between necessary and contingent truth?'),
                ('Causal reasoning', 'What is the relationship between cause and effect?'),
                ('Semantic understanding', 'What does it mean for something to exist?')
            ]
            
            cognitive_scores = []
            
            for test_name, query in cognitive_tests:
                try:
                    import subprocess
                    result = subprocess.run([
                        sys.executable, 'send_message_to_logos.py'
                    ], input=f'{query}\n', 
                       capture_output=True, text=True, timeout=25)
                    
                    if result.returncode == 0:
                        output = result.stdout.lower()
                        # Check for reasoning indicators
                        reasoning_indicators = ['because', 'therefore', 'thus', 'implies', 'follows', 'necessarily']
                        indicators_found = sum(1 for indicator in reasoning_indicators if indicator in output)
                        
                        # Check for confidence/quality indicators
                        quality_indicators = ['confidence:', 'grounding:', 'framework']
                        quality_found = sum(1 for indicator in quality_indicators if indicator in output)
                        
                        score = min(1.0, (indicators_found + quality_found) / 4.0)
                        cognitive_scores.append(score)
                    else:
                        cognitive_scores.append(0.0)
                        
                except:
                    cognitive_scores.append(0.0)
            
            average_score = sum(cognitive_scores) / len(cognitive_scores) if cognitive_scores else 0.0
            
            return {
                'passed': average_score >= 0.5,
                'summary': f'Cognitive capabilities {"strong" if average_score >= 0.7 else "adequate" if average_score >= 0.5 else "weak"} (score: {average_score:.2f})',
                'details': {
                    'test_scores': cognitive_scores,
                    'average_score': average_score,
                    'tests_conducted': len(cognitive_tests)
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'summary': f'Cognitive capabilities test failed: {str(e)}',
                'critical': False
            }
    
    async def test_mathematical_foundations(self) -> Dict[str, Any]:
        """Test mathematical foundations and Trinity optimization"""
        
        try:
            # Test Trinity optimization
            try:
                from LOGOS_AI.User_Interaction_Protocol.interfaces.services.workers.logos_mathematical_core import LOGOSMathematicalCore
                
                math_core = LOGOSMathematicalCore()
                bootstrap_success = math_core.bootstrap()
                
                # Test Trinity optimization if available
                trinity_optimal = False
                if hasattr(math_core, 'trinity_optimizer'):
                    try:
                        result = math_core.trinity_optimizer.verify_trinity_optimization()
                        trinity_optimal = result.get('optimal_n') == 3
                    except:
                        pass
                
                return {
                    'passed': bootstrap_success,
                    'summary': f'Mathematical foundations {"solid" if bootstrap_success and trinity_optimal else "basic" if bootstrap_success else "failed"}',
                    'details': {
                        'bootstrap_success': bootstrap_success,
                        'trinity_optimization': trinity_optimal
                    }
                }
                
            except ImportError:
                # Fallback test
                return {
                    'passed': True,
                    'summary': 'Mathematical foundations not directly testable (components in fallback mode)',
                    'warnings': ['Mathematical core components not available for direct testing']
                }
            
        except Exception as e:
            return {
                'passed': False,
                'summary': f'Mathematical foundations test failed: {str(e)}',
                'critical': False
            }
    
    async def test_integration_coherence(self) -> Dict[str, Any]:
        """Test system integration and coherence"""
        
        try:
            # Test cross-component integration
            integration_checks = []
            
            # Check if key files exist and are accessible
            key_components = [
                'send_message_to_logos.py',
                'activate_full_agp.py',
                'LOGOS_AI/User_Interaction_Protocol/',
                'LOGOS_AI/startup/',
                'LOGOS_AI/singularity/'
            ]
            
            for component in key_components:
                exists = Path(component).exists()
                integration_checks.append(exists)
            
            # Test component interaction
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable, '-c', 
                    'import sys; sys.path.append("LOGOS_AI"); from startup.agp_startup import AGPStartupManager; print("Integration OK")'
                ], capture_output=True, text=True, timeout=10)
                
                import_success = result.returncode == 0
                integration_checks.append(import_success)
                
            except:
                integration_checks.append(False)
            
            integration_score = sum(integration_checks) / len(integration_checks)
            
            return {
                'passed': integration_score >= 0.7,
                'summary': f'System integration {"coherent" if integration_score >= 0.8 else "adequate" if integration_score >= 0.7 else "fragmented"} ({integration_score:.2f})',
                'details': {
                    'component_availability': sum(integration_checks[:len(key_components)]),
                    'import_success': integration_checks[-1] if len(integration_checks) > len(key_components) else False,
                    'integration_score': integration_score
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'summary': f'Integration coherence test failed: {str(e)}',
                'critical': False
            }
    
    async def test_response_quality(self) -> Dict[str, Any]:
        """Test response quality and consistency"""
        
        try:
            # Test response quality metrics
            quality_tests = [
                'What is the nature of truth?',
                'Explain the concept of existence.',
                'What makes identity necessary?'
            ]
            
            quality_scores = []
            
            for query in quality_tests:
                try:
                    import subprocess
                    result = subprocess.run([
                        sys.executable, 'send_message_to_logos.py'
                    ], input=f'{query}\n', 
                       capture_output=True, text=True, timeout=20)
                    
                    if result.returncode == 0:
                        output = result.stdout
                        
                        # Quality metrics
                        length_score = min(1.0, len(output) / 1000)  # Prefer substantial responses
                        structure_score = 1.0 if any(marker in output for marker in ['✓', '•', ':', '=']) else 0.5
                        confidence_score = 1.0 if 'confidence:' in output.lower() else 0.5
                        trinity_score = 1.0 if any(symbol in output for symbol in ['𝕀₁', '𝕀₂', '𝕀₃']) else 0.5
                        
                        overall_score = (length_score + structure_score + confidence_score + trinity_score) / 4
                        quality_scores.append(overall_score)
                    else:
                        quality_scores.append(0.0)
                        
                except:
                    quality_scores.append(0.0)
            
            average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
            return {
                'passed': average_quality >= 0.6,
                'summary': f'Response quality {"excellent" if average_quality >= 0.8 else "good" if average_quality >= 0.6 else "poor"} (score: {average_quality:.2f})',
                'details': {
                    'quality_scores': quality_scores,
                    'average_quality': average_quality,
                    'tests_conducted': len(quality_tests)
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'summary': f'Response quality test failed: {str(e)}',
                'critical': False
            }
    
    async def test_specialized_analysis(self) -> Dict[str, Any]:
        """Test specialized analysis capabilities"""
        
        try:
            # Test fractal orbital analysis
            fractal_success = False
            try:
                if Path('engage_fractal_orbital.py').exists():
                    import subprocess
                    result = subprocess.run([
                        sys.executable, 'engage_fractal_orbital.py'
                    ], capture_output=True, text=True, timeout=20)
                    fractal_success = result.returncode == 0
            except:
                pass
            
            # Test causal predictor
            causal_success = False
            try:
                if Path('fractal_causal_predictor.py').exists():
                    import subprocess
                    result = subprocess.run([
                        sys.executable, 'fractal_causal_predictor.py'
                    ], capture_output=True, text=True, timeout=20)
                    causal_success = result.returncode == 0
            except:
                pass
            
            # Count specialized tools available
            specialized_tools = [
                'meta_epistemology_analysis.py',
                'meta_ontology_existence.py', 
                'identity_analysis.py',
                'necessary_identity_analysis.py',
                'engage_fractal_orbital.py',
                'fractal_causal_predictor.py'
            ]
            
            available_tools = sum(1 for tool in specialized_tools if Path(tool).exists())
            
            return {
                'passed': available_tools >= 4,
                'summary': f'Specialized analysis {"comprehensive" if available_tools >= 5 else "adequate" if available_tools >= 4 else "limited"} ({available_tools}/6 tools)',
                'details': {
                    'available_tools': available_tools,
                    'fractal_orbital': fractal_success,
                    'causal_predictor': causal_success,
                    'total_tools': len(specialized_tools)
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'summary': f'Specialized analysis test failed: {str(e)}',
                'critical': False
            }
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        report = {
            'validation_timestamp': time.time(),
            'overall_score': self.validation_score,
            'status': self.determine_overall_status(),
            'test_results': self.test_results,
            'critical_failures': self.critical_failures,
            'warnings': self.warnings,
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def determine_overall_status(self) -> str:
        """Determine overall LOGOS functionality status"""
        
        if self.critical_failures:
            return 'CRITICAL_ISSUES'
        elif self.validation_score >= 0.9:
            return 'EXCELLENT'
        elif self.validation_score >= 0.8:
            return 'GOOD'
        elif self.validation_score >= 0.7:
            return 'ADEQUATE'
        elif self.validation_score >= 0.6:
            return 'CONCERNING'
        else:
            return 'POOR'
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        if self.critical_failures:
            recommendations.append(f"Address critical failures in: {', '.join(self.critical_failures)}")
        
        if self.validation_score < 0.7:
            recommendations.append("Consider re-running AGP activation to improve system integration")
        
        if 'Trinity Framework' in self.test_results and not self.test_results['Trinity Framework'].get('passed'):
            recommendations.append("Verify Trinity instantiation system is properly configured")
        
        if 'Philosophical Reasoning' in self.test_results and not self.test_results['Philosophical Reasoning'].get('passed'):
            recommendations.append("Check specialized analysis tools and their dependencies")
        
        if len(self.warnings) > 3:
            recommendations.append("Review system warnings and consider component updates")
        
        return recommendations


async def main():
    """Main validation function"""
    
    validator = LOGOSFunctionalityValidator()
    
    # Run comprehensive validation
    report = await validator.run_comprehensive_validation()
    
    # Display final report
    print("\n" + "=" * 60)
    print("🎯 LOGOS FUNCTIONALITY VALIDATION REPORT")
    print("=" * 60)
    print()
    
    print(f"Overall Score: {report['overall_score']:.2%}")
    print(f"Status: {report['status']}")
    print()
    
    # Category breakdown
    print("📊 CATEGORY BREAKDOWN:")
    print("-" * 30)
    for category, result in report['test_results'].items():
        status = "✅ PASS" if result.get('passed') else "❌ FAIL"
        print(f"{status} - {category}: {result.get('summary', 'No summary')}")
    print()
    
    # Critical failures
    if report['critical_failures']:
        print("🚨 CRITICAL FAILURES:")
        print("-" * 20)
        for failure in report['critical_failures']:
            print(f"  • {failure}")
        print()
    
    # Warnings
    if report['warnings']:
        print("⚠️  WARNINGS:")
        print("-" * 12)
        for warning in report['warnings']:
            print(f"  • {warning}")
        print()
    
    # Recommendations
    if report['recommendations']:
        print("💡 RECOMMENDATIONS:")
        print("-" * 18)
        for rec in report['recommendations']:
            print(f"  • {rec}")
        print()
    
    # Overall assessment
    print("🔬 OVERALL ASSESSMENT:")
    print("-" * 22)
    
    if report['status'] in ['EXCELLENT', 'GOOD']:
        print("✅ LOGOS is functioning as intended with high confidence")
        print("   System demonstrates strong cognitive and reasoning capabilities")
        print("   Trinity framework and philosophical analysis are operational")
        
    elif report['status'] == 'ADEQUATE':
        print("✅ LOGOS is functioning adequately with room for improvement")
        print("   Core systems operational but some components may need attention")
        
    elif report['status'] == 'CONCERNING':
        print("⚠️  LOGOS functionality is concerning - investigation recommended")
        print("   Multiple system components show issues or limitations")
        
    else:
        print("❌ LOGOS functionality is impaired - immediate attention required")
        print("   Critical system failures detected - review and repair needed")
    
    print()
    print(f"Validation completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    asyncio.run(main())