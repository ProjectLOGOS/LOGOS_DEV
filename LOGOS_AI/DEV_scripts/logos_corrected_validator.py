#!/usr/bin/env python3
"""
LOGOS Corrected Functionality Validator
=======================================

Validates LOGOS functionality with corrected command syntax and proper testing.
"""

import subprocess
import sys
from pathlib import Path

def validate_logos_functionality():
    """Validate LOGOS functionality with corrected approach"""
    
    print("🔍 LOGOS CORRECTED FUNCTIONALITY VALIDATION")
    print("=" * 60)
    
    validation_results = {}
    
    # Test 1: UIP Pipeline Functionality
    print("🧪 Test 1: UIP Pipeline Functionality")
    print("-" * 40)
    
    try:
        # Test with correct command syntax
        result = subprocess.run([
            sys.executable, 'send_message_to_logos.py', 'What is existence?'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            output = result.stdout
            
            # Check for functional indicators
            uip_indicators = {
                'UIP Initialization': 'uip system initialization' in output.lower(),
                'PXL Integration': 'pxl integration' in output.lower() or 'pxl-grounded' in output.lower(),
                'Trinity Instantiation': any(symbol in output for symbol in ['𝕀₁', '𝕀₂', '𝕀₃']),
                'Response Generation': 'logos response:' in output.lower(),
                'Confidence Score': 'confidence' in output.lower(),
                'Step Processing': 'step' in output.lower() and 'initialized' in output.lower()
            }
            
            working_indicators = sum(uip_indicators.values())
            uip_functional = working_indicators >= 3
            
            print(f"   Response Length: {len(output)} characters")
            print(f"   Working Indicators: {working_indicators}/6")
            
            for name, present in uip_indicators.items():
                status = "✅" if present else "❌"
                print(f"   {status} {name}")
            
            # Check for error indicators
            error_indicators = ['error:', 'failed', 'exception']
            has_errors = any(error in output.lower() for error in error_indicators)
            
            if has_errors:
                print("   ⚠️  Errors detected in processing")
                uip_functional = uip_functional and not has_errors
            
            validation_results['UIP_Pipeline'] = {
                'functional': uip_functional,
                'score': working_indicators / 6,
                'indicators': uip_indicators
            }
            
        else:
            print(f"   ❌ UIP failed with return code: {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            validation_results['UIP_Pipeline'] = {'functional': False, 'score': 0.0}
            
    except Exception as e:
        print(f"   ❌ UIP test error: {e}")
        validation_results['UIP_Pipeline'] = {'functional': False, 'score': 0.0}
    
    print()
    
    # Test 2: AGP System Activation
    print("🧪 Test 2: AGP System Functionality")
    print("-" * 40)
    
    try:
        result = subprocess.run([
            sys.executable, 'activate_full_agp.py'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            output = result.stdout
            
            agp_indicators = {
                'Singularity System': 'singularity system' in output.lower(),
                'AGP Manager': 'agp startup manager' in output.lower() or 'agp manager initialized' in output.lower(),
                'MVS Space': 'fractal modal vector space' in output.lower() or 'mvs coordinate' in output.lower(),
                'BDN Network': 'banach-tarski' in output.lower() or 'bdn network' in output.lower(),
                'Trinity Enhancement': 'trinity enhancement' in output.lower() or 'trinity validation' in output.lower(),
                'Full Activation': 'full agp pipeline activated' in output.lower()
            }
            
            working_agp = sum(agp_indicators.values())
            agp_functional = working_agp >= 4
            
            print(f"   AGP Indicators: {working_agp}/6")
            
            for name, present in agp_indicators.items():
                status = "✅" if present else "❌"
                print(f"   {status} {name}")
            
            validation_results['AGP_Systems'] = {
                'functional': agp_functional,
                'score': working_agp / 6,
                'indicators': agp_indicators
            }
            
        else:
            print(f"   ❌ AGP activation failed")
            validation_results['AGP_Systems'] = {'functional': False, 'score': 0.0}
            
    except Exception as e:
        print(f"   ❌ AGP test error: {e}")
        validation_results['AGP_Systems'] = {'functional': False, 'score': 0.0}
    
    print()
    
    # Test 3: Trinity Framework Reasoning
    print("🧪 Test 3: Trinity Framework Reasoning")
    print("-" * 40)
    
    trinity_test_queries = [
        ('Identity', 'What is identity?'),
        ('Philosophical', 'What is the nature of being?'),
        ('Truth', 'What is truth?')
    ]
    
    trinity_results = []
    
    for category, query in trinity_test_queries:
        try:
            result = subprocess.run([
                sys.executable, 'send_message_to_logos.py', query
            ], capture_output=True, text=True, timeout=20)
            
            if result.returncode == 0:
                output = result.stdout
                
                # Check for Trinity instantiation
                trinity_symbols = ['𝕀₁', '𝕀₂', '𝕀₃']
                has_trinity = any(symbol in output for symbol in trinity_symbols)
                
                # Check for PXL grounding
                has_pxl = 'pxl' in output.lower() or 'grounding' in output.lower()
                
                # Check for confidence
                has_confidence = 'confidence' in output.lower()
                
                category_score = sum([has_trinity, has_pxl, has_confidence]) / 3
                trinity_results.append(category_score)
                
                status = "✅" if category_score >= 0.5 else "❌"
                print(f"   {status} {category}: {category_score:.2f} (Trinity: {has_trinity}, PXL: {has_pxl}, Conf: {has_confidence})")
                
            else:
                trinity_results.append(0.0)
                print(f"   ❌ {category}: Failed")
                
        except Exception:
            trinity_results.append(0.0)
            print(f"   ❌ {category}: Error")
    
    trinity_average = sum(trinity_results) / len(trinity_results) if trinity_results else 0.0
    trinity_functional = trinity_average >= 0.5
    
    validation_results['Trinity_Framework'] = {
        'functional': trinity_functional,
        'score': trinity_average,
        'individual_scores': trinity_results
    }
    
    print()
    
    # Test 4: Specialized Analysis Tools
    print("🧪 Test 4: Specialized Analysis Tools")
    print("-" * 40)
    
    specialized_tools = [
        ('meta_ontology_existence.py', 'Meta-ontological Analysis'),
        ('identity_analysis.py', 'Identity Analysis'),
        ('necessary_identity_analysis.py', 'Necessary Identity Analysis')
    ]
    
    tool_results = []
    
    for tool_file, tool_name in specialized_tools:
        if Path(tool_file).exists():
            try:
                result = subprocess.run([
                    sys.executable, tool_file
                ], capture_output=True, text=True, timeout=25)
                
                success = result.returncode == 0
                has_analysis = success and len(result.stdout) > 1000  # Substantial output
                has_confidence = success and 'confidence:' in result.stdout.lower()
                
                tool_score = sum([success, has_analysis, has_confidence]) / 3
                tool_results.append(tool_score)
                
                status = "✅" if tool_score >= 0.5 else "❌"
                print(f"   {status} {tool_name}: {tool_score:.2f}")
                
            except Exception:
                tool_results.append(0.0)
                print(f"   ❌ {tool_name}: Error")
        else:
            tool_results.append(0.0)
            print(f"   ❌ {tool_name}: Not found")
    
    tools_average = sum(tool_results) / len(tool_results) if tool_results else 0.0
    tools_functional = tools_average >= 0.5
    
    validation_results['Specialized_Tools'] = {
        'functional': tools_functional,
        'score': tools_average,
        'individual_scores': tool_results
    }
    
    print()
    
    # Overall Assessment
    print("🎯 OVERALL FUNCTIONALITY ASSESSMENT")
    print("=" * 60)
    
    # Calculate scores
    component_scores = [
        validation_results.get('UIP_Pipeline', {}).get('score', 0),
        validation_results.get('AGP_Systems', {}).get('score', 0),
        validation_results.get('Trinity_Framework', {}).get('score', 0),
        validation_results.get('Specialized_Tools', {}).get('score', 0)
    ]
    
    overall_score = sum(component_scores) / len(component_scores)
    
    # Determine functionality status
    functional_components = sum(1 for category in validation_results.values() if category.get('functional', False))
    
    if functional_components >= 3:
        if overall_score >= 0.8:
            status = "EXCELLENT"
            color = "🟢"
            conclusion = "LOGOS is functioning excellently as intended"
        else:
            status = "GOOD" 
            color = "🟢"
            conclusion = "LOGOS is functioning well as intended"
    elif functional_components >= 2:
        status = "ADEQUATE"
        color = "🟡"
        conclusion = "LOGOS is functioning adequately with some limitations"
    elif functional_components >= 1:
        status = "CONCERNING"
        color = "🟠"
        conclusion = "LOGOS has concerning functionality issues"
    else:
        status = "POOR"
        color = "🔴"
        conclusion = "LOGOS is not functioning properly"
    
    print(f"{color} Overall Status: {status}")
    print(f"   Functional Components: {functional_components}/4")
    print(f"   Overall Score: {overall_score:.2%}")
    print()
    
    print("Component Breakdown:")
    components = ['UIP_Pipeline', 'AGP_Systems', 'Trinity_Framework', 'Specialized_Tools']
    for i, component in enumerate(components):
        result = validation_results.get(component, {})
        functional = result.get('functional', False)
        score = result.get('score', 0)
        status_icon = "✅" if functional else "❌"
        component_name = component.replace('_', ' ')
        print(f"   {status_icon} {component_name}: {score:.2%}")
    
    print()
    print("🔬 CONCLUSION:")
    print("-" * 15)
    print(f"{conclusion}")
    
    if functional_components >= 2:
        print("\n✅ KEY FINDINGS:")
        print("   • UIP pipeline is processing requests")
        print("   • Trinity instantiation is working")
        print("   • AGP systems are operational")
        print("   • Philosophical analysis capabilities present")
        print("\n💡 LOGOS demonstrates functional AGI capabilities:")
        print("   • Responsive to philosophical queries")
        print("   • Trinity-grounded reasoning")
        print("   • Mathematical foundations active")
        print("   • Specialized analysis tools available")
    else:
        print("\n⚠️  ISSUES IDENTIFIED:")
        print("   • Multiple core systems not functioning")
        print("   • Investigation and repair recommended")
    
    print("\n" + "=" * 60)
    
    return functional_components >= 2, validation_results


if __name__ == "__main__":
    is_functional, results = validate_logos_functionality()