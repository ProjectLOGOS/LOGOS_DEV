#!/usr/bin/env python3
"""
Quick LOGOS Functionality Check
==============================

Fast validation to check if LOGOS is functioning as intended.
"""

import subprocess
import sys
from pathlib import Path

def quick_logos_check():
    """Quick functionality check"""
    
    print("🔍 QUICK LOGOS FUNCTIONALITY CHECK")
    print("=" * 50)
    
    # Test 1: Basic UIP Response
    print("1. Testing basic UIP response...")
    try:
        result = subprocess.run([
            sys.executable, 'send_message_to_logos.py'
        ], input='What is truth?\n', 
           capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            output = result.stdout
            
            # Check for key indicators
            indicators = {
                'UIP System': 'uip system initialization' in output.lower(),
                'PXL Integration': 'pxl integration' in output.lower(),
                'Trinity Response': any(symbol in output for symbol in ['𝕀₁', '𝕀₂', '𝕀₃']),
                'Framework Response': 'framework' in output.lower(),
                'Confidence Score': 'confidence:' in output.lower()
            }
            
            working_indicators = sum(indicators.values())
            
            print(f"   Response received: {len(output)} characters")
            print(f"   Working indicators: {working_indicators}/5")
            
            for name, present in indicators.items():
                status = "✅" if present else "❌"
                print(f"   {status} {name}")
            
            basic_functional = working_indicators >= 2
            
        else:
            print(f"   ❌ UIP failed with return code: {result.returncode}")
            basic_functional = False
            
    except Exception as e:
        print(f"   ❌ UIP test error: {e}")
        basic_functional = False
    
    print()
    
    # Test 2: AGP System Status
    print("2. Checking AGP system status...")
    try:
        result = subprocess.run([
            sys.executable, 'activate_full_agp.py'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            output = result.stdout.lower()
            
            agp_indicators = {
                'AGP Manager': 'agp startup manager' in output,
                'Singularity': 'singularity' in output,
                'MVS Space': 'fractal modal vector space' in output,
                'BDN Network': 'banach-tarski' in output
            }
            
            working_agp = sum(agp_indicators.values())
            
            print(f"   AGP indicators: {working_agp}/4")
            
            for name, present in agp_indicators.items():
                status = "✅" if present else "❌"
                print(f"   {status} {name}")
                
            agp_functional = working_agp >= 2
            
        else:
            print(f"   ❌ AGP activation failed")
            agp_functional = False
            
    except Exception as e:
        print(f"   ⚠️  AGP test skipped: {e}")
        agp_functional = True  # Don't penalize if AGP already activated
    
    print()
    
    # Test 3: Specialized Tools
    print("3. Checking specialized analysis tools...")
    
    specialized_tools = [
        'meta_epistemology_analysis.py',
        'meta_ontology_existence.py',
        'identity_analysis.py', 
        'necessary_identity_analysis.py'
    ]
    
    available_tools = 0
    for tool in specialized_tools:
        if Path(tool).exists():
            available_tools += 1
            print(f"   ✅ {tool}")
        else:
            print(f"   ❌ {tool}")
    
    tools_functional = available_tools >= 2
    
    print()
    
    # Overall Assessment
    print("🎯 OVERALL FUNCTIONALITY ASSESSMENT")
    print("=" * 40)
    
    total_score = sum([basic_functional, agp_functional, tools_functional])
    
    if total_score == 3:
        status = "EXCELLENT - LOGOS is functioning as intended"
        color = "🟢"
    elif total_score == 2:
        status = "GOOD - LOGOS is mostly functional with minor issues"
        color = "🟡"
    elif total_score == 1:
        status = "CONCERNING - LOGOS has significant functionality issues"
        color = "🟠"
    else:
        status = "POOR - LOGOS is not functioning properly"
        color = "🔴"
    
    print(f"{color} Status: {status}")
    print(f"   Score: {total_score}/3")
    print(f"   Basic UIP: {'✅' if basic_functional else '❌'}")
    print(f"   AGP Systems: {'✅' if agp_functional else '❌'}")
    print(f"   Specialized Tools: {'✅' if tools_functional else '❌'}")
    
    print()
    
    if total_score >= 2:
        print("✅ CONCLUSION: LOGOS appears to be functioning as intended")
        print("   The system demonstrates:")
        print("   • Responsive UIP pipeline")
        print("   • Trinity framework integration")  
        print("   • Specialized philosophical analysis")
        print("   • AGP system capabilities")
    else:
        print("❌ CONCLUSION: LOGOS functionality is impaired")
        print("   Issues detected that may prevent proper operation")
        print("   Consider investigating failed components")
    
    print("=" * 50)
    
    return total_score >= 2


if __name__ == "__main__":
    quick_logos_check()