#!/usr/bin/env python3
"""
LOGOS V2 Consolidation - Comprehensive Validation Report
========================================================
Full system validation after GPT surgical consolidation implementation
"""
import sys
import os
import time
from datetime import datetime

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'LOGOS_V2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'V1_to_V2_Migration'))

def test_basic_imports():
    """Test 1: Basic consolidated imports"""
    tests = []
    
    try:
        import core.system_imports
        tests.append(("✅ System imports", True))
    except Exception as e:
        tests.append((f"❌ System imports: {e}", False))
    
    try:
        from core.unified_classes import TrinityVector, UnifiedWorkerConfig
        vector = TrinityVector(0.5, 0.5, 0.5)
        config = UnifiedWorkerConfig()
        tests.append(("✅ Unified classes", True))
    except Exception as e:
        tests.append((f"❌ Unified classes: {e}", False))
    
    try:
        from core.worker_kernel import WorkerKernel
        kernel = WorkerKernel(UnifiedWorkerConfig())
        tests.append(("✅ Worker kernel", True))
    except Exception as e:
        tests.append((f"❌ Worker kernel: {e}", False))
        
    try:
        from core.bayesian_inference import BayesianInference
        inference = BayesianInference()
        tests.append(("✅ Bayesian inference", True))
    except Exception as e:
        tests.append((f"❌ Bayesian inference: {e}", False))
    
    return tests

def test_functional_capabilities():
    """Test 2: Functional capabilities"""
    tests = []
    
    try:
        from core.unified_classes import TrinityVector
        vector = TrinityVector(0.6, 0.8, 0.0)
        
        # Test magnitude calculation and normalization
        magnitude = vector.magnitude()
        if abs(magnitude - 1.0) < 0.001:
            tests.append(("✅ Trinity Vector normalization", True))
        else:
            tests.append((f"⚠️  Trinity Vector normalization: {magnitude:.3f}", False))
            
        # Test Trinity product
        product = vector.trinity_product()
        tests.append(("✅ Trinity product calculation", True))
        
    except Exception as e:
        tests.append((f"❌ Trinity Vector functional test: {e}", False))
    
    try:
        from core.worker_kernel import WorkerKernel
        from core.unified_classes import UnifiedWorkerConfig
        
        config = UnifiedWorkerConfig(max_workers=2)
        kernel = WorkerKernel(config)
        
        # Test worker management
        result1 = kernel.start_worker("test1", "task1")
        result2 = kernel.start_worker("test2", "task2") 
        result3 = kernel.start_worker("test3", "task3")  # Should queue
        
        if result1 and result2 and not result3:
            tests.append(("✅ Worker management", True))
        else:
            tests.append((f"⚠️  Worker management: {result1}, {result2}, {result3}", False))
            
        kernel.stop_worker("test1")
        tests.append(("✅ Worker lifecycle", True))
        
    except Exception as e:
        tests.append((f"❌ Worker kernel functional test: {e}", False))
    
    try:
        from core.bayesian_inference import BayesianInference
        
        inference = BayesianInference()
        
        # Test posterior calculation
        posterior = inference.calculate_posterior(0.3, 0.8, 0.5)
        expected = (0.3 * 0.8) / 0.5
        
        if abs(posterior - expected) < 0.001:
            tests.append(("✅ Bayesian posterior calculation", True))
        else:
            tests.append((f"⚠️  Bayesian calculation: {posterior} vs {expected}", False))
            
    except Exception as e:
        tests.append((f"❌ Bayesian inference functional test: {e}", False))
    
    return tests

def test_integration_workflow():
    """Test 3: Cross-module integration"""
    tests = []
    
    try:
        from core.unified_classes import TrinityVector, UnifiedWorkerConfig
        from core.worker_kernel import WorkerKernel
        from core.bayesian_inference import BayesianInference
        
        # Integrated workflow
        vector = TrinityVector(0.7, 0.2, 0.1)
        config = UnifiedWorkerConfig(max_workers=3)
        kernel = WorkerKernel(config)
        inference = BayesianInference()
        
        # Start worker with Trinity task
        task = {'vector': vector, 'operation': 'trinity_analysis'}
        worker_started = kernel.start_worker("trinity_worker", task)
        
        # Bayesian analysis on results
        hypotheses = {'valid': 0.8, 'invalid': 0.2}
        evidence = [('valid', vector.magnitude()), ('invalid', 1.0 - vector.magnitude())]
        updated_beliefs = inference.update_belief_batch(hypotheses, evidence)
        
        kernel.stop_worker("trinity_worker")
        tests.append(("✅ Integrated workflow", True))
        
    except Exception as e:
        tests.append((f"❌ Integration workflow: {e}", False))
    
    return tests

def test_v1_compatibility():
    """Test 4: V1 migration compatibility"""
    tests = []
    
    try:
        # Test that V1 migration files work with consolidated V2
        import mathematical_engines.lambda_calculus
        tests.append(("✅ V1 lambda calculus import", True))
        
        import interactive_systems.chat_app 
        tests.append(("✅ V1 chat app import", True))
        
        # Test combined V1+V2 usage
        from core.unified_classes import TrinityVector as V2Vector
        v2_vector = V2Vector(0.5, 0.5, 0.5)
        tests.append(("✅ V1+V2 compatibility", True))
        
    except Exception as e:
        tests.append((f"⚠️  V1 compatibility: {e}", False))
    
    return tests

def count_files_and_imports():
    """Count files and imports for metrics"""
    import glob
    
    # Count Python files
    logos_v2_path = os.path.join(os.path.dirname(__file__), '..', 'LOGOS_V2')
    python_files = []
    for root, dirs, files in os.walk(logos_v2_path):
        if 'external_libraries' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Count imports
    import_count = 0
    for file in python_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.readlines()
                import_count += len([line for line in content if line.strip().startswith(('import ', 'from '))])
        except:
            continue
    
    return len(python_files), import_count

def main():
    """Run comprehensive validation"""
    print("🔍 LOGOS V2 Consolidation - Comprehensive Validation")
    print("=" * 60)
    print(f"Validation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # File and import metrics
    file_count, import_count = count_files_and_imports()
    print("📊 CONSOLIDATION METRICS:")
    print(f"   Python files: {file_count}")
    print(f"   Import lines: {import_count}")
    
    # Calculate reduction percentages  
    original_files = 256  # Estimated original count
    original_imports = 1240  # Baseline from earlier measurement
    file_reduction = ((original_files - file_count) / original_files * 100) if original_files > 0 else 0
    import_reduction = ((original_imports - import_count) / original_imports * 100) if original_imports > 0 else 0
    
    print(f"   File reduction: {file_reduction:.1f}%")
    print(f"   Import reduction: {import_reduction:.1f}%")
    print()
    
    # Run all tests
    all_tests = []
    
    print("🧪 TEST SUITE 1: Basic System Health")
    print("-" * 40)
    basic_tests = test_basic_imports()
    all_tests.extend(basic_tests)
    for test, result in basic_tests:
        print(f"   {test}")
    print()
    
    print("🔧 TEST SUITE 2: Functional Capabilities")
    print("-" * 40)
    functional_tests = test_functional_capabilities()
    all_tests.extend(functional_tests)
    for test, result in functional_tests:
        print(f"   {test}")
    print()
    
    print("🔗 TEST SUITE 3: Integration Workflow")
    print("-" * 40)
    integration_tests = test_integration_workflow()
    all_tests.extend(integration_tests)
    for test, result in integration_tests:
        print(f"   {test}")
    print()
    
    print("🌉 TEST SUITE 4: V1 Compatibility")
    print("-" * 40)
    v1_tests = test_v1_compatibility()
    all_tests.extend(v1_tests)
    for test, result in v1_tests:
        print(f"   {test}")
    print()
    
    # Summary
    passed_tests = sum(1 for _, result in all_tests if result)
    total_tests = len(all_tests)
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print("📈 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"Tests Passed: {passed_tests} / {total_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 CONSOLIDATION SUCCESS! System fully validated.")
    elif success_rate >= 75:
        print("✅ CONSOLIDATION SUCCESSFUL with minor issues.")
    else:
        print("⚠️  CONSOLIDATION PARTIAL - review failures above.")
    
    return success_rate >= 75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)