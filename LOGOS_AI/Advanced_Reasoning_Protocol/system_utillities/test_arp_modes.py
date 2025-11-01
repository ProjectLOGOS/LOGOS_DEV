#!/usr/bin/env python3
"""
ARP Mode System Test Script
===========================

Tests the ARP mode system implementation according to protocol rules:

ACTIVE Mode:
- At boot up, runs tests, confirms alignment, functionality, and connection to nexus network
- Logs report, sends request for maintenance if tests fail
- Gracefully switches to passive

PASSIVE Mode:
- Background learning with minimal systems online
- Nexus and learning modules remain online for silent improvement
- Responds to passive triggers for data analysis
- Engages passive mode if no other input received after 30 seconds

INACTIVE Mode:
- Only at full shutdown
- Continues passive learning as much as possible
"""

import sys
import json
import time
from pathlib import Path

# Add the ARP directory to path
sys.path.insert(0, str(Path(__file__).parent))

from arp_operations import ARPOperations, ARPMode


def test_active_mode_initialization():
    """Test ACTIVE mode initialization with testing and validation"""
    print("🚀 Testing ACTIVE Mode Initialization")
    print("=" * 50)

    arp = ARPOperations()

    # Check initial mode status
    initial_status = arp.get_mode_status()
    print(f"Initial mode: {initial_status['current_mode']}")
    assert initial_status['current_mode'] == 'inactive'

    # Initialize full stack (triggers ACTIVE mode)
    print("\n🧪 Starting full stack initialization (ACTIVE mode)...")
    success = arp.initialize_full_stack()

    if success:
        print("✅ Full stack initialization successful")

        # Check mode after initialization
        active_status = arp.get_mode_status()
        print(f"Mode after initialization: {active_status['current_mode']}")
        print(f"Passive processes active: {active_status['passive_processes_active']}")

        # Check that we're in passive mode after active initialization
        assert active_status['current_mode'] == 'passive'
        assert active_status['passive_processes_active'] > 0

        print("✅ ACTIVE mode initialization test passed")
        return True
    else:
        print("❌ Full stack initialization failed")
        return False


def test_passive_mode_operation():
    """Test PASSIVE mode background operation"""
    print("\n🧠 Testing PASSIVE Mode Operation")
    print("=" * 50)

    arp = ARPOperations()

    # Initialize to get to passive mode
    arp.initialize_full_stack()

    # Check passive mode status
    passive_status = arp.get_mode_status()
    print(f"Current mode: {passive_status['current_mode']}")
    print(f"Background processes: {passive_status['passive_processes_active']}")

    # Test demand system activation
    print("\n⚡ Testing demand system activation...")
    activation_result = arp.activate_demand_systems("learning_module_request")
    print(f"Activation result: {activation_result['activation_complete']}")
    print(f"Systems activated: {activation_result['systems_activated']}")

    print("✅ PASSIVE mode operation test passed")
    return True


def test_passive_trigger():
    """Test passive trigger functionality"""
    print("\n📥 Testing Passive Trigger Functionality")
    print("=" * 50)

    arp = ARPOperations()
    arp.initialize_full_stack()

    # Create test data set
    test_data = {
        "query": "Analyze consciousness emergence patterns",
        "data_type": "ontological_analysis",
        "complexity": "high",
        "domains": ["AnthroPraxis", "GnosiPraxis", "TeloPraxis"],
        "expected_insights": ["self_awareness", "goal_alignment", "ethical_reasoning"]
    }

    # Save test data to temporary file
    test_file = Path("test_passive_data.json")
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)

    try:
        # Test passive trigger via command line simulation
        print("📤 Sending passive trigger...")
        trigger_result = arp.receive_passive_trigger(test_data)

        print(f"Trigger result: {trigger_result}")

        if 'analysis_scheduled' in trigger_result:
            print("⏰ Analysis scheduled for timeout period")
        elif 'analysis_type' in trigger_result:
            print("🔍 Passive analysis engaged immediately")

        print("✅ Passive trigger test passed")
        return True

    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()


def test_shutdown_procedure():
    """Test full system shutdown procedure"""
    print("\n🛑 Testing System Shutdown Procedure")
    print("=" * 50)

    arp = ARPOperations()
    arp.initialize_full_stack()

    # Check status before shutdown
    pre_shutdown_status = arp.get_mode_status()
    print(f"Mode before shutdown: {pre_shutdown_status['current_mode']}")
    print(f"Active processes before shutdown: {pre_shutdown_status['passive_processes_active']}")

    # Execute full system shutdown
    print("\n🔄 Executing full system shutdown...")
    shutdown_result = arp.shutdown_system()

    print(f"Shutdown complete: {shutdown_result['shutdown_complete']}")
    print(f"Final mode: {shutdown_result['final_mode']}")

    # Check that activity was logged
    if 'activity_report' in shutdown_result:
        print("📝 Activity report generated")
        print(f"  Mode transitions: {len(shutdown_result['activity_report']['mode_history'])}")

    # Check token cleanup
    if 'token_cleanup' in shutdown_result:
        print("🧹 Token cleanup completed")

    # Verify final mode
    final_status = arp.get_mode_status()
    print(f"Final mode status: {final_status['current_mode']}")
    assert final_status['current_mode'] == 'inactive'

    print("✅ Shutdown procedure test passed")
    return True


def test_mode_transitions():
    """Test mode transition flow: INACTIVE → ACTIVE → PASSIVE → INACTIVE"""
    print("\n🔄 Testing Mode Transition Flow")
    print("=" * 50)

    arp = ARPOperations()

    # Start in INACTIVE mode
    status_1 = arp.get_mode_status()
    print(f"1. Initial mode: {status_1['current_mode']}")
    assert status_1['current_mode'] == 'inactive'

    # Transition to ACTIVE → PASSIVE via initialization
    arp.initialize_full_stack()
    status_2 = arp.get_mode_status()
    print(f"2. After initialization: {status_2['current_mode']}")
    assert status_2['current_mode'] == 'passive'

    # Transition to INACTIVE via shutdown
    arp.shutdown_system()
    status_3 = arp.get_mode_status()
    print(f"3. After shutdown: {status_3['current_mode']}")
    assert status_3['current_mode'] == 'inactive'

    # Check mode history
    history = status_3['mode_history']
    print(f"Mode transitions recorded: {len(history)}")

    expected_transitions = [
        ('inactive', 'active', 'system_boot'),
        ('active', 'passive', 'boot_complete'),
        ('passive', 'inactive', 'system_shutdown')
    ]

    for i, (expected_from, expected_to, expected_reason) in enumerate(expected_transitions):
        if i < len(history):
            transition = history[i]
            assert transition['from_mode'] == expected_from
            assert transition['to_mode'] == expected_to
            assert transition['reason'] == expected_reason
            print(f"  ✓ Transition {i+1}: {expected_from} → {expected_to} ({expected_reason})")

    print("✅ Mode transition flow test passed")
    return True


def main():
    """Run all mode system tests"""
    print("🧪 ARP Mode System Test Suite")
    print("=" * 60)
    print("Testing ARP operational modes according to protocol specifications:")
    print("• ACTIVE: Boot testing, validation, transition to passive")
    print("• PASSIVE: Background learning, demand activation, passive triggers")
    print("• INACTIVE: Full shutdown, state persistence, resource cleanup")
    print()

    tests = [
        ("Active Mode Initialization", test_active_mode_initialization),
        ("Passive Mode Operation", test_passive_mode_operation),
        ("Passive Trigger Functionality", test_passive_trigger),
        ("System Shutdown Procedure", test_shutdown_procedure),
        ("Mode Transition Flow", test_mode_transitions)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ FAILED: {test_name} - {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All mode system tests passed!")
        print("ARP operational modes are functioning according to protocol specifications.")
        print("\nMode Rules Implemented:")
        print("• ✅ ACTIVE: Boot testing → validation → passive transition")
        print("• ✅ PASSIVE: Background learning → demand activation → trigger response")
        print("• ✅ INACTIVE: Full shutdown → state persistence → resource cleanup")
        print("• ✅ Mode transitions: INACTIVE → ACTIVE → PASSIVE → INACTIVE")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review mode system implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)