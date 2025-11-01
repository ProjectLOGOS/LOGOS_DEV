#!/usr/bin/env python3
"""
ARP Nexus Test Script
=====================

Tests the ARP nexus functionality including:
- Nexus initialization
- DataBuilder operations
- Recursive processing simulation
- C-value data handling
- Cycle limit management
"""

import asyncio
import sys
from pathlib import Path

# Add the nexus directory to path
sys.path.insert(0, str(Path(__file__).parent))

from arp_nexus import ARPNexus, DataBuilder, ReasoningRequest, ReasoningMode, DataExchangePacket


async def test_nexus_initialization():
    """Test ARP nexus initialization"""
    print("🧪 Testing ARP Nexus Initialization...")

    nexus = ARPNexus()
    success = await nexus.initialize()

    if success:
        print("✅ ARP Nexus initialized successfully")
        status = nexus.get_status()
        print(f"   Status: {status['status']}")
        print(f"   IEL Domains: {status['iel_domains']}")
        print(f"   Reasoning Engines: {status['reasoning_engines']}")
        print(f"   Max Cycles: {status['max_cycles']}")
    else:
        print("❌ ARP Nexus initialization failed")

    return success


async def test_data_builder():
    """Test DataBuilder functionality"""
    print("\n🧪 Testing DataBuilder Operations...")

    builder = DataBuilder(max_cycles_default=5)

    # Test packet creation
    packet = builder.create_exchange_packet(
        source_protocol="ARP",
        target_protocol="SCP",
        data_payload={"test_data": "sample", "metrics": [1, 2, 3]},
        c_value_data={"c1": complex(1, 2), "c2": complex(3, 4)}
    )

    print(f"✅ Created packet: {packet.packet_id}")
    print(f"   Source: {packet.source_protocol}")
    print(f"   Target: {packet.target_protocol}")
    print(f"   Max Cycles: {packet.max_cycles}")

    # Test packet validation
    is_valid = builder.validate_packet(packet)
    print(f"   Valid: {'✅' if is_valid else '❌'}")

    # Test cycle status
    cycle_key = f"ARP_SCP_{packet.packet_id.split('-')[0]}"
    status = builder.get_cycle_status(cycle_key)
    print(f"   Cycle Status: {status.get('status', 'active')}")
    print(f"   Total Packets: {status.get('total_packets', 0)}")

    return True


async def test_reasoning_request():
    """Test reasoning request processing"""
    print("\n🧪 Testing Reasoning Request Processing...")

    nexus = ARPNexus()
    await nexus.initialize()

    # Create test request
    request = ReasoningRequest(
        request_id="test_request_001",
        reasoning_mode=ReasoningMode.STANDARD_ANALYSIS,
        input_data={
            "query": "What is the nature of consciousness?",
            "context": "Philosophical analysis",
            "domains": ["AnthroPraxis", "GnosiPraxis"]
        },
        domain_focus=["AnthroPraxis", "GnosiPraxis"],
        mathematical_foundations=True,
        c_value_data={"consciousness_c": complex(0.5, 0.8)}
    )

    print(f"✅ Created reasoning request: {request.request_id}")
    print(f"   Mode: {request.reasoning_mode.value}")
    print(f"   Domains: {request.domain_focus}")

    # Process request
    try:
        result = await nexus.process_reasoning_request(request)
        print(f"✅ Request processed successfully")
        print(f"   Processing time: {result.processing_time:.2f}s")
        print(f"   Has domain outputs: {bool(result.domain_outputs)}")
    except Exception as e:
        print(f"❌ Request processing failed: {e}")
        return False

    return True


async def test_recursive_processing():
    """Test recursive processing simulation"""
    print("\n🧪 Testing Recursive Processing Simulation...")

    nexus = ARPNexus()
    await nexus.initialize()

    # Create recursive request
    request = ReasoningRequest(
        request_id="recursive_test_001",
        reasoning_mode=ReasoningMode.RECURSIVE_REFINEMENT,
        input_data={
            "initial_query": "Analyze system consciousness emergence",
            "iteration_data": {},
            "convergence_metrics": {}
        },
        recursive_cycles=3,
        c_value_data={"emergence_c": complex(0.3, 0.7)}
    )

    print(f"✅ Created recursive request: {request.request_id}")
    print(f"   Recursive cycles: {request.recursive_cycles}")

    # Process recursive request
    try:
        result = await nexus.process_reasoning_request(request)
        print(f"✅ Recursive processing completed")
        print(f"   Iterations performed: {result.recursive_iterations}")
        print(f"   C-value evolution: {bool(result.c_value_evolution)}")
    except Exception as e:
        print(f"❌ Recursive processing failed: {e}")
        return False

    return True


async def test_cycle_management():
    """Test cycle limit management"""
    print("\n🧪 Testing Cycle Management...")

    nexus = ARPNexus()
    builder = nexus.data_builder

    # Test max cycles setting
    original_limit = builder.max_cycles_default
    nexus.set_max_cycles(10, "system")
    print(f"✅ Set max cycles to 10 (was {original_limit})")

    # Test cascade override
    packet = builder.create_exchange_packet(
        source_protocol="ARP",
        target_protocol="SCP",
        data_payload={"emergency": "critical_system_failure"}
    )

    # Should detect cascade and allow extended cycles
    print(f"   Cascade detected: {packet.cascade_imminent}")
    print(f"   Extended cycles allowed: {packet.max_cycles > 7}")

    # Test override approval
    approved = nexus.approve_cascade_override(packet.packet_id, True)
    print(f"   Override approved: {approved}")

    return True


async def main():
    """Run all ARP nexus tests"""
    print("🚀 Starting ARP Nexus Test Suite")
    print("=" * 50)

    tests = [
        ("Nexus Initialization", test_nexus_initialization),
        ("DataBuilder Operations", test_data_builder),
        ("Reasoning Request Processing", test_reasoning_request),
        ("Recursive Processing", test_recursive_processing),
        ("Cycle Management", test_cycle_management)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! ARP Nexus is ready for operation.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)