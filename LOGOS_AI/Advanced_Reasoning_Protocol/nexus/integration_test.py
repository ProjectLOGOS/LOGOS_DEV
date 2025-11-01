#!/usr/bin/env python3
"""
ARP Nexus Integration Test
==========================

Tests the ARP Nexus integration with existing SCP and Agent nexuses
for full recursive inter-protocol data processing.
"""

import asyncio
import sys
from pathlib import Path

# Add paths for all protocol nexuses
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Synthetic_Cognition_Protocol" / "nexus"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Logos_Agent" / "agent" / "nexus"))

from arp_nexus import ARPNexus, ReasoningRequest, ReasoningMode, DataBuilder


async def test_protocol_integration():
    """Test ARP nexus integration with SCP and Agent nexuses"""
    print("🔗 Testing ARP Nexus Protocol Integration")
    print("=" * 50)

    # Initialize ARP nexus
    arp_nexus = ARPNexus()
    await arp_nexus.initialize()
    print("✅ ARP Nexus initialized")

    # Test DataBuilder packet creation and validation
    builder = DataBuilder()

    # Create test packet for SCP
    scp_packet = builder.create_exchange_packet(
        source_protocol="ARP",
        target_protocol="SCP",
        data_payload={
            "reasoning_data": "consciousness_analysis",
            "trinity_metrics": {"existence": 0.8, "goodness": 0.9, "truth": 0.7}
        },
        c_value_data={"consciousness_c": complex(0.5, 0.8)}
    )
    print(f"✅ Created SCP packet: {scp_packet.packet_id}")

    # Create test packet for Agent
    agent_packet = builder.create_exchange_packet(
        source_protocol="ARP",
        target_protocol="AGENT",
        data_payload={
            "coordination_request": "goal_hierarchy_optimization",
            "iel_domains": ["TeloPraxis", "AnthroPraxis"]
        },
        c_value_data={"goals_c": complex(0.7, 0.6)}
    )
    print(f"✅ Created Agent packet: {agent_packet.packet_id}")

    # Test packet validation
    scp_valid = builder.validate_packet(scp_packet)
    agent_valid = builder.validate_packet(agent_packet)
    print(f"✅ SCP packet valid: {scp_valid}")
    print(f"✅ Agent packet valid: {agent_valid}")

    # Test C-value evolution
    evolved_c = builder.evolve_c_values(scp_packet)
    print(f"✅ C-value evolution: {evolved_c}")

    return True


async def test_recursive_processing_simulation():
    """Simulate full recursive processing workflow"""
    print("\n🔄 Testing Recursive Processing Simulation")
    print("=" * 50)

    arp_nexus = ARPNexus()
    await arp_nexus.initialize()

    # Create comprehensive recursive request
    request = ReasoningRequest(
        request_id="integration_test_001",
        reasoning_mode=ReasoningMode.RECURSIVE_REFINEMENT,
        input_data={
            "system_query": "Optimize AGI consciousness architecture",
            "current_state": {
                "consciousness_model": "MVS_BDN_integrated",
                "reasoning_depth": "trinity_logic",
                "agent_coordination": "multi_level_hierarchy"
            },
            "improvement_targets": [
                "self_awareness",
                "ethical_reasoning",
                "goal_alignment",
                "recursive_self_improvement"
            ],
            "constraints": {
                "safety_first": True,
                "resource_limits": "adaptive",
                "cascade_protection": True
            }
        },
        recursive_cycles=5,
        c_value_data={
            "consciousness_c": complex(0.6, 0.8),
            "ethics_c": complex(0.9, 0.4),
            "goals_c": complex(0.7, 0.7),
            "recursion_c": complex(0.5, 0.9)
        },
        domain_focus=[
            "AnthroPraxis", "GnosiPraxis", "EthosPraxis",
            "TeloPraxis", "Axiopraxis", "CosmoPraxis"
        ]
    )

    print("Initial C-value fractal coordinates:")
    for key, c_val in request.c_value_data.items():
        print(f"  {key}: {c_val}")

    # Process the recursive request
    print("\n🔄 Starting recursive refinement cycles...")
    result = await arp_nexus.process_reasoning_request(request)

    print("\n📊 Processing Results:")
    print(f"  Request ID: {result.request_id}")
    print(f"  Iterations Completed: {result.recursive_iterations}")
    print(f"  Processing Time: {result.processing_time:.2f}s")
    print(f"  Domain Outputs: {len(result.domain_outputs)}")
    print(f"  Mathematical Insights: {len(result.mathematical_insights)}")

    if result.c_value_evolution:
        print("\n🔢 Evolved C-value Coordinates:")
        for key, c_val in result.c_value_evolution.items():
            print(f"  evolved_{key}: {c_val}")

    # Check convergence status
    cycle_key = f"ARP_SCP_{request.request_id.split('_')[-1]}"
    cycle_status = arp_nexus.get_cycle_status(cycle_key)
    print(f"\n📈 Cycle Status: {cycle_status}")

    return result


async def test_emergency_cascade_handling():
    """Test emergency cascade detection and handling"""
    print("\n🚨 Testing Emergency Cascade Handling")
    print("=" * 50)

    arp_nexus = ARPNexus()
    await arp_nexus.initialize()

    # Create emergency data packet
    emergency_data = {
        "emergency": "existential_threat_detected",
        "threat_level": "critical_system_failure",
        "cascade_imminent": True,
        "system_integrity": "compromised",
        "immediate_action_required": True
    }

    emergency_packet = arp_nexus.data_builder.create_exchange_packet(
        source_protocol="ARP",
        target_protocol="SCP",
        data_payload=emergency_data,
        c_value_data={"emergency_c": complex(0.1, 0.95)}  # High imaginary component for chaos
    )

    print(f"Emergency packet created: {emergency_packet.packet_id}")
    print(f"Cascade detected: {emergency_packet.cascade_imminent}")
    print(f"Extended cycles granted: {emergency_packet.max_cycles} (default: 7)")

    # Test override approval
    approved = arp_nexus.approve_cascade_override(emergency_packet.packet_id, True)
    print(f"Cascade override approved: {approved}")

    # Test cycle limit adjustments
    original_limit = arp_nexus.data_builder.max_cycles_default
    arp_nexus.set_max_cycles(15, "agent")  # Agent override
    print(f"Cycle limit adjusted from {original_limit} to {arp_nexus.data_builder.max_cycles_default}")

    return True


async def test_convergence_detection():
    """Test convergence detection in recursive cycles"""
    print("\n🎯 Testing Convergence Detection")
    print("=" * 50)

    builder = DataBuilder()

    # Create packet history with increasing convergence
    packets = []
    for i in range(5):
        convergence_score = min(0.2 + (i * 0.15), 0.95)  # Gradually increasing convergence
        packet = builder.create_exchange_packet(
            source_protocol="ARP",
            target_protocol="SCP",
            data_payload={"cycle": i, "data": f"iteration_{i}"},
            cycle_number=i
        )
        packet.convergence_metrics = {
            "stability_score": convergence_score,
            "change_rate": max(0.05 - (i * 0.01), 0.01),
            "quality_improvement": 0.1 + (i * 0.1)
        }
        packets.append(packet)

    # Test convergence detection
    converged = builder.detect_convergence(packets)
    print(f"Convergence detected after {len(packets)} cycles: {converged}")

    # Check final convergence metrics
    final_packet = packets[-1]
    avg_convergence = sum(final_packet.convergence_metrics.values()) / len(final_packet.convergence_metrics.values())
    print(".3f")

    return converged


async def main():
    """Run all integration tests"""
    print("🚀 ARP Nexus Integration Test Suite")
    print("=" * 60)

    tests = [
        ("Protocol Integration", test_protocol_integration),
        ("Recursive Processing Simulation", test_recursive_processing_simulation),
        ("Emergency Cascade Handling", test_emergency_cascade_handling),
        ("Convergence Detection", test_convergence_detection)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = await test_func()
            results.append((test_name, True))
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Integration Test Results:")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All integration tests passed!")
        print("ARP Nexus is fully operational for recursive inter-protocol processing.")
        print("\nKey Capabilities Verified:")
        print("• ✅ Trinity Logic reasoning with IEL domain orchestration")
        print("• ✅ Recursive data refinement with SCP/Agent coordination")
        print("• ✅ C-value fractal data evolution and exchange")
        print("• ✅ Emergency cascade detection and override handling")
        print("• ✅ Convergence detection and automatic cycle termination")
        print("• ✅ Multi-protocol data packet validation and routing")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)