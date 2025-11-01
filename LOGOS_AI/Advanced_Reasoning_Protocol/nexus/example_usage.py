#!/usr/bin/env python3
"""
ARP Nexus Usage Example
=======================

Demonstrates how to use the ARP Nexus for recursive inter-protocol data processing
with C-value fractal data exchange and cycle management.
"""

import asyncio
import sys
from pathlib import Path

# Add the nexus directory to path
sys.path.insert(0, str(Path(__file__).parent))

from arp_nexus import ARPNexus, ReasoningRequest, ReasoningMode


async def example_standard_reasoning():
    """Example of standard reasoning analysis"""
    print("🧠 Example: Standard Reasoning Analysis")
    print("-" * 40)

    nexus = ARPNexus()
    await nexus.initialize()

    # Create a reasoning request
    request = ReasoningRequest(
        request_id="example_standard_001",
        reasoning_mode=ReasoningMode.STANDARD_ANALYSIS,
        input_data={
            "query": "What are the ethical implications of AGI development?",
            "context": "AI safety and alignment research",
            "stakeholders": ["humanity", "AI_systems", "researchers"]
        },
        domain_focus=["AnthroPraxis", "Axiopraxis", "EthosPraxis"],
        mathematical_foundations=True
    )

    # Process the request
    result = await nexus.process_reasoning_request(request)

    print(f"Request ID: {result.request_id}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    print(f"Domain Outputs: {len(result.domain_outputs)} domains engaged")
    print(f"Mathematical Insights: {len(result.mathematical_insights)} insights generated")
    print()


async def example_recursive_refinement():
    """Example of recursive refinement with SCP and Agent coordination"""
    print("🔄 Example: Recursive Data Refinement")
    print("-" * 40)

    nexus = ARPNexus()
    await nexus.initialize()

    # Create a recursive refinement request
    request = ReasoningRequest(
        request_id="example_recursive_001",
        reasoning_mode=ReasoningMode.RECURSIVE_REFINEMENT,
        input_data={
            "initial_problem": "Design optimal AGI safety measures",
            "current_solution": {
                "safety_layers": ["monitoring", "constraints"],
                "verification_methods": ["formal_proofs", "testing"]
            },
            "improvement_goals": ["robustness", "adaptability", "transparency"]
        },
        recursive_cycles=5,  # Allow up to 5 refinement cycles
        c_value_data={
            "safety_c": complex(0.8, 0.6),      # Safety optimization fractal
            "robustness_c": complex(0.7, 0.7),  # Robustness fractal
            "ethics_c": complex(0.9, 0.4)       # Ethical alignment fractal
        }
    )

    print("Initial C-values:")
    for key, c_val in request.c_value_data.items():
        print(f"  {key}: {c_val}")

    # Process the recursive request
    result = await nexus.process_reasoning_request(request)

    print(f"Request ID: {result.request_id}")
    print(f"Recursive Iterations: {result.recursive_iterations}")
    print(f"Processing Time: {result.processing_time:.2f}s")

    if result.c_value_evolution:
        print("Evolved C-values:")
        for key, c_val in result.c_value_evolution.items():
            print(f"  {key}: {c_val}")

    # Check cycle status
    cycle_key = f"ARP_SCP_{request.request_id.split('_')[-1]}"
    status = nexus.get_cycle_status(cycle_key)
    print(f"Cycle Status: {status}")
    print()


async def example_deep_ontological():
    """Example of deep ontological analysis"""
    print("🌌 Example: Deep Ontological Analysis")
    print("-" * 40)

    nexus = ARPNexus()
    await nexus.initialize()

    # Create deep ontological request
    request = ReasoningRequest(
        request_id="example_ontological_001",
        reasoning_mode=ReasoningMode.DEEP_ONTOLOGICAL,
        input_data={
            "ontological_question": "What is the fundamental nature of intelligence?",
            "paradigms": ["biological", "artificial", "hybrid"],
            "dimensions": ["existence", "goodness", "truth"]
        },
        domain_focus=["OntoPraxis", "GnosiPraxis", "CosmoPraxis"],
        mathematical_foundations=True,
        formal_verification=True
    )

    # Process the request
    result = await nexus.process_reasoning_request(request)

    print(f"Request ID: {result.request_id}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    print(f"Formal Proofs: {len(result.formal_proofs) if result.formal_proofs else 0} generated")
    print()


async def example_cycle_management():
    """Example of cycle limit management and cascade handling"""
    print("⚙️ Example: Cycle Management & Cascade Handling")
    print("-" * 50)

    nexus = ARPNexus()

    # Demonstrate cycle limit changes
    print(f"Default max cycles: {nexus.data_builder.max_cycles_default}")

    # System can adjust limits
    nexus.set_max_cycles(12, "system")
    print(f"System adjusted to: {nexus.data_builder.max_cycles_default}")

    # Create emergency packet that triggers cascade detection
    from arp_nexus import DataExchangePacket
    emergency_packet = nexus.data_builder.create_exchange_packet(
        source_protocol="ARP",
        target_protocol="SCP",
        data_payload={
            "emergency": "critical_system_failure",
            "threat_level": "existential",
            "cascade_detected": True
        }
    )

    print(f"Emergency packet created: {emergency_packet.packet_id}")
    print(f"Cascade imminent: {emergency_packet.cascade_imminent}")
    print(f"Extended cycles allowed: {emergency_packet.max_cycles}")

    # Approve cascade override
    approved = nexus.approve_cascade_override(emergency_packet.packet_id, True)
    print(f"Cascade override approved: {approved}")
    print()


async def main():
    """Run all usage examples"""
    print("🚀 ARP Nexus Usage Examples")
    print("=" * 50)
    print()

    examples = [
        ("Standard Reasoning", example_standard_reasoning),
        ("Recursive Refinement", example_recursive_refinement),
        ("Deep Ontological Analysis", example_deep_ontological),
        ("Cycle Management", example_cycle_management)
    ]

    for example_name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"❌ {example_name} failed: {e}")
            print()

    print("=" * 50)
    print("✅ All examples completed!")
    print()
    print("Key Features Demonstrated:")
    print("• Trinity Logic reasoning with IEL domain orchestration")
    print("• Recursive data refinement with SCP/Agent coordination")
    print("• C-value fractal data evolution and exchange")
    print("• Cycle limit management with cascade detection")
    print("• Formal verification and mathematical foundation integration")


if __name__ == "__main__":
    asyncio.run(main())