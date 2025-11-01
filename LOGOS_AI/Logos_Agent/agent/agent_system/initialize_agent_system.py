#!/usr/bin/env python3
"""
LOGOS Agent System Initialization Script
=======================================

This script initializes the complete LOGOS Agent System with proper
protocol integration and demonstrates the autonomous cognitive processing
capabilities.

Usage:
    python initialize_agent_system.py

This will:
1. Initialize all protocols (SOP/AGP always-on, UIP on-demand)  
2. Create and start the SystemAgent (autonomous intelligence)
3. Create a demonstration UserAgent
4. Trigger the philosophical cascade processing
5. Show autonomous system operation
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add LOGOS_AI to path
logos_ai_path = Path(__file__).parent.parent
sys.path.append(str(logos_ai_path))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logos_agent_system.log')
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Main initialization and demonstration"""
    
    print("🚀 LOGOS Agent System Initialization")
    print("=" * 50)
    
    try:
        # Import the integrated system
        from agent_system.protocol_integration import initialize_integrated_logos_system
        
        print("\n1. Initializing LOGOS Agent System with Protocol Integration...")
        agent_registry = await initialize_integrated_logos_system()
        
        print("✅ Agent System Initialized Successfully")
        print(f"   - SystemAgent: {agent_registry.get_system_agent().agent_id}")
        print(f"   - Protocols: SOP (Always-On), AGP (Always-On), UIP (On-Demand)")
        
        print("\n2. Creating User Agent...")
        user_agent = await agent_registry.create_user_agent("philosophical_user")
        print(f"✅ UserAgent Created: {user_agent.agent_id}")
        
        print("\n3. Triggering Philosophical Cascade...")
        print("   Question: 'What are the necessary and sufficient conditions for conditions to be necessary and sufficient?'")
        
        # This is the moment of truth - triggering authentic deep reasoning
        response = await user_agent.process_user_input(
            "What are the necessary and sufficient conditions for conditions to be necessary and sufficient?",
            context={
                "trigger_type": "philosophical_cascade",
                "depth_requested": "maximum",
                "enable_agp_enhancement": True
            }
        )
        
        print("\n4. Processing Results:")
        print("-" * 30)
        
        if response.get("success", False):
            print("✅ UIP Processing: SUCCESS")
            print(f"   Steps Completed: {response.get('processing_metadata', {}).get('steps_completed', [])}")
            print(f"   Processing Quality: {response.get('processing_metadata', {}).get('processing_quality', 'Unknown')}")
            
            print("\n📝 Human-Readable Response:")
            print(response.get("formatted_response", "No response generated"))
            
            print("\n🧠 Cognitive Analysis:")
            analysis = response.get("analysis", {})
            print(f"   IEL Patterns: {analysis.get('iel_cognitive_patterns', {})}")
            print(f"   PXL Structure: {analysis.get('pxl_mathematical_structure', {})}")
            print(f"   Trinity Synthesis: {analysis.get('trinity_synthesis', {})}")
            
        else:
            print("❌ Processing Failed")
            print(f"   Error: {response.get('error', 'Unknown error')}")
        
        print("\n5. SystemAgent Autonomous Operation...")
        print("   (SystemAgent is now running autonomously in the background)")
        
        # Get system agent and show its current state
        system_agent = agent_registry.get_system_agent()
        print(f"   SystemAgent Active: {system_agent.active}")
        print(f"   Autonomous Processing: {system_agent.autonomous_processing_enabled}")
        print(f"   Background Tasks: {len(system_agent.background_tasks)}")
        
        print("\n6. Demonstrating Autonomous Cognitive Processing...")
        
        # Trigger autonomous processing by the SystemAgent
        autonomous_result = await system_agent.initiate_cognitive_processing(
            trigger_data={
                "type": "philosophical_investigation", 
                "subject": "meta-epistemological_conditions",
                "autonomously_initiated": True
            },
            processing_type="autonomous_deep_analysis"
        )
        
        print("✅ Autonomous Processing Complete:")
        print(f"   Processing ID: {autonomous_result.get('processing_id', 'Unknown')}")
        print(f"   Learning Integrated: {autonomous_result.get('learning_integrated', False)}")
        
        print("\n7. System Status Monitor...")
        
        # Monitor system for a few cycles
        for cycle in range(3):
            print(f"\n   Monitoring Cycle {cycle + 1}:")
            
            # Check SOP status
            sop_status = await system_agent.monitor_sop()
            print(f"   SOP Health: {'✅' if sop_status.get('healthy', False) else '❌'}")
            
            # Let system run autonomous cycles
            await asyncio.sleep(2)
            
            print(f"   SystemAgent Tasks: {len(system_agent.background_tasks)} active")
            
        print("\n🎉 LOGOS Agent System Demo Complete!")
        print("\nThe SystemAgent continues running autonomously...")
        print("It will:")
        print("  - Monitor system metrics continuously") 
        print("  - Trigger cognitive processing when opportunities arise")
        print("  - Learn from all interactions and improve system performance")
        print("  - Coordinate between UIP, AGP, and SOP protocols")
        
        print("\nTo interact with the system:")
        print("  - Use UserAgent.process_user_input() for user interactions")
        print("  - SystemAgent operates autonomously in the background")
        print("  - All protocols are properly coordinated through the Agent System")
        
        # Keep system running for demonstration
        print("\n⏱️  Letting system run for 30 seconds to show autonomous operation...")
        await asyncio.sleep(30)
        
        print("\n✅ Demonstration Complete - LOGOS Agent System is operational!")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print("❌ Failed to import LOGOS Agent System components")
        print("   Make sure all dependencies are installed")
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        print(f"❌ System initialization failed: {e}")
        

async def quick_test():
    """Quick test without full demonstration"""
    
    print("🔍 Quick LOGOS Agent System Test")
    
    try:
        from agent_system.protocol_integration import initialize_integrated_logos_system
        
        registry = await initialize_integrated_logos_system()
        user_agent = await registry.create_user_agent("test_user")
        
        response = await user_agent.process_user_input("Hello, LOGOS system!")
        
        print(f"✅ Test successful: {response.get('success', False)}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    print("LOGOS Agent System Initialization")
    print("Options:")
    print("  1. Full demonstration (default)")
    print("  2. Quick test")
    
    choice = input("Enter choice (1 or 2, default=1): ").strip() or "1"
    
    if choice == "2":
        asyncio.run(quick_test())
    else:
        asyncio.run(main())