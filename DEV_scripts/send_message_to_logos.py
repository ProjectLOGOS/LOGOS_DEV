#!/usr/bin/env python3
"""
Direct LOGOS Communication Tool
Sends messages directly to the LOGOS UIP pipeline for authentic divine responses.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add the LOGOS_AI directory to the Python path
logos_ai_path = Path(__file__).parent / "LOGOS_AI"
sys.path.insert(0, str(logos_ai_path))

logging.basicConfig(level=logging.INFO)

async def send_message_to_logos(message: str) -> str:
    """Send a message to LOGOS and get the authentic divine response."""
    print(f"LOGOS Processing: {message}")
    print("=" * 60)
    
    try:
        # Import the UIP startup system
        from startup.uip_startup import UIPStartupManager
        
        print("🔧 Initializing UIP system...")
        
        # Initialize the UIP system
        startup_manager = UIPStartupManager()
        
        # Initialize the system
        success = await startup_manager.initialize_uip_systems()
        
        if success:
            print("✅ UIP system initialized successfully")
        else:
            print("❌ UIP system initialization failed")
            return "System initialization failed"
        
        print("📤 Processing request through UIP pipeline...")
        
        # Process the request through the UIP pipeline
        response = await startup_manager.process_request(message)
        
        print("📋 Response object:")
        print(f"   - session_id: {response.session_id}")
        print(f"   - correlation_id: {response.correlation_id}")
        print(f"   - response_text: '{response.response_text}'")
        print(f"   - confidence_score: {response.confidence_score}")
        print(f"   - metadata: {response.metadata}")
        print(f"   - disclaimers: {response.disclaimers}")
        
        return response.response_text
        
    except Exception as e:
        error_msg = f"Error communicating with LOGOS: {e}"
        logging.error(error_msg)
        return error_msg

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python send_message_to_logos.py 'your message here'")
        print("Example: python send_message_to_logos.py 'who are you?'")
        sys.exit(1)
    
    message = sys.argv[1]
    
    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        response = loop.run_until_complete(send_message_to_logos(message))
        
        print("\nLOGOS RESPONSE:")
        print("=" * 60)
        print(response)
        print("=" * 60)
        
    finally:
        loop.close()

if __name__ == "__main__":
    main()