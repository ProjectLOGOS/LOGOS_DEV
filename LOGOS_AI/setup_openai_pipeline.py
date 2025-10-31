"""
LOGOS AI - OpenAI Pipeline Setup Script
=======================================

This script sets up the complete OpenAI integration pipeline for LOGOS AI.
It installs dependencies, configures the API, and tests the integration.
"""

import os
import sys
import subprocess
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def install_dependencies():
    """Install required Python packages"""
    
    logger.info("🔧 Installing OpenAI Pipeline dependencies...")
    
    packages = [
        "openai>=1.40.0",
        "pygame>=2.5.0", 
        "numpy>=1.24.0",
        "requests>=2.28.0"
    ]
    
    for package in packages:
        try:
            logger.info(f"   Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--user"
            ])
            logger.info(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"   ❌ Failed to install {package}: {e}")
            return False
    
    logger.info("✅ All dependencies installed successfully")
    return True


def setup_api_key():
    """Setup OpenAI API key configuration"""
    
    logger.info("🔑 Setting up OpenAI API key...")
    
    # Check if API key already exists
    existing_key = os.getenv("OPENAI_API_KEY")
    if existing_key:
        logger.info(f"   ✅ OpenAI API key already configured (ending in ...{existing_key[-4:]})")
        return True
    
    print("\n" + "="*60)
    print("🔑 OpenAI API Key Configuration")
    print("="*60)
    print("To use the OpenAI integrated pipeline, you need an API key from:")
    print("   👉 https://platform.openai.com/api-keys")
    print()
    print("💰 Pricing estimate for LOGOS development:")
    print("   - Whisper (Speech-to-Text): $0.006/minute")
    print("   - GPT-4 (Processing): $0.03-0.06/1K tokens")  
    print("   - TTS (Text-to-Speech): $0.015/1K characters")
    print("   - Expected monthly cost: $50-200 during active development")
    print()
    
    while True:
        choice = input("Do you want to configure your API key now? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            break
        elif choice in ['n', 'no']:
            logger.info("⚠️  Skipping API key configuration - you can set it later with:")
            logger.info("   Windows: $env:OPENAI_API_KEY=\"your-key-here\"")
            logger.info("   Linux/Mac: export OPENAI_API_KEY=\"your-key-here\"")
            return False
        else:
            print("Please enter 'y' or 'n'")
    
    # Get API key from user
    while True:
        api_key = input("\nEnter your OpenAI API key: ").strip()
        if len(api_key) > 20 and api_key.startswith("sk-"):
            break
        else:
            print("❌ Invalid API key format. Please enter a valid key starting with 'sk-'")
    
    # Set environment variable for current session
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Create .env file for future sessions
    env_file = Path(".env")
    try:
        with open(env_file, "a") as f:
            f.write(f"\n# OpenAI API Configuration\nOPENAI_API_KEY={api_key}\n")
        logger.info(f"✅ API key saved to {env_file}")
    except Exception as e:
        logger.warning(f"⚠️  Could not save to .env file: {e}")
    
    logger.info("✅ OpenAI API key configured successfully")
    return True


async def test_openai_integration():
    """Test OpenAI integration"""
    
    logger.info("🧪 Testing OpenAI integration...")
    
    try:
        # Import and test the pipeline
        from User_Interaction_Protocol.interfaces.external.openai_integrated_pipeline import (
            initialize_pipeline, get_pipeline, OpenAIConfig, quick_text_interaction
        )
        
        # Initialize pipeline
        logger.info("   Initializing pipeline...")
        config = OpenAIConfig()
        
        if not initialize_pipeline(config):
            logger.error("   ❌ Pipeline initialization failed")
            return False
        
        logger.info("   ✅ Pipeline initialized successfully")
        
        # Test text interaction
        logger.info("   Testing text interaction...")
        test_response = await quick_text_interaction("Hello, what are your capabilities?")
        
        if "error" not in test_response.lower():
            logger.info("   ✅ Text interaction test passed")
            logger.info(f"   📝 Response preview: {test_response[:100]}...")
        else:
            logger.error(f"   ❌ Text interaction failed: {test_response}")
            return False
        
        # Check pipeline status
        pipeline = get_pipeline()
        if pipeline:
            stats = pipeline.get_usage_stats()
            logger.info("   📊 Pipeline statistics:")
            logger.info(f"      Requests processed: {stats['requests_processed']}")
            logger.info(f"      Total cost: ${stats['total_cost']:.4f}")
        
        logger.info("✅ OpenAI integration test completed successfully")
        return True
        
    except ImportError as e:
        logger.error(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"   ❌ Integration test failed: {e}")
        return False


def test_logos_system():
    """Test LOGOS AI system with OpenAI pipeline"""
    
    logger.info("🚀 Testing LOGOS AI system startup...")
    
    try:
        # Test if LOGOS_AI.py can import successfully
        import sys
        sys.path.append(".")
        
        # Try importing the main controller
        from LOGOS_AI import LOGOSMasterController, LOGOSSystemConfiguration
        
        # Create test configuration
        config = LOGOSSystemConfiguration()
        config.enable_openai_pipeline = True
        config.development_mode = True
        
        logger.info("   ✅ LOGOS AI imports successful")
        logger.info("   ✅ OpenAI pipeline integration ready")
        
        return True
        
    except ImportError as e:
        logger.error(f"   ❌ LOGOS AI import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"   ❌ LOGOS AI test failed: {e}")
        return False


def main():
    """Main setup process"""
    
    print("=" * 60)
    print("🚀 LOGOS AI - OpenAI Pipeline Setup")
    print("=" * 60)
    print("This script will configure the complete OpenAI integration pipeline")
    print("including Whisper, GPT-4, and TTS capabilities for LOGOS AI.")
    print()
    
    success_steps = []
    
    # Step 1: Install dependencies
    if install_dependencies():
        success_steps.append("Dependencies")
    else:
        logger.error("❌ Dependency installation failed - setup cannot continue")
        return False
    
    # Step 2: Setup API key
    if setup_api_key():
        success_steps.append("API Key")
    else:
        logger.warning("⚠️  API key not configured - some features will be limited")
    
    # Step 3: Test integration (only if API key configured)
    if os.getenv("OPENAI_API_KEY"):
        if asyncio.run(test_openai_integration()):
            success_steps.append("Integration Test")
    else:
        logger.info("⏭️  Skipping integration test (no API key)")
    
    # Step 4: Test LOGOS system
    if test_logos_system():
        success_steps.append("LOGOS System")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SETUP SUMMARY")
    print("=" * 60)
    
    if len(success_steps) >= 2:  # Dependencies + at least one other step
        print("🎉 SETUP SUCCESSFUL!")
        print()
        print("✅ Completed steps:")
        for step in success_steps:
            print(f"   • {step}")
        
        print()
        print("🚀 Ready to start LOGOS AI with OpenAI pipeline:")
        print("   python LOGOS_AI.py")
        print()
        print("🎤 Voice interaction capabilities:")
        print("   • Whisper speech-to-text")
        print("   • GPT-4 intelligent processing")  
        print("   • High-quality text-to-speech")
        print("   • LOGOS tool integration")
        
        return True
    else:
        print("⚠️  SETUP INCOMPLETE")
        print("Some steps failed - check the log messages above.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Setup failed with error: {e}")
        sys.exit(1)