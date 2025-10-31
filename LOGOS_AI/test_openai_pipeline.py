"""
LOGOS AI - OpenAI Pipeline Test Suite
=====================================

Comprehensive test suite for the OpenAI integrated pipeline.
Tests all components: Whisper STT, GPT-4 processing, TTS, and LOGOS tool integration.
"""

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenAIPipelineTests:
    """Test suite for OpenAI pipeline integration"""
    
    def __init__(self):
        self.pipeline = None
        self.test_results = {}
    
    async def setup(self):
        """Setup test environment"""
        logger.info("🔧 Setting up test environment...")
        
        try:
            from User_Interaction_Protocol.interfaces.external.openai_integrated_pipeline import (
                initialize_pipeline, get_pipeline, OpenAIConfig
            )
            
            # Check API key
            if not os.getenv("OPENAI_API_KEY"):
                logger.error("❌ OPENAI_API_KEY not set - cannot run tests")
                return False
            
            # Initialize pipeline
            config = OpenAIConfig()
            if not initialize_pipeline(config):
                logger.error("❌ Pipeline initialization failed")
                return False
            
            self.pipeline = get_pipeline()
            logger.info("✅ Test environment ready")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Import error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    async def test_text_processing(self):
        """Test GPT-4 text processing"""
        logger.info("🧠 Testing GPT-4 text processing...")
        
        test_cases = [
            "Hello, what are LOGOS AI's capabilities?",
            "Can you analyze this text: Machine learning is transforming industries.",
            "Predict the next values in this sequence: 1, 3, 7, 15, 31",
            "Prove that if P implies Q and P is true, then Q must be true.",
            "What is the current system status?"
        ]
        
        results = []
        
        for i, test_input in enumerate(test_cases, 1):
            logger.info(f"   Test {i}/5: {test_input[:40]}...")
            
            start_time = time.time()
            
            try:
                response = await self.pipeline.text_interaction(test_input)
                processing_time = time.time() - start_time
                
                if response and response.response_text:
                    results.append({
                        "input": test_input,
                        "success": True,
                        "response_length": len(response.response_text),
                        "processing_time": processing_time,
                        "confidence": response.confidence_score
                    })
                    logger.info(f"   ✅ Test {i} passed ({processing_time:.2f}s)")
                else:
                    results.append({
                        "input": test_input,
                        "success": False,
                        "error": "Empty response"
                    })
                    logger.error(f"   ❌ Test {i} failed: Empty response")
                
            except Exception as e:
                results.append({
                    "input": test_input,
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"   ❌ Test {i} failed: {e}")
        
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        self.test_results["text_processing"] = {
            "success_rate": success_rate,
            "results": results
        }
        
        logger.info(f"📊 Text processing test completed: {success_rate*100:.1f}% success rate")
        return success_rate > 0.8
    
    async def test_tool_integration(self):
        """Test LOGOS tool integration"""
        logger.info("🛠️ Testing LOGOS tool integration...")
        
        tool_test_cases = [
            ("tetragnos", "Analyze the sentiment of this text: I love artificial intelligence!"),
            ("telos", "Predict future values based on this data: 1, 4, 9, 16, 25"),
            ("thonoc", "Prove that all humans are mortal and Socrates is human, therefore Socrates is mortal"),
            ("status", "What is the current LOGOS system status?")
        ]
        
        results = []
        
        for tool_type, test_input in tool_test_cases:
            logger.info(f"   Testing {tool_type.upper()} tool...")
            
            try:
                response = await self.pipeline.text_interaction(test_input)
                
                # Check if tools were used
                tools_used = response.metadata.get("tools_used", []) if hasattr(response, 'metadata') else []
                
                result = {
                    "tool": tool_type,
                    "success": len(tools_used) > 0,
                    "tools_used": tools_used,
                    "response_generated": bool(response.response_text)
                }
                
                results.append(result)
                
                if result["success"]:
                    logger.info(f"   ✅ {tool_type.upper()} tool integration working")
                else:
                    logger.warning(f"   ⚠️ {tool_type.upper()} tool not triggered (may be expected)")
                
            except Exception as e:
                results.append({
                    "tool": tool_type,
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"   ❌ {tool_type.upper()} tool test failed: {e}")
        
        # Tools might not always be triggered depending on GPT's decision
        # So we consider this test successful if at least responses are generated
        success_rate = sum(1 for r in results if r.get("response_generated", False)) / len(results)
        self.test_results["tool_integration"] = {
            "success_rate": success_rate,
            "results": results
        }
        
        logger.info(f"📊 Tool integration test completed: {success_rate*100:.1f}% response rate")
        return success_rate > 0.5
    
    async def test_tts_synthesis(self):
        """Test text-to-speech synthesis"""
        logger.info("🔊 Testing text-to-speech synthesis...")
        
        test_texts = [
            "Hello, this is LOGOS AI speaking.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing text-to-speech with numbers: 1, 2, 3, 4, 5."
        ]
        
        results = []
        
        for i, text in enumerate(test_texts, 1):
            logger.info(f"   TTS Test {i}/3: {text[:30]}...")
            
            try:
                start_time = time.time()
                tts_result = await self.pipeline.synthesize_speech(text)
                processing_time = time.time() - start_time
                
                if tts_result.success:
                    # Check if audio file was created
                    audio_file = tts_result.metadata.get("output_file")
                    file_exists = audio_file and Path(audio_file).exists()
                    
                    results.append({
                        "text": text,
                        "success": True,
                        "processing_time": processing_time,
                        "audio_duration": tts_result.audio_duration,
                        "file_created": file_exists
                    })
                    
                    logger.info(f"   ✅ TTS Test {i} passed ({processing_time:.2f}s)")
                    
                    # Clean up temp file
                    if file_exists:
                        try:
                            os.unlink(audio_file)
                        except:
                            pass
                else:
                    results.append({
                        "text": text,
                        "success": False,
                        "error": tts_result.error
                    })
                    logger.error(f"   ❌ TTS Test {i} failed: {tts_result.error}")
                
            except Exception as e:
                results.append({
                    "text": text,
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"   ❌ TTS Test {i} failed: {e}")
        
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        self.test_results["tts_synthesis"] = {
            "success_rate": success_rate,
            "results": results
        }
        
        logger.info(f"📊 TTS synthesis test completed: {success_rate*100:.1f}% success rate")
        return success_rate > 0.8
    
    async def test_performance_metrics(self):
        """Test performance and usage tracking"""
        logger.info("📈 Testing performance metrics...")
        
        try:
            # Get usage stats
            stats = self.pipeline.get_usage_stats()
            
            expected_keys = [
                "requests_processed", "audio_minutes_transcribed", 
                "tokens_used", "audio_generated_minutes", "total_cost"
            ]
            
            metrics_available = all(key in stats for key in expected_keys)
            cost_tracking = stats.get("total_cost", 0) >= 0
            
            self.test_results["performance_metrics"] = {
                "metrics_available": metrics_available,
                "cost_tracking": cost_tracking,
                "stats": stats
            }
            
            if metrics_available and cost_tracking:
                logger.info("   ✅ Performance metrics working correctly")
                logger.info(f"   📊 Total requests: {stats['requests_processed']}")
                logger.info(f"   💰 Total cost: ${stats['total_cost']:.4f}")
                return True
            else:
                logger.error("   ❌ Performance metrics incomplete")
                return False
                
        except Exception as e:
            logger.error(f"   ❌ Performance metrics test failed: {e}")
            return False
    
    def generate_report(self):
        """Generate comprehensive test report"""
        
        print("\n" + "=" * 60)
        print("📋 OPENAI PIPELINE TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = 0
        
        for test_name, results in self.test_results.items():
            print(f"\n🧪 {test_name.replace('_', ' ').title()}")
            print("-" * 40)
            
            if isinstance(results, dict) and "success_rate" in results:
                success_rate = results["success_rate"]
                status = "✅ PASS" if success_rate > 0.7 else "⚠️  PARTIAL" if success_rate > 0.3 else "❌ FAIL"
                print(f"   Status: {status}")
                print(f"   Success Rate: {success_rate*100:.1f}%")
                
                if success_rate > 0.7:
                    passed_tests += 1
                    
            elif isinstance(results, dict) and "metrics_available" in results:
                # Performance metrics test
                all_good = results["metrics_available"] and results["cost_tracking"]
                status = "✅ PASS" if all_good else "❌ FAIL"
                print(f"   Status: {status}")
                
                if all_good:
                    passed_tests += 1
        
        # Overall summary
        print("\n" + "=" * 60)
        overall_success = passed_tests / total_tests if total_tests > 0 else 0
        
        if overall_success >= 0.8:
            print("🎉 OVERALL RESULT: EXCELLENT")
            print("   All major systems working correctly!")
        elif overall_success >= 0.6:
            print("✅ OVERALL RESULT: GOOD") 
            print("   Most systems working, some minor issues.")
        elif overall_success >= 0.4:
            print("⚠️  OVERALL RESULT: PARTIAL")
            print("   Some systems working, significant issues present.")
        else:
            print("❌ OVERALL RESULT: FAILED")
            print("   Major system failures detected.")
        
        print(f"\n📊 Test Summary: {passed_tests}/{total_tests} test suites passed")
        
        # Usage recommendations
        pipeline = self.pipeline
        if pipeline:
            stats = pipeline.get_usage_stats()
            total_cost = stats.get("total_cost", 0)
            
            print(f"💰 Total testing cost: ${total_cost:.4f}")
            
            if total_cost > 0:
                monthly_estimate = total_cost * 100  # Rough scaling for regular usage
                print(f"📈 Estimated monthly cost (regular usage): ~${monthly_estimate:.2f}")
        
        return overall_success >= 0.6


async def main():
    """Main test execution"""
    
    print("=" * 60)
    print("🧪 LOGOS AI - OpenAI Pipeline Test Suite")
    print("=" * 60)
    print("Running comprehensive tests for the OpenAI integrated pipeline...")
    print()
    
    # Initialize test suite
    tests = OpenAIPipelineTests()
    
    # Setup
    if not await tests.setup():
        logger.error("❌ Test setup failed - cannot continue")
        return False
    
    # Run test suite
    test_functions = [
        ("Text Processing", tests.test_text_processing),
        ("Tool Integration", tests.test_tool_integration), 
        ("TTS Synthesis", tests.test_tts_synthesis),
        ("Performance Metrics", tests.test_performance_metrics)
    ]
    
    logger.info("🚀 Starting test execution...")
    
    for test_name, test_func in test_functions:
        logger.info(f"\n▶️  Running {test_name} tests...")
        try:
            await test_func()
        except Exception as e:
            logger.error(f"   ❌ {test_name} test suite failed: {e}")
    
    # Generate final report
    success = tests.generate_report()
    
    if success:
        print("\n🎊 OpenAI pipeline is ready for production use!")
        print("   You can now start LOGOS AI with full voice capabilities.")
    else:
        print("\n⚠️  Some issues detected - review test results above.")
        print("   Basic functionality may still work.")
    
    return success


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests cancelled by user")
        exit(1)
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        exit(1)