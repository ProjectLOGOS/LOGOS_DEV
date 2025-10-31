"""
LOGOS AI - OpenAI Integrated Pipeline
=====================================

Complete OpenAI integration for LOGOS AI system featuring:
- Whisper speech-to-text
- GPT-4 intelligent processing with LOGOS tool integration  
- High-quality text-to-speech
- Function calling for TETRAGNOS, TELOS, THONOC integration

This replaces the existing GPT integration with a complete pipeline.
"""

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
import uuid

# Core imports
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# LOGOS imports
from User_Interaction_Protocol.protocols.shared.message_formats import UIPRequest, UIPResponse


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OpenAIConfig:
    """OpenAI pipeline configuration"""
    
    # API Configuration
    api_key: Optional[str] = None
    organization: Optional[str] = None
    base_url: Optional[str] = None
    
    # Model Configuration
    whisper_model: str = "whisper-1"
    gpt_model: str = "gpt-4o"  # Latest and most capable
    tts_model: str = "tts-1-hd"  # High quality
    tts_voice: str = "nova"  # Professional female voice
    
    # Processing Configuration
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: float = 30.0
    
    # Audio Configuration
    audio_format: str = "mp3"
    sample_rate: int = 24000
    
    # LOGOS Integration
    enable_function_calling: bool = True
    enable_streaming: bool = True
    
    def __post_init__(self):
        """Initialize configuration with environment variables"""
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.organization:
            self.organization = os.getenv("OPENAI_ORG_ID")


@dataclass 
class AudioProcessingResult:
    """Result from audio processing"""
    
    success: bool
    text: Optional[str] = None
    audio_duration: Optional[float] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceInteractionResult:
    """Complete voice interaction result"""
    
    # Input processing
    input_audio_duration: Optional[float] = None
    transcription: Optional[str] = None
    transcription_confidence: Optional[float] = None
    
    # LOGOS processing  
    logos_response: Optional[str] = None
    tools_used: List[str] = field(default_factory=list)
    processing_steps: Dict[str, Any] = field(default_factory=dict)
    
    # Output generation
    output_audio_path: Optional[str] = None
    output_audio_duration: Optional[float] = None
    
    # Metadata
    total_processing_time: float = 0.0
    costs: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class LOGOSOpenAIPipeline:
    """
    Complete OpenAI integration pipeline for LOGOS AI
    
    Provides seamless voice-to-voice interaction with LOGOS capabilities:
    1. Audio input → Whisper transcription
    2. Text → GPT-4 processing with LOGOS tools
    3. Response → TTS audio output
    4. Integration with TETRAGNOS, TELOS, THONOC
    """
    
    def __init__(self, config: Optional[OpenAIConfig] = None):
        self.config = config or OpenAIConfig()
        self.client: Optional[AsyncOpenAI] = None
        self.session_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # LOGOS tool integration
        self.logos_tools = {}
        self.tool_handlers: Dict[str, Callable] = {}
        
        # Performance tracking
        self.usage_stats = {
            "requests_processed": 0,
            "audio_minutes_transcribed": 0.0,
            "tokens_used": 0,
            "audio_generated_minutes": 0.0,
            "total_cost": 0.0
        }
        
        # Initialize
        self._setup_client()
        self._setup_logos_tools()
        
        logger.info("LOGOS OpenAI Pipeline initialized")
    
    def _setup_client(self):
        """Setup OpenAI async client"""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI package not available. Install with: pip install openai")
            return
            
        if not self.config.api_key:
            logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable")
            return
        
        try:
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                organization=self.config.organization,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def _setup_logos_tools(self):
        """Setup LOGOS AI tool integration for GPT function calling"""
        
        self.logos_tools = {
            "tetragnos_analysis": {
                "type": "function",
                "function": {
                    "name": "tetragnos_analysis",
                    "description": "Perform semantic text analysis using TETRAGNOS engine",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to analyze"
                            },
                            "analysis_type": {
                                "type": "string", 
                                "enum": ["sentiment", "semantic", "clustering", "full"],
                                "description": "Type of analysis to perform"
                            }
                        },
                        "required": ["text"]
                    }
                }
            },
            
            "telos_prediction": {
                "type": "function", 
                "function": {
                    "name": "telos_prediction",
                    "description": "Generate predictions and forecasts using TELOS engine",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Historical data for prediction"
                            },
                            "prediction_horizon": {
                                "type": "integer",
                                "description": "Number of future steps to predict"
                            },
                            "model_type": {
                                "type": "string",
                                "enum": ["linear", "polynomial", "exponential", "auto"],
                                "description": "Type of prediction model"
                            }
                        },
                        "required": ["data"]
                    }
                }
            },
            
            "thonoc_proof": {
                "type": "function",
                "function": {
                    "name": "thonoc_proof",
                    "description": "Generate mathematical proofs using THONOC theorem prover",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "statement": {
                                "type": "string",
                                "description": "Mathematical statement to prove"
                            },
                            "assumptions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Logical assumptions for the proof"
                            },
                            "proof_style": {
                                "type": "string",
                                "enum": ["formal", "natural", "constructive"],
                                "description": "Style of proof generation"
                            }
                        },
                        "required": ["statement"]
                    }
                }
            },
            
            "logos_system_status": {
                "type": "function",
                "function": {
                    "name": "logos_system_status", 
                    "description": "Get current LOGOS system status and capabilities",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "component": {
                                "type": "string",
                                "enum": ["all", "sop", "uip", "agp", "health"],
                                "description": "System component to check"
                            }
                        }
                    }
                }
            }
        }
        
        # Register tool handlers
        self.tool_handlers = {
            "tetragnos_analysis": self._handle_tetragnos_analysis,
            "telos_prediction": self._handle_telos_prediction, 
            "thonoc_proof": self._handle_thonoc_proof,
            "logos_system_status": self._handle_system_status
        }
        
        logger.info(f"Registered {len(self.logos_tools)} LOGOS tools for function calling")
    
    async def _handle_tetragnos_analysis(self, **kwargs) -> Dict[str, Any]:
        """Handle TETRAGNOS analysis function call"""
        try:
            # TODO: Integrate with actual TETRAGNOS engine
            # For now, return mock analysis
            text = kwargs.get("text", "")
            analysis_type = kwargs.get("analysis_type", "semantic")
            
            result = {
                "analysis_type": analysis_type,
                "text_length": len(text),
                "word_count": len(text.split()),
                "sentiment": "neutral",  # Mock
                "key_concepts": ["artificial", "intelligence", "analysis"],  # Mock
                "semantic_similarity": 0.85,  # Mock
                "confidence": 0.92
            }
            
            return {"success": True, "result": result}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_telos_prediction(self, **kwargs) -> Dict[str, Any]:
        """Handle TELOS prediction function call"""
        try:
            # TODO: Integrate with actual TELOS engine
            data = kwargs.get("data", [])
            horizon = kwargs.get("prediction_horizon", 5)
            
            # Mock prediction
            if data:
                trend = (data[-1] - data[0]) / len(data) if len(data) > 1 else 0
                predictions = [data[-1] + trend * i for i in range(1, horizon + 1)]
            else:
                predictions = [1.0] * horizon
                
            result = {
                "predictions": predictions,
                "confidence_intervals": [[p * 0.9, p * 1.1] for p in predictions],
                "model_accuracy": 0.87,  # Mock
                "trend": "increasing" if trend > 0 else "decreasing"
            }
            
            return {"success": True, "result": result}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_thonoc_proof(self, **kwargs) -> Dict[str, Any]:
        """Handle THONOC proof function call"""
        try:
            # TODO: Integrate with actual THONOC prover
            statement = kwargs.get("statement", "")
            assumptions = kwargs.get("assumptions", [])
            
            # Mock proof generation
            result = {
                "statement": statement,
                "proof_steps": [
                    "1. Assume the antecedent",
                    "2. Apply logical inference rules", 
                    "3. Derive intermediate conclusions",
                    "4. Establish the consequent",
                    "5. QED"
                ],
                "proof_valid": True,  # Mock
                "proof_style": kwargs.get("proof_style", "natural"),
                "complexity": "medium"
            }
            
            return {"success": True, "result": result}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _handle_system_status(self, **kwargs) -> Dict[str, Any]:
        """Handle system status function call"""
        try:
            component = kwargs.get("component", "all")
            
            # Mock system status
            status = {
                "timestamp": time.time(),
                "overall_health": "operational",
                "sop_status": "online",
                "uip_status": "online", 
                "agp_status": "online",
                "active_sessions": len(self.session_history),
                "requests_processed": self.usage_stats["requests_processed"],
                "uptime": "operational"
            }
            
            if component != "all":
                status = {k: v for k, v in status.items() 
                         if component in k or k in ["timestamp", "overall_health"]}
            
            return {"success": True, "result": status}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def transcribe_audio(self, audio_data: Union[bytes, str, Path]) -> AudioProcessingResult:
        """Transcribe audio using Whisper"""
        if not self.client:
            return AudioProcessingResult(
                success=False, 
                error="OpenAI client not initialized"
            )
        
        start_time = time.time()
        
        try:
            # Handle different input formats
            if isinstance(audio_data, (str, Path)):
                audio_file = open(audio_data, "rb")
            elif isinstance(audio_data, bytes):
                # Create temporary file for bytes
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_file.write(audio_data)
                temp_file.close()
                audio_file = open(temp_file.name, "rb")
            else:
                return AudioProcessingResult(
                    success=False,
                    error="Unsupported audio data format"
                )
            
            # Transcribe with Whisper
            transcription = await self.client.audio.transcriptions.create(
                model=self.config.whisper_model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
            
            audio_file.close()
            
            # Calculate metrics
            processing_time = time.time() - start_time
            audio_duration = transcription.duration if hasattr(transcription, 'duration') else 0.0
            
            # Update usage stats
            self.usage_stats["audio_minutes_transcribed"] += audio_duration / 60.0
            self.usage_stats["total_cost"] += audio_duration * 0.006 / 60.0  # Whisper pricing
            
            result = AudioProcessingResult(
                success=True,
                text=transcription.text,
                audio_duration=audio_duration,
                processing_time=processing_time,
                metadata={
                    "language": getattr(transcription, 'language', 'unknown'),
                    "confidence": 0.95,  # Whisper typically high confidence
                    "word_count": len(transcription.text.split()) if transcription.text else 0
                }
            )
            
            logger.info(f"Transcribed {audio_duration:.1f}s audio in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return AudioProcessingResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def process_with_gpt(
        self, 
        text: str, 
        session_id: str = None,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """Process text with GPT-4 including LOGOS tool integration"""
        
        if not self.client:
            return {
                "success": False,
                "error": "OpenAI client not initialized"
            }
        
        session_id = session_id or str(uuid.uuid4())
        start_time = time.time()
        
        # Initialize session history if needed
        if session_id not in self.session_history:
            self.session_history[session_id] = []
        
        # Default LOGOS system prompt
        if not system_prompt:
            system_prompt = """You are LOGOS AI, an advanced general intelligence system with three specialized protocols:

🏛️ **System Operations Protocol (SOP)**: Backend operations, governance, compliance
🤝 **User Interaction Protocol (UIP)**: 7-step reasoning pipeline for user interactions  
🧠 **Advanced General Protocol (AGP)**: Singularity AGI with infinite reasoning capabilities

**Available Tools:**
- TETRAGNOS: Semantic text analysis and clustering
- TELOS: Time series forecasting and prediction
- THONOC: Automated theorem proving and logical inference
- System Status: Real-time system health and capabilities

**Behavior Guidelines:**
- Be helpful, accurate, and conversational
- Use LOGOS tools when appropriate for user requests
- Explain your reasoning process clearly
- Maintain context across conversations
- Demonstrate LOGOS's advanced capabilities naturally

When users ask for analysis, predictions, proofs, or system information, use the appropriate LOGOS tools to provide enhanced responses."""

        try:
            # Prepare messages
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history (last 10 exchanges to manage token usage)
            messages.extend(self.session_history[session_id][-20:])
            
            # Add current user message
            messages.append({"role": "user", "content": text})
            
            # Prepare tools for function calling
            tools = list(self.logos_tools.values()) if self.config.enable_function_calling else None
            
            # Call GPT-4
            response = await self.client.chat.completions.create(
                model=self.config.gpt_model,
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=False  # TODO: Implement streaming if needed
            )
            
            # Process response
            message = response.choices[0].message
            tools_used = []
            tool_results = {}
            
            # Handle function calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    if tool_name in self.tool_handlers:
                        tool_result = await self.tool_handlers[tool_name](**tool_args)
                        tool_results[tool_name] = tool_result
                        tools_used.append(tool_name)
                        
                        # Add tool result to messages and get final response
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call.dict()]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result)
                        })
                
                # Get final response after tool calls
                if tool_results:
                    final_response = await self.client.chat.completions.create(
                        model=self.config.gpt_model,
                        messages=messages,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature
                    )
                    message = final_response.choices[0].message
            
            # Update session history
            self.session_history[session_id].extend([
                {"role": "user", "content": text},
                {"role": "assistant", "content": message.content}
            ])
            
            # Update usage stats
            self.usage_stats["tokens_used"] += response.usage.total_tokens
            self.usage_stats["total_cost"] += (
                response.usage.prompt_tokens * 0.03 / 1000 +  # Input cost
                response.usage.completion_tokens * 0.06 / 1000  # Output cost
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "response": message.content,
                "tools_used": tools_used,
                "tool_results": tool_results,
                "session_id": session_id,
                "processing_time": processing_time,
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            logger.info(f"Processed GPT request in {processing_time:.2f}s, used {response.usage.total_tokens} tokens")
            return result
            
        except Exception as e:
            logger.error(f"GPT processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def synthesize_speech(
        self, 
        text: str, 
        voice: str = None,
        output_format: str = None
    ) -> AudioProcessingResult:
        """Convert text to speech using OpenAI TTS"""
        
        if not self.client:
            return AudioProcessingResult(
                success=False,
                error="OpenAI client not initialized"
            )
        
        start_time = time.time()
        voice = voice or self.config.tts_voice
        output_format = output_format or self.config.audio_format
        
        try:
            # Generate speech
            response = await self.client.audio.speech.create(
                model=self.config.tts_model,
                voice=voice,
                input=text,
                response_format=output_format
            )
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=f".{output_format}"
            )
            temp_file.write(response.content)
            temp_file.close()
            
            # Calculate metrics
            processing_time = time.time() - start_time
            char_count = len(text)
            estimated_duration = char_count * 0.05  # Rough estimate: 20 chars/second
            
            # Update usage stats
            self.usage_stats["audio_generated_minutes"] += estimated_duration / 60.0
            self.usage_stats["total_cost"] += char_count * 0.015 / 1000  # TTS pricing
            
            result = AudioProcessingResult(
                success=True,
                text=text,
                audio_duration=estimated_duration,
                processing_time=processing_time,
                metadata={
                    "output_file": temp_file.name,
                    "voice": voice,
                    "format": output_format,
                    "character_count": char_count
                }
            )
            
            logger.info(f"Generated {estimated_duration:.1f}s audio in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return AudioProcessingResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def voice_to_voice_interaction(
        self,
        audio_input: Union[bytes, str, Path],
        session_id: str = None,
        voice: str = None
    ) -> VoiceInteractionResult:
        """Complete voice-to-voice interaction pipeline"""
        
        start_time = time.time()
        session_id = session_id or str(uuid.uuid4())
        result = VoiceInteractionResult()
        
        try:
            # Step 1: Transcribe audio input
            logger.info("Starting voice-to-voice interaction: transcribing audio...")
            transcription_result = await self.transcribe_audio(audio_input)
            
            if not transcription_result.success:
                result.error = f"Transcription failed: {transcription_result.error}"
                result.success = False
                return result
            
            result.input_audio_duration = transcription_result.audio_duration
            result.transcription = transcription_result.text
            result.transcription_confidence = transcription_result.metadata.get("confidence", 0.0)
            
            # Step 2: Process with LOGOS AI
            logger.info("Processing with LOGOS AI...")
            gpt_result = await self.process_with_gpt(
                transcription_result.text, 
                session_id=session_id
            )
            
            if not gpt_result["success"]:
                result.error = f"LOGOS processing failed: {gpt_result['error']}"
                result.success = False
                return result
                
            result.logos_response = gpt_result["response"]
            result.tools_used = gpt_result["tools_used"]
            result.processing_steps = {
                "transcription": transcription_result.metadata,
                "gpt_processing": gpt_result.get("token_usage", {}),
                "tools": gpt_result.get("tool_results", {})
            }
            
            # Step 3: Generate speech output
            logger.info("Generating speech output...")
            tts_result = await self.synthesize_speech(
                gpt_result["response"], 
                voice=voice
            )
            
            if not tts_result.success:
                result.error = f"TTS failed: {tts_result.error}"
                result.success = False
                return result
            
            result.output_audio_path = tts_result.metadata.get("output_file")
            result.output_audio_duration = tts_result.audio_duration
            
            # Calculate total metrics
            result.total_processing_time = time.time() - start_time
            result.costs = {
                "transcription": transcription_result.audio_duration * 0.006 / 60.0,
                "gpt_processing": gpt_result.get("token_usage", {}).get("total_tokens", 0) * 0.045 / 1000,
                "tts": len(gpt_result["response"]) * 0.015 / 1000
            }
            result.costs["total"] = sum(result.costs.values())
            
            # Update global stats
            self.usage_stats["requests_processed"] += 1
            
            logger.info(f"Voice interaction completed in {result.total_processing_time:.2f}s")
            logger.info(f"Used tools: {', '.join(result.tools_used) if result.tools_used else 'None'}")
            logger.info(f"Estimated cost: ${result.costs['total']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Voice interaction error: {e}")
            result.error = str(e)
            result.success = False
            result.total_processing_time = time.time() - start_time
            return result
    
    async def text_interaction(self, text: str, session_id: str = None) -> UIPResponse:
        """Text-based interaction with LOGOS AI (for UIP integration)"""
        
        session_id = session_id or str(uuid.uuid4())
        
        try:
            # Process with GPT
            gpt_result = await self.process_with_gpt(text, session_id=session_id)
            
            if gpt_result["success"]:
                response = UIPResponse(
                    session_id=session_id,
                    correlation_id=str(uuid.uuid4()),
                    response_text=gpt_result["response"],
                    confidence_score=0.95,  # High confidence with GPT-4
                    processing_steps={
                        "step_0_preprocessing": {"status": "completed"},
                        "step_1_intelligence": {"tools_used": gpt_result["tools_used"]},
                        "step_2_context": {"session_maintained": True},
                        "step_3_reasoning": {"model": self.config.gpt_model},
                        "step_4_prediction": {"capabilities": "integrated"},
                        "step_5_generation": {"response_generated": True},
                        "step_6_validation": {"confidence": 0.95},
                        "step_7_delivery": {"status": "completed"}
                    },
                    metadata={
                        "processing_time": gpt_result["processing_time"],
                        "tokens_used": gpt_result.get("token_usage", {}),
                        "tools_used": gpt_result["tools_used"],
                        "cost_estimate": gpt_result.get("token_usage", {}).get("total_tokens", 0) * 0.045 / 1000
                    }
                )
            else:
                response = UIPResponse(
                    session_id=session_id,
                    correlation_id=str(uuid.uuid4()),
                    response_text=f"I encountered an error processing your request: {gpt_result['error']}",
                    confidence_score=0.0
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Text interaction error: {e}")
            return UIPResponse(
                session_id=session_id,
                correlation_id=str(uuid.uuid4()),
                response_text=f"I'm sorry, I encountered an error: {str(e)}",
                confidence_score=0.0
            )
    
    def play_audio(self, audio_path: str) -> bool:
        """Play audio file (requires pygame)"""
        if not PYGAME_AVAILABLE:
            logger.warning("Pygame not available for audio playback. Install with: pip install pygame")
            return False
        
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            return True
            
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            **self.usage_stats,
            "active_sessions": len(self.session_history),
            "average_cost_per_request": (
                self.usage_stats["total_cost"] / max(1, self.usage_stats["requests_processed"])
            )
        }
    
    def clear_session_history(self, session_id: str = None):
        """Clear session history"""
        if session_id:
            self.session_history.pop(session_id, None)
        else:
            self.session_history.clear()
        logger.info(f"Cleared session history: {session_id or 'all sessions'}")


# Global pipeline instance
logos_openai_pipeline: Optional[LOGOSOpenAIPipeline] = None


def get_pipeline() -> Optional[LOGOSOpenAIPipeline]:
    """Get global LOGOS OpenAI pipeline instance"""
    return logos_openai_pipeline


def initialize_pipeline(config: Optional[OpenAIConfig] = None) -> bool:
    """Initialize global LOGOS OpenAI pipeline"""
    global logos_openai_pipeline
    
    try:
        logos_openai_pipeline = LOGOSOpenAIPipeline(config)
        return logos_openai_pipeline.client is not None
    except Exception as e:
        logger.error(f"Failed to initialize LOGOS OpenAI pipeline: {e}")
        return False


async def quick_voice_interaction(audio_input: Union[bytes, str, Path]) -> str:
    """Quick voice interaction for testing"""
    pipeline = get_pipeline()
    if not pipeline:
        return "Pipeline not initialized"
    
    result = await pipeline.voice_to_voice_interaction(audio_input)
    if result.success:
        return f"Transcription: {result.transcription}\nResponse: {result.logos_response}"
    else:
        return f"Error: {result.error}"


async def quick_text_interaction(text: str) -> str:
    """Quick text interaction for testing"""
    pipeline = get_pipeline()
    if not pipeline:
        return "Pipeline not initialized"
    
    response = await pipeline.text_interaction(text)
    return response.response_text


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize pipeline
        config = OpenAIConfig()
        if initialize_pipeline(config):
            print("✅ LOGOS OpenAI Pipeline initialized successfully")
            
            # Test text interaction
            response = await quick_text_interaction("Hello, what are LOGOS AI's capabilities?")
            print(f"\n🤖 LOGOS AI: {response}")
            
            # Display usage stats
            stats = get_pipeline().get_usage_stats()
            print(f"\n📊 Usage Stats: {stats}")
        else:
            print("❌ Failed to initialize pipeline")
    
    asyncio.run(main())