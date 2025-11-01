"""
LOGOS ARP Stack Compiler
========================

Compiles reasoning data from ARP components into structured objects.
Collects data from reasoning engines, learning modules, and mathematical foundations,
compiles into unified data objects, applies translations, and catalogs by origin.

Core Responsibilities:
- Data collection from ARP reasoning engines
- Object compilation and structuring
- Translation and formatting
- Origin-based cataloging (reasoning/learning/mathematical)
- Integration with LOGOS data pipeline
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from reasoning_engines import get_reasoning_engine_suite

# Import learning modules and IEL components
try:
    from learning_modules import (
        UnifiedTorchAdapter,
        FeatureExtractor,
        DeepLearningAdapter
    )
    LEARNING_MODULES_AVAILABLE = True
except ImportError:
    LEARNING_MODULES_AVAILABLE = False
    UnifiedTorchAdapter = None
    FeatureExtractor = None
    DeepLearningAdapter = None

try:
    from iel_domains import get_iel_domain_suite
    from iel_toolkit import IELOverlay, IELRegistry
    IEL_COMPONENTS_AVAILABLE = True
except ImportError:
    IEL_COMPONENTS_AVAILABLE = False
    get_iel_domain_suite = None
    IELOverlay = None
    IELRegistry = None

logger = logging.getLogger(__name__)


class DataOrigin(Enum):
    """Origins of compiled data objects"""
    REASONING = "reasoning"
    LEARNING = "learning"
    MATHEMATICAL = "mathematical"
    TEMPORAL = "temporal"
    LANGUAGE = "language"
    BAYESIAN = "bayesian"


class CompilationResult(Enum):
    """Results of compilation operations"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    EMPTY = "empty"


@dataclass
class CompiledDataObject:
    """Structured data object compiled from ARP components"""
    object_id: str
    origin: DataOrigin
    timestamp: datetime
    data_type: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    translations: List[Dict[str, Any]] = field(default_factory=list)
    validation_status: str = "pending"
    compilation_result: CompilationResult = CompilationResult.SUCCESS


@dataclass
class ARPStackCompilation:
    """Complete compilation of ARP stack data"""
    compilation_id: str
    timestamp: datetime
    compiled_objects: List[CompiledDataObject] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class ARPStackCompiler:
    """
    Compiles data from ARP reasoning engines into structured objects.

    Collects data from all reasoning engines, applies translations,
    formats correctly, and catalogs by origin for downstream processing.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ARP Stack Compiler.

        Args:
            config_path: Path to compiler configuration
        """
        self.config_path = config_path or "config/arp_compiler.json"
        
        # Initialize logger first
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.reasoning_suite = get_reasoning_engine_suite()
        
        # Initialize learning modules
        self.learning_suite = self._initialize_learning_modules()
        
        # Initialize IEL components
        self.iel_suite = self._initialize_iel_components()
        
        self.compilation_history: List[ARPStackCompilation] = []

        # Load configuration
        self.config = self._load_config()

        self.logger.info("ARP Stack Compiler initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load compiler configuration"""
        try:
            config_path = Path(self.config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}")

        # Default configuration
        return {
            "data_collection": {
                "max_objects_per_compilation": 100,
                "timeout_seconds": 30,
                "parallel_collection": True
            },
            "translation": {
                "auto_translate": True,
                "target_formats": ["json", "structured"],
                "preserve_original": True
            },
            "validation": {
                "enable_validation": True,
                "strict_mode": False
            },
            "cataloging": {
                "by_origin": True,
                "by_type": True,
                "include_metadata": True
            }
        }

    def _initialize_learning_modules(self) -> Dict[str, Any]:
        """Initialize learning modules suite"""
        suite = {}
        
        if not LEARNING_MODULES_AVAILABLE:
            self.logger.warning("Learning modules not available")
            return suite
            
        try:
            # Add UnifiedTorchAdapter if not already in reasoning suite
            if UnifiedTorchAdapter and "torch" not in self.reasoning_suite:
                suite["torch_adapter"] = UnifiedTorchAdapter()
                
            # Add FeatureExtractor
            if FeatureExtractor:
                suite["feature_extractor"] = FeatureExtractor()
                
            # Add DeepLearningAdapter
            if DeepLearningAdapter:
                suite["deep_learning"] = DeepLearningAdapter()
                
            self.logger.info(f"Initialized {len(suite)} learning modules")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize learning modules: {e}")
            
        return suite
        
    def _initialize_iel_components(self) -> Dict[str, Any]:
        """Initialize IEL domains and toolkit components"""
        suite = {}
        
        if not IEL_COMPONENTS_AVAILABLE:
            self.logger.warning("IEL components not available")
            return suite
            
        try:
            # Add IEL domain suite
            if get_iel_domain_suite:
                suite["iel_domains"] = get_iel_domain_suite()
                
            # Add IEL toolkit components
            if IELOverlay:
                suite["iel_overlay"] = IELOverlay()
                
            if IELRegistry:
                suite["iel_registry"] = IELRegistry()
                
            self.logger.info(f"Initialized {len(suite)} IEL components")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize IEL components: {e}")
            
        return suite

    async def compile_arp_stack(self) -> ARPStackCompilation:
        """
        Compile complete ARP stack into structured data objects.

        Returns:
            Complete compilation result
        """
        compilation_id = f"arp_comp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        compilation = ARPStackCompilation(
            compilation_id=compilation_id,
            timestamp=datetime.now(timezone.utc)
        )

        try:
            self.logger.info(f"Starting ARP stack compilation: {compilation_id}")

            # Collect data from all ARP components
            collected_data = {}
            
            # Collect from reasoning engines
            reasoning_data = await self._collect_reasoning_data()
            collected_data.update(reasoning_data)
            
            # Collect from learning modules
            learning_data = await self._collect_learning_data()
            collected_data.update(learning_data)
            
            # Collect from IEL components
            iel_data = await self._collect_iel_data()
            collected_data.update(iel_data)

            # Compile into structured objects
            compiled_objects = await self._compile_data_objects(collected_data)

            # Apply translations
            if self.config["translation"]["auto_translate"]:
                compiled_objects = await self._apply_translations(compiled_objects)

            # Validate compiled objects
            if self.config["validation"]["enable_validation"]:
                compiled_objects = await self._validate_objects(compiled_objects)

            # Catalog by origin
            cataloged_objects = self._catalog_objects(compiled_objects)

            compilation.compiled_objects = cataloged_objects
            compilation.summary = self._generate_summary(cataloged_objects)
            compilation.compilation_result = CompilationResult.SUCCESS

            self.logger.info(f"ARP stack compilation completed: {len(cataloged_objects)} objects")

        except Exception as e:
            error_msg = f"ARP stack compilation failed: {e}"
            self.logger.error(error_msg)
            compilation.errors.append(error_msg)
            compilation.compilation_result = CompilationResult.FAILED

        # Store in history
        self.compilation_history.append(compilation)

        return compilation

    async def _collect_reasoning_data(self) -> Dict[str, Any]:
        """
        Collect data from all reasoning engines in the suite.

        Returns:
            Collected data organized by engine
        """
        collected_data = {}
        timeout = self.config["data_collection"]["timeout_seconds"]

        if self.config["data_collection"]["parallel_collection"]:
            # Parallel collection
            tasks = []
            for engine_name, engine in self.reasoning_suite.items():
                task = asyncio.create_task(self._collect_engine_data(engine_name, engine))
                tasks.append(task)

            # Wait for all with timeout
            try:
                results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
                for engine_name, result in zip(self.reasoning_suite.keys(), results):
                    if isinstance(result, Exception):
                        self.logger.warning(f"Failed to collect from {engine_name}: {result}")
                        collected_data[engine_name] = {"error": str(result)}
                    else:
                        collected_data[engine_name] = result
            except asyncio.TimeoutError:
                self.logger.warning("Data collection timed out")
                collected_data["timeout"] = True
        else:
            # Sequential collection
            for engine_name, engine in self.reasoning_suite.items():
                try:
                    data = await asyncio.wait_for(
                        self._collect_engine_data(engine_name, engine),
                        timeout=timeout
                    )
                    collected_data[engine_name] = data
                except Exception as e:
                    self.logger.warning(f"Failed to collect from {engine_name}: {e}")
                    collected_data[engine_name] = {"error": str(e)}

        return collected_data

    async def _collect_learning_data(self) -> Dict[str, Any]:
        """
        Collect data from learning modules.

        Returns:
            Collected data organized by module
        """
        collected_data = {}
        
        if not self.learning_suite:
            return collected_data
            
        for module_name, module in self.learning_suite.items():
            try:
                data = await self._collect_module_data(module_name, module)
                collected_data[module_name] = data
            except Exception as e:
                self.logger.warning(f"Failed to collect from learning module {module_name}: {e}")
                collected_data[module_name] = {"error": str(e)}
                
        return collected_data
        
    async def _collect_iel_data(self) -> Dict[str, Any]:
        """
        Collect data from IEL components.

        Returns:
            Collected data organized by component
        """
        collected_data = {}
        
        if not self.iel_suite:
            return collected_data
            
        for component_name, component in self.iel_suite.items():
            try:
                if component_name == "iel_domains" and isinstance(component, dict):
                    # Handle domain suite specially
                    for domain_name, domain in component.items():
                        try:
                            data = await self._collect_module_data(f"iel_{domain_name}", domain)
                            collected_data[f"iel_{domain_name}"] = data
                        except Exception as e:
                            collected_data[f"iel_{domain_name}"] = {"error": str(e)}
                else:
                    data = await self._collect_module_data(component_name, component)
                    collected_data[component_name] = data
            except Exception as e:
                self.logger.warning(f"Failed to collect from IEL component {component_name}: {e}")
                collected_data[component_name] = {"error": str(e)}
                
        return collected_data

    async def _collect_module_data(self, module_name: str, module: Any) -> Dict[str, Any]:
        """
        Collect data from a specific learning module or IEL component.

        Args:
            module_name: Name of the module/component
            module: Module/component instance

        Returns:
            Collected data
        """
        data = {
            "module_name": module_name,
            "module_type": type(module).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {}
        }

        try:
            # Try different data collection methods for modules
            if hasattr(module, 'get_summary'):
                data["data"]["summary"] = module.get_summary()
            elif hasattr(module, 'summary'):
                data["data"]["summary"] = module.summary

            # Try to get current state
            if hasattr(module, 'get_state'):
                data["data"]["state"] = module.get_state()
            elif hasattr(module, 'state'):
                data["data"]["state"] = module.state

            # Try to get recent activity
            if hasattr(module, 'get_recent_activity'):
                data["data"]["activity"] = module.get_recent_activity()
            elif hasattr(module, 'activity_history'):
                data["data"]["activity"] = module.activity_history[-10:]

        except Exception as e:
            data["error"] = str(e)
            self.logger.debug(f"Error collecting data from {module_name}: {e}")

        return data

    async def _collect_engine_data(self, engine_name: str, engine: Any) -> Dict[str, Any]:
        """
        Collect data from a specific reasoning engine.

        Args:
            engine_name: Name of the engine
            engine: Engine instance

        Returns:
            Collected data
        """
        data = {
            "engine_name": engine_name,
            "engine_type": type(engine).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {}
        }

        try:
            # Try different data collection methods
            if hasattr(engine, 'get_inference_summary'):
                data["data"]["summary"] = engine.get_inference_summary()
            elif hasattr(engine, 'get_summary'):
                data["data"]["summary"] = engine.get_summary()
            elif hasattr(engine, 'summary'):
                data["data"]["summary"] = engine.summary

            # Try to get current state
            if hasattr(engine, 'get_state'):
                data["data"]["state"] = engine.get_state()
            elif hasattr(engine, 'state'):
                data["data"]["state"] = engine.state

            # Try to get recent inferences/results
            if hasattr(engine, 'inference_history'):
                data["data"]["history"] = engine.inference_history[-10:]  # Last 10 items
            elif hasattr(engine, 'history'):
                data["data"]["history"] = engine.history[-10:]

        except Exception as e:
            data["error"] = str(e)
            self.logger.debug(f"Error collecting data from {engine_name}: {e}")

        return data

    async def _compile_data_objects(self, collected_data: Dict[str, Any]) -> List[CompiledDataObject]:
        """
        Compile collected data into structured objects.

        Args:
            collected_data: Data collected from engines

        Returns:
            List of compiled data objects
        """
        compiled_objects = []

        for engine_name, data in collected_data.items():
            if "error" in data or "timeout" in data:
                continue

            try:
                # Determine origin
                origin = self._determine_origin(engine_name)

                # Create compiled object
                obj = CompiledDataObject(
                    object_id=f"{engine_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                    origin=origin,
                    timestamp=datetime.now(timezone.utc),
                    data_type=f"{engine_name}_data",
                    content=data,
                    metadata={
                        "engine_name": engine_name,
                        "collection_method": "automated",
                        "data_size": len(str(data))
                    }
                )

                compiled_objects.append(obj)

            except Exception as e:
                self.logger.warning(f"Failed to compile object for {engine_name}: {e}")

        return compiled_objects

    def _determine_origin(self, component_name: str) -> DataOrigin:
        """Determine the origin of data based on component name"""
        origin_map = {
            # Reasoning engines
            "bayesian": DataOrigin.BAYESIAN,
            "natural_language": DataOrigin.LANGUAGE,
            "semantic": DataOrigin.LANGUAGE,
            "translation_bridge": DataOrigin.LANGUAGE,
            "temporal": DataOrigin.TEMPORAL,
            "lambda_calculus": DataOrigin.REASONING,
            
            # Learning modules
            "torch": DataOrigin.LEARNING,
            "torch_adapter": DataOrigin.LEARNING,
            "feature_extractor": DataOrigin.LEARNING,
            "deep_learning": DataOrigin.LEARNING,
            
            # IEL components (these are reasoning-based but domain-specific)
            "iel_overlay": DataOrigin.REASONING,
            "iel_registry": DataOrigin.REASONING,
        }
        
        # Handle IEL domains dynamically
        if component_name.startswith("iel_"):
            return DataOrigin.REASONING
            
        return origin_map.get(component_name, DataOrigin.REASONING)

    async def _apply_translations(self, objects: List[CompiledDataObject]) -> List[CompiledDataObject]:
        """
        Apply translations to compiled objects.

        Args:
            objects: Objects to translate

        Returns:
            Objects with translations applied
        """
        translated_objects = []

        for obj in objects:
            try:
                # Apply JSON formatting
                if "json" in self.config["translation"]["target_formats"]:
                    json_translation = {
                        "format": "json",
                        "content": json.dumps(obj.content, default=str, indent=2),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    obj.translations.append(json_translation)

                # Apply structured formatting
                if "structured" in self.config["translation"]["target_formats"]:
                    structured_translation = {
                        "format": "structured",
                        "content": self._structure_data(obj.content),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    obj.translations.append(structured_translation)

            except Exception as e:
                self.logger.warning(f"Translation failed for {obj.object_id}: {e}")

            translated_objects.append(obj)

        return translated_objects

    def _structure_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Structure raw data into standardized format"""
        structured = {
            "header": {
                "data_type": data.get("engine_type", "unknown"),
                "timestamp": data.get("timestamp"),
                "version": "1.0"
            },
            "body": data.get("data", {}),
            "metadata": {
                "source": data.get("engine_name"),
                "structured_by": "arp_compiler"
            }
        }
        return structured

    async def _validate_objects(self, objects: List[CompiledDataObject]) -> List[CompiledDataObject]:
        """
        Validate compiled objects.

        Args:
            objects: Objects to validate

        Returns:
            Validated objects
        """
        validated_objects = []

        for obj in objects:
            try:
                # Basic validation
                if not obj.content:
                    obj.validation_status = "empty"
                    obj.compilation_result = CompilationResult.EMPTY
                elif len(str(obj.content)) < 10:
                    obj.validation_status = "minimal"
                else:
                    obj.validation_status = "valid"

                # Strict validation if enabled
                if self.config["validation"]["strict_mode"]:
                    if not self._strict_validation(obj):
                        obj.validation_status = "invalid"
                        obj.compilation_result = CompilationResult.FAILED

            except Exception as e:
                obj.validation_status = "error"
                obj.compilation_result = CompilationResult.FAILED
                self.logger.warning(f"Validation failed for {obj.object_id}: {e}")

            validated_objects.append(obj)

        return validated_objects

    def _strict_validation(self, obj: CompiledDataObject) -> bool:
        """Perform strict validation of compiled object"""
        # Check required fields
        required_fields = ["engine_name", "engine_type", "timestamp"]
        content = obj.content

        for field in required_fields:
            if field not in content:
                return False

        # Check data integrity
        if "data" not in content:
            return False

        return True

    def _catalog_objects(self, objects: List[CompiledDataObject]) -> List[CompiledDataObject]:
        """
        Catalog objects by origin and type.

        Args:
            objects: Objects to catalog

        Returns:
            Cataloged objects with metadata
        """
        for obj in objects:
            # Add cataloging metadata
            if self.config["cataloging"]["by_origin"]:
                obj.metadata["catalog_origin"] = obj.origin.value

            if self.config["cataloging"]["by_type"]:
                obj.metadata["catalog_type"] = obj.data_type

            if self.config["cataloging"]["include_metadata"]:
                obj.metadata["catalog_timestamp"] = datetime.now(timezone.utc).isoformat()
                obj.metadata["catalog_version"] = "1.0"

        return objects

    def _generate_summary(self, objects: List[CompiledDataObject]) -> Dict[str, Any]:
        """Generate compilation summary"""
        summary = {
            "total_objects": len(objects),
            "compilation_timestamp": datetime.now(timezone.utc).isoformat(),
            "origins": {},
            "types": {},
            "validation_status": {},
            "errors": len([obj for obj in objects if obj.compilation_result == CompilationResult.FAILED])
        }

        # Count by origin
        for obj in objects:
            origin = obj.origin.value
            summary["origins"][origin] = summary["origins"].get(origin, 0) + 1

            data_type = obj.data_type
            summary["types"][data_type] = summary["types"].get(data_type, 0) + 1

            status = obj.validation_status
            summary["validation_status"][status] = summary["validation_status"].get(status, 0) + 1

        return summary

    async def get_recent_compilations(self, limit: int = 5) -> List[ARPStackCompilation]:
        """
        Get recent compilation results.

        Args:
            limit: Maximum number of compilations to return

        Returns:
            Recent compilations
        """
        return self.compilation_history[-limit:] if self.compilation_history else []

    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get overall compilation statistics"""
        if not self.compilation_history:
            return {"total_compilations": 0}

        stats = {
            "total_compilations": len(self.compilation_history),
            "successful_compilations": len([c for c in self.compilation_history if c.compilation_result == CompilationResult.SUCCESS]),
            "failed_compilations": len([c for c in self.compilation_history if c.compilation_result == CompilationResult.FAILED]),
            "total_objects_compiled": sum(len(c.compiled_objects) for c in self.compilation_history),
            "average_objects_per_compilation": 0
        }

        if stats["total_compilations"] > 0:
            stats["average_objects_per_compilation"] = stats["total_objects_compiled"] / stats["total_compilations"]

        return stats


# Convenience functions
async def compile_arp_stack(config_path: Optional[str] = None) -> ARPStackCompilation:
    """
    Convenience function to compile ARP stack.

    Args:
        config_path: Path to compiler configuration

    Returns:
        Compilation result
    """
    compiler = ARPStackCompiler(config_path)
    return await compiler.compile_arp_stack()


def get_arp_compiler_stats(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get ARP compiler statistics.

    Args:
        config_path: Path to compiler configuration

    Returns:
        Compiler statistics
    """
    compiler = ARPStackCompiler(config_path)
    return compiler.get_compilation_stats()