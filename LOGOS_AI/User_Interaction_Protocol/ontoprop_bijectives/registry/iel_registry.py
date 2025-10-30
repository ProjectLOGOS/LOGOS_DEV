"""
IEL Registry - UIP Step 2 IEL Ontological Synthesis Gateway

Central registry for dynamic loading and management of all 18 IEL domain
implementations with their bijective ontological property mappings.
Acts as index and orchestrator for IEL ↔ OntoProp processing.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("IEL_ONTO_KIT")


def load_active_domains() -> List[Dict[str, Any]]:
    """
    Load all active IEL domains with their ontological property mappings

    Returns:
        List of active domain configurations with bijective mappings
    """
    try:
        logger.info("Loading active IEL domains from registry")

        # Load bijection mapping data
        bijection_data = _load_bijection_mapping()

        # Get domain availability status
        domain_status = _check_domain_availability()

        # Build active domains list
        active_domains = []

        for domain_name, domain_config in bijection_data.items():
            if domain_status.get(domain_name, {}).get("available", False):
                # Domain is available and properly configured
                active_domain = {
                    "name": domain_name,
                    "ontological_property": domain_config.get(
                        "ontological_property", ""
                    ),
                    "trinity_weights": domain_config.get("trinity_weights", {}),
                    "mapping_confidence": domain_config.get("mapping_confidence", 0.0),
                    "implementation_status": domain_status[domain_name],
                    "activation_priority": _calculate_activation_priority(
                        domain_config
                    ),
                }
                active_domains.append(active_domain)

        # Sort by activation priority
        active_domains.sort(key=lambda d: d["activation_priority"], reverse=True)

        logger.info(f"Loaded {len(active_domains)} active IEL domains")

        return active_domains

    except Exception as e:
        logger.error(f"Failed to load active domains: {e}")
        return _get_fallback_domains()


def get_domain_configuration(domain_name: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for specific IEL domain

    Args:
        domain_name: Name of the IEL domain

    Returns:
        Domain configuration dict or None if not found
    """
    try:
        logger.debug(f"Getting configuration for domain: {domain_name}")

        # Load bijection mapping
        bijection_data = _load_bijection_mapping()

        if domain_name not in bijection_data:
            logger.warning(f"Domain {domain_name} not found in bijection mapping")
            return None

        domain_config = bijection_data[domain_name]

        # Check domain availability
        domain_status = _check_domain_availability().get(domain_name, {})

        # Build complete configuration
        configuration = {
            "name": domain_name,
            "ontological_property": domain_config.get("ontological_property", ""),
            "trinity_weights": domain_config.get("trinity_weights", {}),
            "mapping_confidence": domain_config.get("mapping_confidence", 0.0),
            "implementation_path": domain_status.get("implementation_path", ""),
            "module_status": domain_status.get("available", False),
            "dependencies": _get_domain_dependencies(domain_name),
            "capabilities": _get_domain_capabilities(domain_config),
            "activation_requirements": _get_activation_requirements(domain_config),
        }

        return configuration

    except Exception as e:
        logger.error(f"Failed to get domain configuration for {domain_name}: {e}")
        return None


def _load_bijection_mapping() -> Dict[str, Dict[str, Any]]:
    """Load IEL ↔ ontological property bijection mapping"""
    try:
        # Path to bijection mapping file
        current_dir = Path(__file__).parent.parent
        mapping_file = current_dir / "data" / "iel_ontological_bijection_optimized.json"

        if mapping_file.exists():
            with open(mapping_file, "r", encoding="utf-8") as f:
                bijection_data = json.load(f)

            # Clean data (remove complex/Mandelbrot artifacts per refactor requirements)
            cleaned_data = _clean_bijection_data(bijection_data)

            logger.debug(f"Loaded bijection mapping with {len(cleaned_data)} domains")
            return cleaned_data
        else:
            logger.warning(f"Bijection mapping file not found: {mapping_file}")
            return _get_default_bijection_mapping()

    except Exception as e:
        logger.error(f"Failed to load bijection mapping: {e}")
        return _get_default_bijection_mapping()


def _clean_bijection_data(raw_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Clean bijection data by removing complex/Mandelbrot artifacts"""
    try:
        cleaned_data = {}

        for key, value in raw_data.items():
            if isinstance(value, dict):
                # Remove keys related to complex numbers or Mandelbrot values
                cleaned_value = {}
                for subkey, subvalue in value.items():
                    if subkey not in [
                        "complex_value",
                        "C_value",
                        "mandelbrot_constant",
                        "complex_representation",
                    ]:
                        cleaned_value[subkey] = subvalue

                # Only include if it has essential fields
                if "ontological_property" in cleaned_value:
                    cleaned_data[key] = cleaned_value

        return cleaned_data

    except Exception:
        return {}


def _check_domain_availability() -> Dict[str, Dict[str, Any]]:
    """Check availability of IEL domain implementations"""
    try:
        # Known IEL domain names and their expected paths
        expected_domains = [
            "ModalPraxis",
            "GnosiPraxis",
            "ThemiPraxis",
            "DynaPraxis",
            "HexiPraxis",
            "ChremaPraxis",
            "MuPraxis",
            "TychePraxis",
            "AxioPraxis",
            "ErgoPraxis",
            "AnthropPraxis",
            "TeloPraxis",
            "TopoPraxis",
            "CosmoPraxis",
            "EthoPraxis",
            "LogoPraxis",
            "ChronoPraxis",
            "NumeroPraxis",
        ]

        domain_status = {}

        # Check for domain implementations in main LOGOS system
        logos_intelligence_path = Path(__file__).parent.parent.parent / "intelligence"

        for domain_name in expected_domains:
            status = {
                "available": False,
                "implementation_path": "",
                "module_exists": False,
                "last_checked": "",
            }

            # Check for domain implementation
            possible_paths = [
                logos_intelligence_path
                / "formal_systems"
                / f"{domain_name.lower()}.py",
                logos_intelligence_path / "modal_systems" / f"{domain_name}.py",
                logos_intelligence_path / "domains" / domain_name,
                logos_intelligence_path / f"IEL" / domain_name,
            ]

            for path in possible_paths:
                if path.exists():
                    status["available"] = True
                    status["implementation_path"] = str(path)
                    status["module_exists"] = True
                    break

            # If not found in expected locations, assume available for Step 2 processing
            # (Step 2 works with abstract domain configurations)
            if not status["available"]:
                status["available"] = True  # Enable for bijection processing
                status["implementation_path"] = f"abstract://{domain_name}"
                status["module_exists"] = False

            domain_status[domain_name] = status

        return domain_status

    except Exception as e:
        logger.warning(f"Domain availability check failed: {e}")
        # Return default availability (assume all domains available for Step 2)
        return {
            domain: {
                "available": True,
                "implementation_path": f"abstract://{domain}",
                "module_exists": False,
            }
            for domain in [
                "ModalPraxis",
                "GnosiPraxis",
                "ThemiPraxis",
                "DynaPraxis",
                "HexiPraxis",
                "ChremaPraxis",
                "MuPraxis",
                "TychePraxis",
            ]
        }


def _calculate_activation_priority(domain_config: Dict[str, Any]) -> float:
    """Calculate activation priority for domain"""
    try:
        # Factors affecting priority
        mapping_confidence = domain_config.get("mapping_confidence", 0.0)
        trinity_weights = domain_config.get("trinity_weights", {})

        # Base priority from mapping confidence
        priority = mapping_confidence

        # Boost from Trinity alignment strength
        trinity_sum = sum(abs(weight) for weight in trinity_weights.values())
        trinity_boost = min(trinity_sum / 3.0, 1.0)  # Normalize to [0,1]

        # Combined priority
        combined_priority = priority * 0.7 + trinity_boost * 0.3

        return min(max(combined_priority, 0.0), 1.0)

    except Exception:
        return 0.5


def _get_domain_dependencies(domain_name: str) -> List[str]:
    """Get dependencies for specific domain"""
    # Domain dependency mapping (simplified for Step 2)
    dependencies = {
        "ModalPraxis": ["GnosiPraxis"],
        "GnosiPraxis": [],
        "ThemiPraxis": ["ModalPraxis"],
        "DynaPraxis": ["ThemiPraxis", "ModalPraxis"],
        "HexiPraxis": ["GnosiPraxis"],
        "ChremaPraxis": ["ModalPraxis"],
        "MuPraxis": ["ModalPraxis", "DynaPraxis"],
        "TychePraxis": ["ModalPraxis"],
    }

    return dependencies.get(domain_name, [])


def _get_domain_capabilities(domain_config: Dict[str, Any]) -> List[str]:
    """Get capabilities for domain based on configuration"""
    try:
        capabilities = ["ontological_mapping", "trinity_processing"]

        ontological_prop = domain_config.get("ontological_property", "")

        # Add capabilities based on ontological property
        if "necessity" in ontological_prop.lower():
            capabilities.append("modal_necessity")
        if "knowledge" in ontological_prop.lower():
            capabilities.append("epistemic_processing")
        if "norm" in ontological_prop.lower():
            capabilities.append("normative_reasoning")
        if "action" in ontological_prop.lower():
            capabilities.append("action_theory")
        if "language" in ontological_prop.lower():
            capabilities.append("linguistic_analysis")

        return capabilities

    except Exception:
        return ["ontological_mapping"]


def _get_activation_requirements(domain_config: Dict[str, Any]) -> Dict[str, Any]:
    """Get activation requirements for domain"""
    try:
        trinity_weights = domain_config.get("trinity_weights", {})
        mapping_confidence = domain_config.get("mapping_confidence", 0.0)

        requirements = {
            "minimum_confidence": max(0.3, mapping_confidence * 0.8),
            "trinity_thresholds": {},
            "processing_mode": "standard",
        }

        # Trinity dimension thresholds
        for dimension, weight in trinity_weights.items():
            requirements["trinity_thresholds"][dimension] = abs(weight) * 0.5

        # Processing mode based on complexity
        total_trinity_weight = sum(abs(w) for w in trinity_weights.values())
        if total_trinity_weight > 2.0:
            requirements["processing_mode"] = "intensive"
        elif total_trinity_weight < 1.0:
            requirements["processing_mode"] = "lightweight"

        return requirements

    except Exception:
        return {
            "minimum_confidence": 0.3,
            "trinity_thresholds": {},
            "processing_mode": "standard",
        }


def _get_default_bijection_mapping() -> Dict[str, Dict[str, Any]]:
    """Get default bijection mapping if file loading fails"""
    return {
        "ModalPraxis": {
            "ontological_property": "logical_necessity",
            "trinity_weights": {
                "ethical": 0.3,
                "gnoseological": 0.5,
                "teleological": 0.2,
            },
            "mapping_confidence": 0.9,
        },
        "GnosiPraxis": {
            "ontological_property": "epistemic_knowledge",
            "trinity_weights": {
                "ethical": 0.2,
                "gnoseological": 0.6,
                "teleological": 0.2,
            },
            "mapping_confidence": 0.85,
        },
        "ThemiPraxis": {
            "ontological_property": "normative_obligation",
            "trinity_weights": {
                "ethical": 0.6,
                "gnoseological": 0.2,
                "teleological": 0.2,
            },
            "mapping_confidence": 0.8,
        },
        "DynaPraxis": {
            "ontological_property": "causal_action",
            "trinity_weights": {
                "ethical": 0.25,
                "gnoseological": 0.25,
                "teleological": 0.5,
            },
            "mapping_confidence": 0.75,
        },
        "HexiPraxis": {
            "ontological_property": "linguistic_competence",
            "trinity_weights": {
                "ethical": 0.3,
                "gnoseological": 0.4,
                "teleological": 0.3,
            },
            "mapping_confidence": 0.7,
        },
        "ChremaPraxis": {
            "ontological_property": "temporal_phase",
            "trinity_weights": {
                "ethical": 0.2,
                "gnoseological": 0.3,
                "teleological": 0.5,
            },
            "mapping_confidence": 0.65,
        },
        "MuPraxis": {
            "ontological_property": "recursive_structure",
            "trinity_weights": {
                "ethical": 0.25,
                "gnoseological": 0.5,
                "teleological": 0.25,
            },
            "mapping_confidence": 0.6,
        },
        "TychePraxis": {
            "ontological_property": "probabilistic_event",
            "trinity_weights": {
                "ethical": 0.3,
                "gnoseological": 0.4,
                "teleological": 0.3,
            },
            "mapping_confidence": 0.55,
        },
    }


def _get_fallback_domains() -> List[Dict[str, Any]]:
    """Get fallback domain list if loading fails completely"""
    default_mapping = _get_default_bijection_mapping()

    fallback_domains = []
    for domain_name, config in default_mapping.items():
        fallback_domain = {
            "name": domain_name,
            "ontological_property": config["ontological_property"],
            "trinity_weights": config["trinity_weights"],
            "mapping_confidence": config["mapping_confidence"],
            "implementation_status": {"available": True, "fallback": True},
            "activation_priority": config["mapping_confidence"],
        }
        fallback_domains.append(fallback_domain)

    # Sort by priority
    fallback_domains.sort(key=lambda d: d["activation_priority"], reverse=True)

    return fallback_domains
