#!/usr/bin/env python3
"""
LOGOS Directory Restructuring Script
===================================

This script reorganizes the LOGOS_AI directory according to the new
single-agent architecture specification. It creates the proper directory
structure for SOP, UIP, AGP, System Agent, and GUI components.

Run this script to transform the existing structure into the new architecture.
"""

import os
import shutil
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_directory_structure():
    """Create the new LOGOS single-agent directory structure"""
    
    base_path = Path(".")  # Current LOGOS_AI directory
    
    # Define the new directory structure
    new_structure = {
        "system_agent": [
            "agent_controller.py",
            "protocol_manager.py", 
            "todo_coordinator.py",
            "decision_engine.py"
        ],
        "gui_interface": [
            "user_interface.py",
            "visualization/",
            "session_management.py"
        ],
        "SOP": {
            "nexus": ["sop_nexus.py"],
            "infrastructure": [
                "boot_system.py",
                "maintenance.py", 
                "shared_resources.py"
            ],
            "token_system": [
                "token_manager.py",
                "security_validation.py"
            ],
            "gap_detection": [
                "incompleteness_analyzer.py",
                "todo_generator.py"
            ],
            "file_management": [
                "scaffold_library/",
                "backup_storage/",
                "integration_pipeline.py"
            ],
            "data_storage": [
                "system_logs/",
                "todo_json/",
                "system_memory.py"
            ]
        },
        "UIP": {
            "nexus": ["uip_nexus.py"],
            "reasoning_pipeline": [
                "data_parser.py",
                "reasoning_engine.py",
                "output_validator.py"
            ],
            "mode_management": [
                "mode_controller.py",
                "task_processor.py"
            ],
            "token_integration": [
                "token_validator.py"
            ]
        },
        "AGP": {
            "nexus": ["agp_nexus.py"],
            "cognitive_systems": [
                "multimodal_processor.py",
                "recursive_cognition.py",
                "abstraction_engine.py"
            ],
            "bdn_mvs_integration": [
                "banach_processor.py",
                "mvs_coordinator.py", 
                "iterative_analyzer.py"
            ],
            "todo_processing": [
                "todo_importer.py",
                "solution_generator.py",
                "cross_protocol_coord.py"
            ],
            "system_memory": [
                "memory_importer.py",
                "rolling_todo.py",
                "improvement_tracker.py"
            ]
        },
        "shared_resources": [
            "common_utilities.py",
            "data_formats.py",
            "communication_protocols.py"
        ]
    }
    
    logger.info("🏗️ Creating LOGOS Single-Agent Directory Structure")
    
    # Create directories and placeholder files
    create_structure_recursive(base_path, new_structure)
    
    # Create configuration files
    create_config_files(base_path)
    
    logger.info("✅ Directory structure created successfully")


def create_structure_recursive(base_path: Path, structure: dict):
    """Recursively create directory structure"""
    
    for name, contents in structure.items():
        dir_path = base_path / name
        dir_path.mkdir(exist_ok=True)
        logger.info(f"📁 Created directory: {dir_path}")
        
        if isinstance(contents, dict):
            # Nested directory structure
            create_structure_recursive(dir_path, contents)
        elif isinstance(contents, list):
            # Files and subdirectories
            for item in contents:
                if item.endswith('/'):
                    # Subdirectory
                    subdir_path = dir_path / item.rstrip('/')
                    subdir_path.mkdir(exist_ok=True)
                    logger.info(f"📂 Created subdirectory: {subdir_path}")
                else:
                    # File
                    file_path = dir_path / item
                    if not file_path.exists():
                        file_path.touch()
                        logger.info(f"📄 Created file: {file_path}")


def create_config_files(base_path: Path):
    """Create configuration and documentation files"""
    
    # System configuration
    system_config = {
        "system_name": "LOGOS",
        "architecture": "single_agent",
        "version": "2.0.0",
        "protocols": {
            "SOP": {"status": "always_active", "role": "infrastructure"},
            "UIP": {"status": "on_demand", "role": "reasoning"},
            "AGP": {"status": "default_active", "role": "cognitive"}
        },
        "agent": {
            "type": "system_agent",
            "authority": "full_control", 
            "gui_interface": True
        }
    }
    
    config_path = base_path / "system_config.json"
    with open(config_path, 'w') as f:
        json.dump(system_config, f, indent=2)
    logger.info(f"⚙️ Created system configuration: {config_path}")
    
    # Protocol specifications
    protocol_specs = {
        "SOP": {
            "description": "System Operations Protocol - Infrastructure and maintenance hub",
            "responsibilities": ["infrastructure", "maintenance", "token_system", "gap_detection", "file_management"],
            "access": "system_agent_only",
            "lifecycle": "always_active"
        },
        "UIP": {
            "description": "User Interaction Protocol - Reasoning pipeline and data analysis",
            "responsibilities": ["reasoning", "data_parsing", "output_validation"],
            "access": "system_agent_only",
            "lifecycle": "on_demand",
            "modes": ["inactive", "active_targeted", "active_persistent"]
        },
        "AGP": {
            "description": "Advanced General Protocol - Cognitive systems and self-improvement",
            "responsibilities": ["cognitive_processing", "self_improvement", "todo_processing", "bdn_mvs_integration"],
            "access": "system_agent_only", 
            "lifecycle": "default_active"
        }
    }
    
    specs_path = base_path / "protocol_specifications.json"
    with open(specs_path, 'w') as f:
        json.dump(protocol_specs, f, indent=2)
    logger.info(f"📋 Created protocol specifications: {specs_path}")


def migrate_existing_files():
    """Migrate existing files to new structure"""
    
    logger.info("🔄 Starting file migration")
    
    # Migration mappings
    migrations = [
        # Agent system files
        ("agent_system/logos_agent_system.py", "system_agent/agent_controller.py"),
        ("agent_system/protocol_integration.py", "system_agent/protocol_manager.py"),
        
        # UIP files 
        ("User_Interaction_Protocol/", "UIP/reasoning_pipeline/"),
        
        # AGP/Singularity files
        ("singularity/", "AGP/bdn_mvs_integration/"),
        
        # Mathematics/PXL files
        ("mathematics/", "shared_resources/"),
    ]
    
    for source, destination in migrations:
        source_path = Path(source)
        dest_path = Path(destination)
        
        if source_path.exists():
            try:
                if source_path.is_file():
                    # Migrate single file
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, dest_path)
                    logger.info(f"📋 Migrated: {source} → {destination}")
                else:
                    # Migrate directory contents
                    if not dest_path.exists():
                        dest_path.mkdir(parents=True, exist_ok=True)
                    
                    for item in source_path.rglob('*'):
                        if item.is_file():
                            relative_path = item.relative_to(source_path)
                            new_path = dest_path / relative_path
                            new_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, new_path)
                    
                    logger.info(f"📁 Migrated directory: {source} → {destination}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Migration failed for {source}: {e}")
        else:
            logger.info(f"ℹ️ Source not found, skipping: {source}")


def create_integration_readme():
    """Create README for the new architecture"""
    
    readme_content = """# LOGOS Single-Agent Architecture

## Overview

LOGOS now operates with a single System Agent that controls all protocols and user interactions. This architecture provides better control, security, and coordination between system components.

## Directory Structure

- `system_agent/` - Single controlling agent for all system operations
- `gui_interface/` - User interaction layer (no direct protocol access)
- `SOP/` - System Operations Protocol (infrastructure, maintenance, tokens)
- `UIP/` - User Interaction Protocol (reasoning pipeline, data analysis)
- `AGP/` - Advanced General Protocol (cognitive systems, self-improvement)
- `shared_resources/` - Cross-system utilities and common components

## Protocol Lifecycle

1. **Boot**: All protocols initialize, SOP remains active
2. **Testing**: SOP tests each protocol for functionality and alignment  
3. **Token Distribution**: SOP issues operation tokens to validated protocols
4. **Operation**: System Agent controls protocol modes and task assignment
5. **TODO Processing**: Gap detection → JSON data → Solution → Integration

## Key Features

- **Single Agent Control**: Only System Agent manages all protocols
- **Token-Based Authorization**: SOP manages all system authorization
- **Gap-Driven Improvement**: Continuous system enhancement through TODO pipeline
- **Self-Improving Architecture**: AGP drives autonomous system improvements
- **Secure User Interface**: GUI layer prevents direct protocol access

## Getting Started

1. Run system initialization: `python system_agent/agent_controller.py`
2. Access GUI interface: `python gui_interface/user_interface.py`
3. Monitor system logs in `SOP/data_storage/system_logs/`

## Architecture Documentation

See `SINGLE_AGENT_ARCHITECTURE.md` for detailed specifications.
"""
    
    readme_path = Path("README_NEW_ARCHITECTURE.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"📖 Created architecture README: {readme_path}")


if __name__ == "__main__":
    print("🚀 LOGOS Directory Restructuring")
    print("=" * 50)
    
    # Confirm before proceeding
    response = input("This will create new directory structure. Continue? (y/N): ").strip().lower()
    
    if response == 'y':
        create_directory_structure()
        migrate_existing_files()  
        create_integration_readme()
        
        print("\n✅ Restructuring Complete!")
        print("📁 New directory structure created")
        print("📋 Existing files migrated where possible")
        print("📖 Documentation updated")
        print("\nNext steps:")
        print("1. Review migrated files in new locations")
        print("2. Implement System Agent controller")
        print("3. Build SOP token and gap detection systems")
        print("4. Create protocol nexus interfaces")
        
    else:
        print("❌ Restructuring cancelled")