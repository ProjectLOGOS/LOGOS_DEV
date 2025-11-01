"""
LOGOS AI - Master Intelligence System
=====================================

This module provides the core LOGOS AI functionality with three-protocol architecture:
- User Interaction Protocol (UIP)
- Computational Logic Protocol (CLP)  
- Autonomous Learning Protocol (ALP)

Key Components:
- LOGOS_AI.py: Main orchestration engine
- User_Interaction_Protocol: Interface management and external integrations
- Computational reasoning and modal logic systems
- Autonomous learning and adaptation capabilities

Usage:
    from LOGOS_AI import LOGOS_AI
    
    # Initialize the main AI system
    logos = LOGOS_AI()
    logos.start_system()
"""

__version__ = "1.0.0"
__author__ = "LOGOS Development Team"

# Main system components
from .LOGOS_AI import LOGOS_AI

# Protocol interfaces 
try:
    from .User_Interaction_Protocol import UIP
except ImportError:
    pass  # Handle missing components gracefully

__all__ = [
    'LOGOS_AI',
    'UIP'
]