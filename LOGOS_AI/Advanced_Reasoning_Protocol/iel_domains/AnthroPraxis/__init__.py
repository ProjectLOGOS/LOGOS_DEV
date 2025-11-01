"""
Anthropraxis Domain: Human-AI Interaction Praxis

This domain focuses on the praxis of human-AI interaction, including:
- Natural language interfaces
- Collaborative decision-making
- Ethical alignment
- User experience design for AI systems
"""

from .collaboration import CollaborativeReasoning
from .ethics import EthicalAlignment
from .interaction_models import HumanAIInterface

__all__ = ["HumanAIInterface", "CollaborativeReasoning", "EthicalAlignment"]
