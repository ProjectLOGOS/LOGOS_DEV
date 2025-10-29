# OBDC - Object-Based Denotational Calculus kernel
# Updated for GPT Consolidation Phase 1

from .system_imports import *
from .unified_classes import *
from .logos_core import *
from .adaptive_reasoning import *
from .runtime_services import *

# Optional distributed imports - handle missing dependencies gracefully
try:
    from .distributed import *
except ImportError as e:
    print(f"Warning: Distributed components not available: {e}")

from .learning import LearningCycleManager

# Optional language imports - handle missing dependencies gracefully
try:
    from .language import *
except ImportError as e:
    print(f"Warning: Language components not available: {e}")
