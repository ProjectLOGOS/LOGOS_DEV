# OBDC - Object-Based Denotational Calculus kernel
# Updated for GPT Consolidation Phase 1

# Import available modules
try:
    from .entry import LOGOSCore
except ImportError:
    pass

try:
    from .system_imports import *
except ImportError:
    pass

try:
    from .unified_classes import *
except ImportError:
    pass

try:
    from .worker_integration import *
except ImportError:
    pass

try:
    from .worker_kernel import *
except ImportError:
    pass

try:
    from .iel_integration import *
except ImportError:
    pass

try:
    from .kernel import *
except ImportError:
    pass

try:
    from .logos_monitor import *
except ImportError:
    pass
except ImportError as e:
    print(f"Warning: Language components not available: {e}")
