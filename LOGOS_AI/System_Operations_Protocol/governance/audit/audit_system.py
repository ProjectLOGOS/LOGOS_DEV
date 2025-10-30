"""
Basic Audit System for LOGOS AGI
Provides audit logging functionality for system operations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AuditSystem:
    """Basic audit logging system"""

    def __init__(self, log_path: str = "logs/audit.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, data: Dict[str, Any], level: str = "INFO"):
        """Log an audit event"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data,
            "level": level,
        }

        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")


# Global audit instance
audit = AuditSystem()
