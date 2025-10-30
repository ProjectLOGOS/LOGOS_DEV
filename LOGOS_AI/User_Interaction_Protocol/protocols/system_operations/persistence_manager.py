"""
Persistence Manager - System State Persistence

This module handles persistent storage of adaptive state, configuration,
and system data for the LOGOS adaptive inference system.

⧟ Identity constraint: State integrity via checksums and versioning
⇌ Balance constraint: Storage efficiency vs data completeness
⟹ Causal constraint: State changes tracked with timestamps
⩪ Equivalence constraint: Consistent serialization across sessions
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PersistenceConfig:
    """Configuration for persistence operations."""

    base_path: Path = Path("persistence")
    knowledge_path: Path = Path("persistence/knowledge")
    adaptive_path: Path = Path("persistence/knowledge/adaptive_state.json")
    backup_count: int = 5
    compression_enabled: bool = False
    encryption_enabled: bool = False


class PersistenceManager:
    """Manages persistent storage of system state."""

    def __init__(self, config: Optional[PersistenceConfig] = None):
        self.config = config or PersistenceConfig()
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure required directories exist."""
        try:
            self.config.base_path.mkdir(parents=True, exist_ok=True)
            self.config.knowledge_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Persistence directories initialized: {self.config.base_path}")
        except Exception as e:
            logger.error(f"Failed to create persistence directories: {e}")


def persist_adaptive_state(profile: Dict[str, Any]) -> bool:
    """
    Persist adaptive state profile to storage.

    Args:
        profile: Adaptive inference profile dictionary

    Returns:
        Boolean indicating success/failure
    """
    try:
        # Initialize persistence manager
        manager = PersistenceManager()

        # Add persistence metadata
        enriched_profile = {
            **profile,
            "persistence_meta": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "source": "adaptive_inference_layer",
                "checksum": _calculate_profile_checksum(profile),
            },
        }

        # Ensure directory exists
        manager.config.adaptive_path.parent.mkdir(parents=True, exist_ok=True)

        # Create backup if file exists
        if manager.config.adaptive_path.exists():
            _create_backup(manager.config.adaptive_path, manager.config.backup_count)

        # Write adaptive state
        with open(manager.config.adaptive_path, "w", encoding="utf-8") as f:
            json.dump(enriched_profile, f, indent=2, ensure_ascii=False, default=str)

        # Verify write success
        if manager.config.adaptive_path.exists():
            file_size = manager.config.adaptive_path.stat().st_size
            logger.info(
                f"Adaptive state persisted: {file_size} bytes to {manager.config.adaptive_path}"
            )
            return True
        else:
            logger.error("Adaptive state file not found after write")
            return False

    except Exception as e:
        logger.error(f"Failed to persist adaptive state: {e}")
        return False


def load_adaptive_state() -> Optional[Dict[str, Any]]:
    """
    Load adaptive state profile from storage.

    Returns:
        Adaptive state profile or None if not found/invalid
    """
    try:
        manager = PersistenceManager()

        if not manager.config.adaptive_path.exists():
            logger.info("No existing adaptive state found")
            return None

        with open(manager.config.adaptive_path, "r", encoding="utf-8") as f:
            profile = json.load(f)

        # Validate checksum if available
        if "persistence_meta" in profile and "checksum" in profile["persistence_meta"]:
            # Create profile without metadata for checksum validation
            profile_for_validation = {
                k: v for k, v in profile.items() if k != "persistence_meta"
            }
            expected_checksum = profile["persistence_meta"]["checksum"]
            actual_checksum = _calculate_profile_checksum(profile_for_validation)

            if expected_checksum != actual_checksum:
                logger.warning(
                    f"Adaptive state checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
                )

        logger.info(f"Adaptive state loaded: {len(str(profile))} bytes")
        return profile

    except Exception as e:
        logger.error(f"Failed to load adaptive state: {e}")
        return None


def _calculate_profile_checksum(profile: Dict[str, Any]) -> str:
    """Calculate SHA-256 checksum of profile data."""
    try:
        # Create deterministic JSON representation
        profile_json = json.dumps(
            profile, sort_keys=True, ensure_ascii=True, default=str
        )
        return hashlib.sha256(profile_json.encode("utf-8")).hexdigest()
    except Exception:
        return "checksum_unavailable"


def _create_backup(file_path: Path, max_backups: int = 5):
    """Create timestamped backup of existing file."""
    try:
        if not file_path.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_name(
            f"{file_path.stem}_{timestamp}{file_path.suffix}"
        )

        # Copy file to backup location
        with open(file_path, "rb") as src, open(backup_path, "wb") as dst:
            dst.write(src.read())

        # Clean up old backups
        _cleanup_old_backups(
            file_path.parent, file_path.stem, file_path.suffix, max_backups
        )

        logger.debug(f"Created backup: {backup_path}")

    except Exception as e:
        logger.warning(f"Failed to create backup: {e}")


def _cleanup_old_backups(
    backup_dir: Path, file_stem: str, file_suffix: str, max_backups: int
):
    """Remove old backup files, keeping only the most recent."""
    try:
        # Find all backup files
        pattern = f"{file_stem}_*{file_suffix}"
        backup_files = list(backup_dir.glob(pattern))

        # Sort by creation time, newest first
        backup_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Remove excess backups
        for old_backup in backup_files[max_backups:]:
            old_backup.unlink()
            logger.debug(f"Removed old backup: {old_backup}")

    except Exception as e:
        logger.warning(f"Failed to cleanup old backups: {e}")


def get_persistence_status() -> Dict[str, Any]:
    """Get current persistence system status."""
    try:
        manager = PersistenceManager()

        status = {
            "directories_available": {
                "base_path": manager.config.base_path.exists(),
                "knowledge_path": manager.config.knowledge_path.exists(),
            },
            "adaptive_state_available": manager.config.adaptive_path.exists(),
            "adaptive_state_size": (
                manager.config.adaptive_path.stat().st_size
                if manager.config.adaptive_path.exists()
                else 0
            ),
            "config": {
                "base_path": str(manager.config.base_path),
                "knowledge_path": str(manager.config.knowledge_path),
                "adaptive_path": str(manager.config.adaptive_path),
                "backup_count": manager.config.backup_count,
            },
        }

        # Count backup files
        if manager.config.adaptive_path.exists():
            backup_pattern = f"{manager.config.adaptive_path.stem}_*{manager.config.adaptive_path.suffix}"
            backup_count = len(
                list(manager.config.adaptive_path.parent.glob(backup_pattern))
            )
            status["backup_count"] = backup_count

        return status

    except Exception as e:
        return {"error": str(e), "status": "unavailable"}


__all__ = [
    "persist_adaptive_state",
    "load_adaptive_state",
    "PersistenceManager",
    "PersistenceConfig",
    "get_persistence_status",
]
