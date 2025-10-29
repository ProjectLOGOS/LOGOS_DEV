"""
Protocol Shared System Imports
===============================

Common imports and utilities used across UIP and SOP protocols.
Centralized import management for consistent protocol operation.
"""

# Standard library imports
import os
import sys
import json
import logging
import threading
import time
import uuid
import asyncio
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import defaultdict, deque

# Configure protocol logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/protocol.log', mode='a')
    ]
)

__all__ = [
    # Standard library
    'os', 'sys', 'json', 'logging', 'threading', 'time', 'uuid', 'asyncio', 'hashlib',
    
    # Abstract classes  
    'ABC', 'abstractmethod',
    
    # Data structures
    'dataclass', 'field', 'datetime', 'timedelta', 'Enum', 'Path',
    
    # Typing
    'Any', 'Dict', 'List', 'Optional', 'Tuple', 'Union', 'Callable',
    
    # Collections
    'defaultdict', 'deque'
]