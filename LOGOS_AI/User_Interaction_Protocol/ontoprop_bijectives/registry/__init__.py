"""
Registry Module - UIP Step 2 IEL Ontological Synthesis Gateway

IEL domain registry and dynamic import management.
Handles loading and coordination of all 18 IEL ↔ OntoProp bijections.
"""

from .iel_registry import load_active_domains, get_domain_configuration

__all__ = ['load_active_domains', 'get_domain_configuration']