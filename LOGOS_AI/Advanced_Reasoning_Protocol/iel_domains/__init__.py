"""
Internal Extension Libraries (IEL) Package

This package contains all Internal Extension Libraries for the LOGOS AGI system.
IELs provide specialized reasoning frameworks and domain-specific capabilities.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

__version__ = "0.1.0"


def get_iel_domain_suite() -> Dict[str, Any]:
    """
    Get suite of all IEL domain cores for ARP integration.
    
    Returns:
        Dictionary mapping domain names to their core instances
    """
    suite = {}
    
    try:
        # Import and instantiate each domain core
        from .AestheticoPraxis import AestheticoPraxisCore
        suite["aestheticopraxis"] = AestheticoPraxisCore()
        
        from .AnthroPraxis import AnthroPraxisCore
        suite["anthropraxis"] = AnthroPraxisCore()
        
        from .AxioPraxis import AxioPraxisCore
        suite["axiopraxis"] = AxioPraxisCore()
        
        from .ChronoPraxis import ChronoPraxisCore
        suite["chronopraxis"] = ChronoPraxisCore()
        
        from .CosmoPraxis import CosmoPraxisCore
        suite["cosmopraxis"] = CosmoPraxisCore()
        
        from .ErgoPraxis import ErgoPraxisCore
        suite["ergopraxis"] = ErgoPraxisCore()
        
        from .GlorioPraxis import GlorioPraxisCore
        suite["gloriopraxis"] = GlorioPraxisCore()
        
        from .GnosiPraxis import GnosiPraxisCore
        suite["gnosipraxis"] = GnosiPraxisCore()
        
        from .MakarPraxis import MakarPraxisCore
        suite["makarpraxis"] = MakarPraxisCore()
        
        from .ModalPraxis import ModalPraxisCore
        suite["modalpraxis"] = ModalPraxisCore()
        
        from .PraxeoPraxis import PraxeoPraxisCore
        suite["praxeopraxis"] = PraxeoPraxisCore()
        
        from .RelatioPraxis import RelatioPraxisCore
        suite["relatiopraxis"] = RelatioPraxisCore()
        
        from .TeloPraxis import TeloPraxisCore
        suite["telopraxis"] = TeloPraxisCore()
        
        from .ThemiPraxis import ThemiPraxisCore
        suite["themipraxis"] = ThemiPraxisCore()
        
        from .TheoPraxis import TheoPraxisCore
        suite["theopraxis"] = TheoPraxisCore()
        
        from .TopoPraxis import TopoPraxisCore
        suite["topopraxis"] = TopoPraxisCore()
        
        from .TropoPraxis import TropoPraxisCore
        suite["tropopraxis"] = TropoPraxisCore()
        
        from .ZelosPraxis import ZelosPraxisCore
        suite["zelospraxis"] = ZelosPraxisCore()
        
        logger.info(f"Initialized {len(suite)} IEL domains")
        
    except ImportError as e:
        logger.warning(f"Some IEL domains could not be imported: {e}")
    except Exception as e:
        logger.error(f"Error initializing IEL domains: {e}")
    
    return suite


__all__ = ["get_iel_domain_suite"]
