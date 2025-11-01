# LOGOS AI System - Copilot Instructions

## Architecture Overview

LOGOS implements a **5-Protocol Architecture** for AGI development:

- **SOP** (System Operations Protocol): Infrastructure, governance, resource management
- **ARP** (Advanced Reasoning Protocol): Mathematical reasoning with Trinity Logic (𝔼-𝔾-𝕋) and 18+ IEL domains
- **SCP** (Synthetic Cognition Protocol): MVS/BDN cognitive systems and consciousness models
- **UIP** (User Interface Protocol): 7-step reasoning pipeline for user interactions
- **LAP** (Logos Agentic Protocol): Multi-agent coordination and autonomous operations

**Foundation**: Protopraxic Logic (PXL) core with Coq formal verification and Python execution.

## Critical Workflows

### System Initialization (Strict Order)
```bash
# Phase 1: Foundation
python3 LOGOS_AI/System_Operations_Protocol/sop_operations.py --initialize --full-stack

# Phase 2: Reasoning Foundation  
python3 LOGOS_AI/Advanced_Reasoning_Protocol/arp_operations.py --initialize --full-stack

# Phase 3: Cognitive Enhancement
python3 LOGOS_AI/Synthetic_Cognition_Protocol/scp_operations.py --initialize --full-stack

# Phase 4: User Interface
python3 LOGOS_AI/User_Interface_Protocol/uip_operations.py --initialize --full-stack

# Phase 5: Agent Systems
python3 LOGOS_AI/Logos_Agentic_Protocol/lap_operations.py --initialize --full-stack
```

### Coq Formal Verification
```bash
# Full system build
make -j

# Domain-specific builds
make domain-compatibilism
make domain-empiricism
make domain-modal-ontology

# Individual verification
coqc modules/IEL/ChronoPraxis/domains/Empiricism/Relativity.v
coqc tests/DomainProperties.v
```

### Development Tools
- **DEV_scripts/**: Centralized development utilities (`test_*.py`, `fix_*.py`, `create_*.py`)
- **Testing**: `pytest -m "smoke or not smoke"` for comprehensive validation
- **Linting**: `ruff check .`, `black --check .`, `mypy .`

## Project Conventions

### File Organization
- **Protocol Operations**: Each protocol has `{protocol}_operations.py` and `PROTOCOL_OPERATIONS.txt`
- **IEL Domains**: Located in `LOGOS_AI/Advanced_Reasoning_Protocol/iel_domains/` with Coq (.v) and Python implementations
- **Documentation**: Root-level `Documentation/` directory
- **Development Notes**: `Project_Notes/` for research and implementation logs

### Code Patterns
```python
# Protocol initialization pattern
from iel_domains.{DomainName} import {DomainName}Core

domain_core = {DomainName}Core()
result = domain_core.process_data(data)
```

### Trinity Logic Integration
- **𝔼 (Existence)**: Ontological grounding
- **𝔾 (Goodness)**: Axiological value assessment  
- **𝕋 (Truth)**: Epistemic correctness

All IEL domains map to and extend these operators.

## Integration Points

### Cross-Protocol Communication
Requests route through SOP for authorization, then to target protocols. Example:
```python
# SOP validates and routes
validated_request = sop.validate_request(user_input)
response = target_protocol.process(validated_request)
```

### IEL Domain Usage
```python
from iel_domains.TeloPraxis import TeloPraxisCore

teleologist = TeloPraxisCore()
goal_hierarchy = teleologist.construct_goal_hierarchy(objectives)
```

### External Dependencies
- **PyTorch/TensorFlow**: ML integration in ARP
- **Coq 8.15+**: Formal verification
- **PyMC/ArviZ**: Bayesian inference
- **NetworkX**: Graph analysis

## Key Files & Directories

- `LOGOS_AI/LOGOS_AI.py`: Master system controller
- `LOGOS_AI/system_config.json`: Protocol configuration
- `Documentation/META_ORDER_OF_OPERATIONS.md`: Operational framework
- `DEV_scripts/`: All development utilities
- `LOGOS_AI/Advanced_Reasoning_Protocol/iel_domains/`: 18+ domain implementations

## Development Guidelines

- **Protocol Separation**: Maintain clear boundaries between SOP/ARP/SCP/UIP/LAP
- **IEL Extensions**: New domains extend PXL core with Trinity mappings
- **Testing**: Use pytest with smoke markers for critical path validation
- **Documentation**: Update both code comments and `Documentation/` files
- **Dependencies**: Add to `Documentation/requirements.txt` and test integration

## Common Patterns

### Error Handling
```python
try:
    result = protocol_operation(data)
except ProtocolError as e:
    logger.error(f"Protocol failure: {e}")
    # SOP handles recovery
    sop.recover_operation(protocol_id, e)
```

### Configuration Access
```python
import json
with open('LOGOS_AI/system_config.json') as f:
    config = json.load(f)
protocol_config = config['protocols'][protocol_id]
```</content>
<parameter name="filePath">c:\Users\proje\Downloads\LOGOS_DEV-main.zip\LOGOS_DEV\.github\copilot-instructions.md