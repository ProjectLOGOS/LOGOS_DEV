# LOGOS_V2 Repository Structure Rules

## Overview
This document establishes codified rules for maintaining the clean, professional repository structure of LOGOS_V2 as a production-ready AGI system. The repository is organized into three distinct categories: LOGOS_V2 (production code), LOGOS_V1 (legacy codebase), and Project_Notes (internal development files).

## Repository Categories

### LOGOS_V2 (Production Code)
**Purpose**: Contains only production-ready, deployable AGI system components
**Location**: `/LOGOS_V2/`
**Contents**:
- Core AGI engine and reasoning systems
- Production APIs and services
- Verified mathematical formalisms (IEL, PXL)
- Deployment configurations
- External libraries (vendored dependencies)
- Security and safety frameworks
- Testing suites for production validation

**Strict Rules**:
1. **No development artifacts**: No TODO files, scratch notes, or temporary files
2. **No legacy code**: All code must be current and actively maintained
3. **No experimental features**: Only stable, tested functionality
4. **Clean imports**: All dependencies must be properly managed
5. **Documentation**: Complete READMEs and API documentation required
6. **Testing**: 100% test coverage for critical paths

### LOGOS_V1 (Legacy Codebase)
**Purpose**: Preservation of original LOGOS_AGI and all non-V2 development
**Location**: `/LOGOS_V1/`
**Contents**:
- Original LOGOS_AGI codebase
- Coq formal verification proofs
- Legacy services and APIs
- Historical development artifacts
- Deprecated but potentially useful code

**Rules**:
1. **Read-only**: No new development in LOGOS_V1
2. **Preservation**: Maintain original structure and functionality
3. **Reference only**: Use for historical context or migration reference
4. **No active deployment**: Not intended for production use

### Project_Notes (Internal Development)
**Purpose**: Internal development files, notes, and artifacts
**Location**: `/Project_Notes/`
**Contents**:
- Development notes and TODO files
- Scratch code and experiments
- Meeting notes and planning documents
- Debug logs and crash dumps
- Temporary files and backups
- Research papers and references
- Build artifacts and cache files

**Rules**:
1. **No production code**: Never move production code here
2. **Disposable**: Files here may be cleaned up periodically
3. **Private**: Contains internal-only information
4. **Unstructured**: No enforced organization required

## Directory Structure (LOGOS_V2)

```
LOGOS_V2/
├── IEL/                          # Integrated Ethical Logic formalisms
│   ├── AnthroPraxis/            # Human-centric ethical reasoning
│   ├── AxioPraxis/              # Axiological foundations
│   ├── ChronoPraxis/            # Temporal reasoning domains
│   ├── CosmoPraxis/             # Cosmological reasoning
│   ├── ErgoPraxis/              # Work and action theory
│   ├── GnosiPraxis/             # Epistemological reasoning
│   ├── ModalPraxis/             # Modal logic systems
│   ├── TeloPraxis/              # Teleological reasoning
│   ├── ThemiPraxis/             # Ethical reasoning
│   ├── TheoPraxis/              # Theological reasoning
│   ├── TopoPraxis/              # Topological reasoning
│   └── TropoPraxis/             # Figurative reasoning
├── PXL_Mathematics/             # PXL mathematical foundations
│   ├── Algebra/                 # Algebraic structures
│   ├── BooleanLogic/            # Boolean logic systems
│   ├── CategoryTheory/          # Category theory
│   ├── ConstructiveSets/        # Constructive set theory
│   ├── Core/                    # Core mathematical definitions
│   ├── Examples/                # Mathematical examples
│   ├── Geometry/                # Geometric reasoning
│   ├── MeasureTheory/           # Measure theory
│   ├── Meta/                    # Metamathematical foundations
│   ├── NumberTheory/            # Number theory
│   ├── Optimization/            # Optimization theory
│   ├── Probability/             # Probability theory
│   ├── Topology/                # Topological structures
│   ├── TypeTheory/              # Type theory
│   ├── arithmopraxis/           # Arithmetic praxis
│   ├── scripts/                 # Mathematical scripts
│   └── tests/                   # Mathematical tests
├── core/                        # Core AGI engine
│   ├── logos_core/              # Main reasoning engine
│   │   ├── coherence/           # Trinity-coherence computation
│   │   ├── governance/          # Cryptographic governance
│   │   ├── meta_reasoning/      # Meta-reasoning capabilities
│   │   └── runtime/             # Runtime services
│   ├── adaptive_reasoning/      # Adaptive reasoning systems
│   ├── distributed/             # Distributed computing
│   ├── language/                # Language processing
│   ├── learning/                # Machine learning components
│   ├── logos_alignment_core/    # Alignment core systems
│   ├── proof_gating/            # Proof validation
│   ├── pxl_kernel/              # PXL kernel implementation
│   └── runtime_services/        # Runtime service management
├── services/                    # Production services
│   ├── gui/                     # GUI interfaces
│   ├── gui_interfaces/          # GUI interface components
│   ├── persistence/             # Data persistence
│   ├── policies/                # Policy management
│   └── workers/                 # Worker services
├── subsystems/                  # Specialized subsystems
│   ├── telos/                   # Purpose/goal reasoning
│   ├── tetragnos/               # Language and translation
│   └── thonoc/                  # Bayesian reasoning
├── safety/                      # Safety frameworks
│   ├── eschaton_framework/      # Eschatological safety → **RENAMED TO ultima_framework/**
│   ├── integrity_framework/     # Integrity safeguards
│   └── obdc/                    # Operational safety
├── reasoning_engines/           # Specialized reasoning engines
├── external_libraries/          # Vendored dependencies
├── testing/                     # Test suites
├── deployment/                  # Deployment configurations
├── boot/                        # Bootstrap systems
├── audit/                       # Audit and compliance
├── coq/                         # Coq formal verification
├── examples/                    # Usage examples
├── keys/                        # Cryptographic keys
├── logs/                        # Application logs
├── state/                       # System state
└── data/                        # Data storage
```

## File Placement Rules

### Automatic Classification
When adding new files, classify them according to these rules:

1. **Production Code** → LOGOS_V2
   - Core algorithms and data structures
   - API endpoints and services
   - Configuration files for production
   - Documentation for users/developers
   - Test files for production validation

2. **Legacy Code** → LOGOS_V1
   - Any file from original LOGOS_AGI
   - Deprecated functionality
   - Historical implementations
   - Archive materials

3. **Development Files** → Project_Notes
   - TODO lists and planning documents
   - Debug output and crash dumps
   - Scratch code and experiments
   - Meeting notes and research
   - Temporary files and backups

### File Type Guidelines

| File Type | Primary Location | Secondary Location | Never In |
|-----------|------------------|-------------------|----------|
| `.py` (production) | LOGOS_V2 | - | Project_Notes |
| `.py` (legacy) | LOGOS_V1 | - | - |
| `.py` (experimental) | Project_Notes | - | LOGOS_V2/LOGOS_V1 |
| `.md` (docs) | LOGOS_V2 | Project_Notes | LOGOS_V1 |
| `.md` (notes) | Project_Notes | - | LOGOS_V2/LOGOS_V1 |
| `.json` (config) | LOGOS_V2 | - | Project_Notes |
| `.json` (debug) | Project_Notes | - | LOGOS_V2 |
| `.log` | Project_Notes | LOGOS_V2/logs | - |
| `.txt` (specs) | LOGOS_V2 | - | Project_Notes |
| `.txt` (notes) | Project_Notes | - | LOGOS_V2 |

## Maintenance Procedures

### Regular Cleanup
1. **Monthly Review**: Audit Project_Notes for obsolete files
2. **Quarterly Audit**: Verify LOGOS_V2 contains only production code
3. **Version Updates**: Move deprecated V2 code to V1 before major releases

### Adding New Code
1. **Determine Category**: Classify new work (production/legacy/internal)
2. **Place Appropriately**: Use correct directory structure
3. **Update Documentation**: Maintain this rules document
4. **Test Placement**: Verify imports and functionality work

### Repository Health Checks
Run these commands periodically:

```bash
# Check for misplaced files
find LOGOS_V2 -name "*.log" -o -name "*debug*" -o -name "*todo*"
find LOGOS_V1 -name "*production*" -o -name "*deploy*"
find Project_Notes -name "*core*" -name "*.py"

# Verify clean separation
ls -la LOGOS_V2/ | wc -l  # Should be substantial
ls -la LOGOS_V1/ | wc -l   # Should be legacy-sized
ls -la Project_Notes/ | wc -l  # Should be development-sized
```

## Enforcement
- **Automated Checks**: Implement pre-commit hooks for file placement
- **Code Reviews**: Require reviewer approval for file moves
- **Documentation**: Update this document when structure changes
- **Training**: Ensure all contributors understand these rules

## Contact
For questions about repository organization, consult this document or the project maintainers.