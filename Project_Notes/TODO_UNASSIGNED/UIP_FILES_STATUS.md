# UIP PROTOCOL FILES - MISSING & EXISTING STATUS

## STEP 0 - Preprocessing & Ingress Routing
- [ ] input_sanitizer.py → interfaces/natural_language/ (MISSING)
- [ ] session_initializer.py → protocols/user_interaction/ (MISSING)  
- [ ] progressive_router.py → protocols/user_interaction/ (MISSING)

## STEP 1 - Linguistic Analysis
- [ ] linguistic_analysis.py → protocols/user_interaction/ (MISSING)
- [ ] intent_classifier.py → interfaces/natural_language/ (MISSING)
- [ ] entity_extractor.py → interfaces/natural_language/ (MISSING) 
- [ ] nlp_processor.py → interfaces/natural_language/ (MISSING)
- [ ] semantic_inferencer.py → intelligence/reasoning_engines/ (MISSING)
- [ ] semantic_extractor.py → intelligence/reasoning_engines/ (MISSING)
- [ ] nli_checker.py → intelligence/reasoning_engines/ (MISSING)
- [ ] clarification_orchestrator.py → protocols/user_interaction/ (MISSING)
- [ ] bayesian_resolver.py → protocols/user_interaction/ (MISSING)
- [ ] networks_projector.py → intelligence/trinity/ (MISSING)
- [ ] c_value_estimator.py → intelligence/trinity/ (MISSING)
- [ ] thonoc_orbital_analysis.py → intelligence/trinity/thonoc/ (MISSING)

## STEP 2 - PXL Compliance & Validation
- [ ] pxl_compliance_gate.py → protocols/user_interaction/ (MISSING)
- [x] integrity_safeguard.py → safety/compliance/ (EXISTS - need to verify location)
- [ ] ethical_enforcement.py → safety/compliance/ (MISSING)
- [ ] proof_validator.py → safety/authorization/ (MISSING)
- [ ] authorization_gate.py → safety/authorization/ (MISSING)
- [x] coq_proofs/ → mathematics/formal_verification/ (EXISTS - in coq/)
- [ ] theorem_prover.py → mathematics/formal_verification/ (MISSING)

## STEP 3 - IEL Overlay Analysis  
- [ ] iel_overlay_analysis.py → protocols/user_interaction/ (MISSING)
- [x] iel_registry.py → intelligence/iel_domains/ (EXISTS - check location)
- [x] chrono_praxis/, axio_praxis/, etc. → intelligence/iel_domains/ (MOVED from IEL/)

## STEP 4 - Trinity Invocation
- [ ] trinity_coordinator.py → intelligence/trinity/ (MISSING)
- [x] thonoc/, telos/, tetragnos/ → intelligence/trinity/ (MOVED from subsystems/)

## STEP 5 - Adaptive Inference
- [x] autonomous_learning.py → intelligence/adaptive/ (EXISTS - check core/learning/)
- [ ] self_improvement.py → intelligence/adaptive/ (MISSING)
- [x] bayesian_inference.py → intelligence/reasoning_engines/ (MOVED from reasoning_engines/)
- [x] semantic_transformers.py → intelligence/reasoning_engines/ (MOVED)

## STEP 6 - Resolution & Response Synthesis
- [ ] response_formatter.py → protocols/user_interaction/ (MISSING)
- [x] unified_formalisms.py → intelligence/reasoning_engines/ (EXISTS - check location)
- [x] temporal_predictor.py → intelligence/reasoning_engines/ (MOVED)

## STEP 7 - Compliance Recheck & Audit Stamp
- [ ] compliance_monitor.py → safety/audit/ (MISSING)
- [ ] audit_logger.py → protocols/shared/ (MISSING) 
- [ ] policy_verifier.py → validation/compliance/ (MISSING)
- [ ] audit_validator.py → validation/compliance/ (MISSING)

## STEP 8 - Egress & Delivery
- [ ] response_dispatcher.py → protocols/user_interaction/ (MISSING)
- [x] state_manager.py → persistence/storage/ (EXISTS - check state/)
- [ ] result_cache.py → persistence/cache/ (MISSING)
- [ ] message_formats.py → protocols/shared/ (MISSING)

## REGISTRY & COORDINATION
- [ ] uip_registry.py → protocols/user_interaction/ (MISSING - suggested addition)

## EXISTING FILES NOT IN UIP LIST (Need classification)
- core/ directory contents (various modules)
- services/ remaining contents  
- boot/ directory
- audit/ directory
- deployment/ directory
- tools/ directory
- external_libraries/
- keys/
- logs/
- data/
- dumps/
- examples/

TOTAL MISSING FILES: ~25 core UIP protocol files
TOTAL FILES TO CLASSIFY: ~50+ existing files in other directories