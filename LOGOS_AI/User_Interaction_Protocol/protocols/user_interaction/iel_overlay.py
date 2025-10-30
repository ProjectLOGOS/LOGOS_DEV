"""
IEL Overlay - UIP Step 3
=========================

Applies Integrated Epistemic Logic (IEL) frameworks for enhanced reasoning.
Overlays domain-specific logical structures onto validated requests.
Provides modal, temporal, and multi-dimensional logical analysis.
"""

from protocols.shared.system_imports import *
from protocols.user_interaction.uip_registry import register_uip_handler, UIPStep, UIPContext


@dataclass
class IELOverlayResult:
    """Result structure for IEL overlay application"""
    overlay_applied: bool
    selected_domains: List[str]  # IEL domains used
    modal_analysis: Dict[str, Any]  # Modal logic results
    temporal_analysis: Dict[str, Any]  # Temporal logic results  
    epistemic_analysis: Dict[str, Any]  # Knowledge/belief analysis
    deontic_analysis: Dict[str, Any]  # Normative/obligation analysis
    logical_coherence_score: float  # Overall logical consistency
    reasoning_pathways: List[Dict[str, Any]]  # Available reasoning approaches
    domain_compatibility: Dict[str, float]  # Domain fitness scores
    overlay_metadata: Dict[str, Any]


class IELOverlayEngine:
    """Core IEL overlay application engine"""
    
    def __init__(self):
        """Initialize IEL overlay engine"""
        self.logger = logging.getLogger(__name__)
        self.available_domains = None
        self.modal_analyzers = {}
        self.temporal_analyzers = {}
        self.epistemic_analyzers = {}
        self.deontic_analyzers = {}
        self.initialize_iel_systems()
    
    def initialize_iel_systems(self):
        """Initialize IEL domain systems"""
        try:
            # These will import actual IEL implementations once they exist
            # from intelligence.iel_domains.modal_praxis import ModalPraxisAnalyzer
            # from intelligence.iel_domains.chrono_praxis import ChronoPraxisAnalyzer
            # from intelligence.iel_domains.gnosi_praxis import GnosiPraxisAnalyzer
            # from intelligence.iel_domains.themi_praxis import ThemiPraxisAnalyzer
            
            # Initialize available IEL domains
            self.available_domains = self._load_available_domains()
            
            # Initialize placeholder analyzers
            self.modal_analyzers = self._initialize_modal_analyzers()
            self.temporal_analyzers = self._initialize_temporal_analyzers()
            self.epistemic_analyzers = self._initialize_epistemic_analyzers()
            self.deontic_analyzers = self._initialize_deontic_analyzers()
            
            self.logger.info(f"IEL overlay engine initialized with {len(self.available_domains)} domains")
            
        except ImportError as e:
            self.logger.warning(f"Some IEL components not available: {e}")
    
    def _load_available_domains(self) -> Dict[str, Dict[str, Any]]:
        """Load available IEL domain definitions"""
        return {
            'modal_praxis': {
                'name': 'Modal Praxis',
                'description': 'Possibility, necessity, and modal reasoning',
                'logic_types': ['modal_logic', 'possible_worlds'],
                'applicable_intents': ['analysis_task', 'problem_solving', 'question'],
                'strength_domains': ['possibility_analysis', 'counterfactual_reasoning'],
                'coq_modules': ['modules/IEL/ModalPraxis/']
            },
            'chrono_praxis': {
                'name': 'Chrono Praxis', 
                'description': 'Temporal logic and time-based reasoning',
                'logic_types': ['temporal_logic', 'linear_time', 'branching_time'],
                'applicable_intents': ['analysis_task', 'information_seeking'],
                'strength_domains': ['temporal_analysis', 'causal_reasoning', 'sequence_analysis'],
                'coq_modules': ['modules/IEL/ChronoPraxis/']
            },
            'gnosi_praxis': {
                'name': 'Gnosi Praxis',
                'description': 'Epistemic logic and knowledge reasoning', 
                'logic_types': ['epistemic_logic', 'knowledge_bases', 'belief_revision'],
                'applicable_intents': ['information_seeking', 'question', 'analysis_task'],
                'strength_domains': ['knowledge_analysis', 'belief_reasoning', 'information_processing'],
                'coq_modules': ['modules/IEL/GnosiPraxis/']
            },
            'themi_praxis': {
                'name': 'Themi Praxis',
                'description': 'Deontic logic and normative reasoning',
                'logic_types': ['deontic_logic', 'obligation_systems', 'normative_reasoning'],
                'applicable_intents': ['request', 'command', 'analysis_task'],
                'strength_domains': ['ethical_reasoning', 'obligation_analysis', 'normative_compliance'],
                'coq_modules': ['modules/IEL/ThemiPraxis/']
            },
            'dyna_praxis': {
                'name': 'Dyna Praxis',
                'description': 'Dynamic logic and action reasoning',
                'logic_types': ['dynamic_logic', 'action_logic', 'state_transitions'],
                'applicable_intents': ['command', 'request', 'problem_solving'],
                'strength_domains': ['action_analysis', 'process_reasoning', 'workflow_logic'],
                'coq_modules': ['modules/IEL/DynaPraxis/']
            },
            'hexi_praxis': {
                'name': 'Hexi Praxis',
                'description': 'Dispositional logic and capacity reasoning',
                'logic_types': ['dispositional_logic', 'capacity_analysis'],
                'applicable_intents': ['analysis_task', 'question'],
                'strength_domains': ['capability_analysis', 'potential_reasoning'],
                'coq_modules': ['modules/IEL/HexiPraxis/']
            },
            'chrema_praxis': {
                'name': 'Chrema Praxis',
                'description': 'Resource logic and utilization reasoning',
                'logic_types': ['resource_logic', 'linear_logic'],
                'applicable_intents': ['analysis_task', 'problem_solving'],
                'strength_domains': ['resource_analysis', 'efficiency_reasoning'],
                'coq_modules': ['modules/IEL/ChremaPraxis/']
            },
            'mu_praxis': {
                'name': 'Mu Praxis',
                'description': 'Fixed-point logic and recursive reasoning',
                'logic_types': ['mu_calculus', 'fixed_point_logic'],
                'applicable_intents': ['analysis_task', 'problem_solving'],
                'strength_domains': ['recursive_analysis', 'invariant_reasoning'],
                'coq_modules': ['modules/IEL/MuPraxis/']
            },
            'tyche_praxis': {
                'name': 'Tyche Praxis',
                'description': 'Probabilistic logic and uncertainty reasoning',
                'logic_types': ['probabilistic_logic', 'uncertainty_reasoning'],
                'applicable_intents': ['analysis_task', 'question'],
                'strength_domains': ['probability_analysis', 'uncertainty_handling'],
                'coq_modules': ['modules/IEL/TychePraxis/']
            }
        }
    
    def _initialize_modal_analyzers(self) -> Dict[str, Any]:
        """Initialize modal logic analyzers"""
        return {
            'possibility_analyzer': {
                'description': 'Analyzes possibility and necessity claims',
                'operators': ['◇', '□', '→', '↔'],  # Diamond, box, implication, biconditional
                'methods': ['possible_worlds', 'accessibility_relations', 'modal_equivalence']
            },
            'counterfactual_analyzer': {
                'description': 'Analyzes counterfactual reasoning',
                'operators': ['⊃>', '□→', '◇→'],  # Counterfactual conditional variants
                'methods': ['closest_worlds', 'similarity_metrics', 'causal_chains']
            }
        }
    
    def _initialize_temporal_analyzers(self) -> Dict[str, Any]:
        """Initialize temporal logic analyzers"""
        return {
            'linear_time_analyzer': {
                'description': 'Linear temporal logic analysis',
                'operators': ['○', '□', '◇', 'U', 'R'],  # Next, always, eventually, until, release
                'methods': ['ltl_model_checking', 'temporal_sequences', 'causality_analysis']
            },
            'branching_time_analyzer': {
                'description': 'Branching temporal logic analysis',
                'operators': ['AX', 'EX', 'AG', 'EG', 'AF', 'EF', 'AU', 'EU'],
                'methods': ['ctl_model_checking', 'computation_trees', 'path_quantification']
            }
        }
    
    def _initialize_epistemic_analyzers(self) -> Dict[str, Any]:
        """Initialize epistemic logic analyzers"""
        return {
            'knowledge_analyzer': {
                'description': 'Knowledge and belief analysis',
                'operators': ['K', 'B', 'C', 'D'],  # Knowledge, belief, common knowledge, distributed knowledge
                'methods': ['kripke_models', 'epistemic_accessibility', 'knowledge_update']
            },
            'belief_revision_analyzer': {
                'description': 'Belief revision and update analysis',
                'operators': ['*', '+', '÷'],  # Revision, expansion, contraction
                'methods': ['agm_postulates', 'prioritized_revision', 'iterated_revision']
            }
        }
    
    def _initialize_deontic_analyzers(self) -> Dict[str, Any]:
        """Initialize deontic logic analyzers"""
        return {
            'obligation_analyzer': {
                'description': 'Obligation and permission analysis',
                'operators': ['O', 'P', 'F'],  # Obligation, permission, forbidden
                'methods': ['deontic_accessibility', 'normative_systems', 'conflict_resolution']
            },
            'normative_analyzer': {
                'description': 'Normative reasoning analysis',
                'operators': ['⊕', '⊖', '⊗'],  # Normative operators
                'methods': ['norm_prioritization', 'exception_handling', 'default_reasoning']
            }
        }
    
    async def apply_iel_overlay(self, context: UIPContext) -> IELOverlayResult:
        """
        Apply appropriate IEL overlays based on linguistic analysis and compliance results
        
        Args:
            context: UIP processing context with previous step results
            
        Returns:
            IELOverlayResult: Detailed IEL overlay analysis
        """
        self.logger.info(f"Starting IEL overlay application for {context.correlation_id}")
        
        try:
            # Extract information from previous steps
            linguistic_data = context.step_results.get('linguistic_analysis_complete', {})
            compliance_data = context.step_results.get('pxl_compliance_gate_complete', {})
            
            intent_classification = linguistic_data.get('intent_classification', {})
            entities = linguistic_data.get('entity_extraction', [])
            semantic_repr = linguistic_data.get('semantic_representation', {})
            permitted_operations = compliance_data.get('permitted_operations', [])
            
            # Select appropriate IEL domains
            selected_domains = await self._select_iel_domains(
                intent_classification, entities, semantic_repr, permitted_operations
            )
            
            # Apply domain-specific analysis
            modal_results = await self._apply_modal_analysis(
                context.user_input, selected_domains, semantic_repr
            )
            
            temporal_results = await self._apply_temporal_analysis(
                context.user_input, selected_domains, semantic_repr
            )
            
            epistemic_results = await self._apply_epistemic_analysis(
                context.user_input, selected_domains, semantic_repr
            )
            
            deontic_results = await self._apply_deontic_analysis(
                context.user_input, selected_domains, semantic_repr
            )
            
            # Calculate logical coherence
            coherence_score = self._calculate_logical_coherence(
                modal_results, temporal_results, epistemic_results, deontic_results
            )
            
            # Identify reasoning pathways
            reasoning_pathways = self._identify_reasoning_pathways(
                selected_domains, modal_results, temporal_results, 
                epistemic_results, deontic_results
            )
            
            # Calculate domain compatibility scores
            domain_compatibility = self._calculate_domain_compatibility(
                intent_classification, selected_domains
            )
            
            result = IELOverlayResult(
                overlay_applied=len(selected_domains) > 0,
                selected_domains=selected_domains,
                modal_analysis=modal_results,
                temporal_analysis=temporal_results,
                epistemic_analysis=epistemic_results,
                deontic_analysis=deontic_results,
                logical_coherence_score=coherence_score,
                reasoning_pathways=reasoning_pathways,
                domain_compatibility=domain_compatibility,
                overlay_metadata={
                    'overlay_timestamp': time.time(),
                    'iel_version': '2.0.0',
                    'domains_evaluated': len(self.available_domains),
                    'domains_selected': len(selected_domains),
                    'analysis_depth': self._calculate_analysis_depth(selected_domains)
                }
            )
            
            self.logger.info(
                f"IEL overlay application completed for {context.correlation_id} "
                f"(domains: {len(selected_domains)}, coherence: {coherence_score:.3f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"IEL overlay application failed for {context.correlation_id}: {e}")
            raise
    
    async def _select_iel_domains(
        self,
        intent_classification: Dict[str, float],
        entities: List[Dict[str, Any]],
        semantic_repr: Dict[str, Any],
        permitted_operations: List[str]
    ) -> List[str]:
        """Select appropriate IEL domains for the request"""
        
        selected = []
        primary_intent = max(intent_classification.items(), key=lambda x: x[1])[0] if intent_classification else 'unknown'
        
        # Select domains based on intent patterns
        for domain_key, domain_info in self.available_domains.items():
            compatibility_score = 0.0
            
            # Intent compatibility
            if primary_intent in domain_info['applicable_intents']:
                compatibility_score += 0.4
            
            # Check for domain-specific keywords in user input or entities
            domain_keywords = self._get_domain_keywords(domain_key)
            text_to_check = str(entities) + str(semantic_repr)
            
            keyword_matches = sum(1 for keyword in domain_keywords if keyword in text_to_check.lower())
            compatibility_score += (keyword_matches / max(len(domain_keywords), 1)) * 0.3
            
            # Operational compatibility
            domain_operations = self._get_domain_operations(domain_key)
            operation_overlap = len(set(domain_operations) & set(permitted_operations))
            if domain_operations:
                compatibility_score += (operation_overlap / len(domain_operations)) * 0.3
            
            # Select if compatibility is sufficient
            if compatibility_score >= 0.3:
                selected.append(domain_key)
        
        # Ensure at least one domain is selected for analysis
        if not selected and self.available_domains:
            # Default to modal_praxis for general reasoning
            selected = ['modal_praxis']
        
        # Limit to top 3 domains to avoid complexity
        return selected[:3]
    
    def _get_domain_keywords(self, domain_key: str) -> List[str]:
        """Get keywords associated with a specific IEL domain"""
        keyword_map = {
            'modal_praxis': ['possible', 'necessary', 'might', 'must', 'could', 'should', 'maybe'],
            'chrono_praxis': ['time', 'when', 'before', 'after', 'sequence', 'temporal', 'duration'],
            'gnosi_praxis': ['know', 'believe', 'think', 'knowledge', 'information', 'understand'],
            'themi_praxis': ['should', 'ought', 'obligation', 'duty', 'permission', 'allowed', 'forbidden'],
            'dyna_praxis': ['action', 'do', 'perform', 'execute', 'process', 'workflow', 'steps'],
            'hexi_praxis': ['capable', 'ability', 'capacity', 'potential', 'disposition', 'can'],
            'chrema_praxis': ['resource', 'cost', 'efficiency', 'optimization', 'allocation'],
            'mu_praxis': ['recursive', 'iteration', 'loop', 'fixed point', 'invariant'],
            'tyche_praxis': ['probability', 'chance', 'uncertain', 'random', 'likely', 'risk']
        }
        return keyword_map.get(domain_key, [])
    
    def _get_domain_operations(self, domain_key: str) -> List[str]:
        """Get operations supported by a specific IEL domain"""
        operation_map = {
            'modal_praxis': ['possibility_analysis', 'necessity_analysis', 'counterfactual_reasoning'],
            'chrono_praxis': ['temporal_analysis', 'causal_reasoning', 'sequence_analysis'],
            'gnosi_praxis': ['knowledge_analysis', 'information_processing', 'belief_reasoning'],
            'themi_praxis': ['ethical_reasoning', 'normative_analysis', 'obligation_analysis'],
            'dyna_praxis': ['action_analysis', 'process_reasoning', 'workflow_analysis'],
            'hexi_praxis': ['capability_analysis', 'potential_assessment'],
            'chrema_praxis': ['resource_analysis', 'efficiency_analysis'],
            'mu_praxis': ['recursive_analysis', 'fixed_point_analysis'],
            'tyche_praxis': ['probability_analysis', 'uncertainty_analysis']
        }
        return operation_map.get(domain_key, [])
    
    async def _apply_modal_analysis(
        self, 
        user_input: str, 
        selected_domains: List[str],
        semantic_repr: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply modal logic analysis"""
        
        if 'modal_praxis' not in selected_domains:
            return {'applied': False, 'reason': 'Modal praxis not selected'}
        
        # Analyze modal operators and constructions
        modal_operators = self._detect_modal_operators(user_input)
        possibility_claims = self._extract_possibility_claims(user_input)
        necessity_claims = self._extract_necessity_claims(user_input)
        
        return {
            'applied': True,
            'modal_operators_detected': modal_operators,
            'possibility_claims': possibility_claims,
            'necessity_claims': necessity_claims,
            'modal_logic_applicable': len(modal_operators) > 0 or len(possibility_claims) > 0,
            'recommended_analysis': 'possible_worlds_semantics' if possibility_claims else 'modal_equivalence'
        }
    
    async def _apply_temporal_analysis(
        self,
        user_input: str,
        selected_domains: List[str], 
        semantic_repr: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply temporal logic analysis"""
        
        if 'chrono_praxis' not in selected_domains:
            return {'applied': False, 'reason': 'Chrono praxis not selected'}
        
        # Analyze temporal constructions
        temporal_markers = self._detect_temporal_markers(user_input)
        causal_chains = self._extract_causal_chains(user_input)
        temporal_sequences = self._extract_temporal_sequences(user_input)
        
        return {
            'applied': True,
            'temporal_markers_detected': temporal_markers,
            'causal_chains': causal_chains,
            'temporal_sequences': temporal_sequences,
            'temporal_logic_applicable': len(temporal_markers) > 0 or len(causal_chains) > 0,
            'recommended_analysis': 'linear_time_logic' if temporal_sequences else 'branching_time_logic'
        }
    
    async def _apply_epistemic_analysis(
        self,
        user_input: str,
        selected_domains: List[str],
        semantic_repr: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply epistemic logic analysis"""
        
        if 'gnosi_praxis' not in selected_domains:
            return {'applied': False, 'reason': 'Gnosi praxis not selected'}
        
        # Analyze epistemic constructions
        knowledge_claims = self._extract_knowledge_claims(user_input)
        belief_statements = self._extract_belief_statements(user_input)
        uncertainty_expressions = self._extract_uncertainty_expressions(user_input)
        
        return {
            'applied': True,
            'knowledge_claims': knowledge_claims,
            'belief_statements': belief_statements,
            'uncertainty_expressions': uncertainty_expressions,
            'epistemic_logic_applicable': len(knowledge_claims) > 0 or len(belief_statements) > 0,
            'recommended_analysis': 'knowledge_update' if knowledge_claims else 'belief_revision'
        }
    
    async def _apply_deontic_analysis(
        self,
        user_input: str,
        selected_domains: List[str],
        semantic_repr: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply deontic logic analysis"""
        
        if 'themi_praxis' not in selected_domains:
            return {'applied': False, 'reason': 'Themi praxis not selected'}
        
        # Analyze deontic constructions
        obligations = self._extract_obligations(user_input)
        permissions = self._extract_permissions(user_input)
        prohibitions = self._extract_prohibitions(user_input)
        
        return {
            'applied': True,
            'obligations': obligations,
            'permissions': permissions,
            'prohibitions': prohibitions,
            'deontic_logic_applicable': len(obligations) > 0 or len(permissions) > 0,
            'recommended_analysis': 'obligation_systems' if obligations else 'permission_systems'
        }
    
    def _detect_modal_operators(self, text: str) -> List[str]:
        """Detect modal operators and constructions in text"""
        modal_patterns = [
            'possible', 'possibly', 'might', 'may', 'could',
            'necessary', 'necessarily', 'must', 'has to', 'needs to'
        ]
        return [pattern for pattern in modal_patterns if pattern in text.lower()]
    
    def _extract_possibility_claims(self, text: str) -> List[str]:
        """Extract possibility claims from text"""
        # Simple extraction - would be more sophisticated in real implementation
        sentences = text.split('.')
        possibility_sentences = [
            s.strip() for s in sentences 
            if any(word in s.lower() for word in ['possible', 'might', 'could', 'may'])
        ]
        return possibility_sentences
    
    def _extract_necessity_claims(self, text: str) -> List[str]:
        """Extract necessity claims from text"""
        sentences = text.split('.')
        necessity_sentences = [
            s.strip() for s in sentences
            if any(word in s.lower() for word in ['must', 'necessary', 'has to', 'needs to'])
        ]
        return necessity_sentences
    
    def _detect_temporal_markers(self, text: str) -> List[str]:
        """Detect temporal markers in text"""
        temporal_patterns = [
            'when', 'before', 'after', 'during', 'while', 'since', 'until',
            'first', 'then', 'next', 'finally', 'simultaneously'
        ]
        return [pattern for pattern in temporal_patterns if pattern in text.lower()]
    
    def _extract_causal_chains(self, text: str) -> List[Dict[str, str]]:
        """Extract causal chains from text"""
        # Placeholder implementation
        causal_markers = ['because', 'causes', 'leads to', 'results in', 'due to']
        chains = []
        for marker in causal_markers:
            if marker in text.lower():
                # Would extract actual cause-effect relationships
                chains.append({'marker': marker, 'context': 'placeholder'})
        return chains
    
    def _extract_temporal_sequences(self, text: str) -> List[str]:
        """Extract temporal sequences from text"""
        # Placeholder implementation
        sequence_markers = ['first', 'second', 'then', 'next', 'finally']
        sequences = [marker for marker in sequence_markers if marker in text.lower()]
        return sequences
    
    def _extract_knowledge_claims(self, text: str) -> List[str]:
        """Extract knowledge claims from text"""
        knowledge_patterns = ['know that', 'aware that', 'certain that', 'sure that']
        sentences = text.split('.')
        return [
            s.strip() for s in sentences
            if any(pattern in s.lower() for pattern in knowledge_patterns)
        ]
    
    def _extract_belief_statements(self, text: str) -> List[str]:
        """Extract belief statements from text"""
        belief_patterns = ['believe', 'think', 'suppose', 'assume', 'expect']
        sentences = text.split('.')
        return [
            s.strip() for s in sentences
            if any(pattern in s.lower() for pattern in belief_patterns)
        ]
    
    def _extract_uncertainty_expressions(self, text: str) -> List[str]:
        """Extract uncertainty expressions from text"""
        uncertainty_patterns = ['uncertain', 'not sure', 'maybe', 'perhaps', 'unclear']
        return [pattern for pattern in uncertainty_patterns if pattern in text.lower()]
    
    def _extract_obligations(self, text: str) -> List[str]:
        """Extract obligation statements from text"""
        obligation_patterns = ['should', 'ought to', 'must', 'have to', 'required to']
        sentences = text.split('.')
        return [
            s.strip() for s in sentences
            if any(pattern in s.lower() for pattern in obligation_patterns)
        ]
    
    def _extract_permissions(self, text: str) -> List[str]:
        """Extract permission statements from text"""
        permission_patterns = ['allowed to', 'permitted to', 'can', 'may', 'authorized to']
        sentences = text.split('.')
        return [
            s.strip() for s in sentences
            if any(pattern in s.lower() for pattern in permission_patterns)
        ]
    
    def _extract_prohibitions(self, text: str) -> List[str]:
        """Extract prohibition statements from text"""
        prohibition_patterns = ['forbidden', 'not allowed', 'cannot', 'must not', 'prohibited']
        sentences = text.split('.')
        return [
            s.strip() for s in sentences
            if any(pattern in s.lower() for pattern in prohibition_patterns)
        ]
    
    def _calculate_logical_coherence(
        self,
        modal_results: Dict[str, Any],
        temporal_results: Dict[str, Any],
        epistemic_results: Dict[str, Any],
        deontic_results: Dict[str, Any]
    ) -> float:
        """Calculate overall logical coherence score"""
        
        coherence_score = 1.0
        
        # Check for logical contradictions between domains
        # This is a simplified check - real implementation would be more sophisticated
        
        # Modal-temporal coherence
        if (modal_results.get('applied') and temporal_results.get('applied')):
            # Check for temporal-modal consistency
            coherence_score *= 0.95  # Slight penalty for complexity
        
        # Epistemic-deontic coherence
        if (epistemic_results.get('applied') and deontic_results.get('applied')):
            knowledge_claims = epistemic_results.get('knowledge_claims', [])
            obligations = deontic_results.get('obligations', [])
            
            # Simple consistency check
            if knowledge_claims and obligations:
                coherence_score *= 0.9  # Penalty for potential knowledge-obligation conflicts
        
        return max(0.0, min(1.0, coherence_score))
    
    def _identify_reasoning_pathways(
        self,
        selected_domains: List[str],
        modal_results: Dict[str, Any],
        temporal_results: Dict[str, Any],
        epistemic_results: Dict[str, Any],
        deontic_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify available reasoning pathways"""
        
        pathways = []
        
        for domain in selected_domains:
            domain_info = self.available_domains.get(domain, {})
            pathway = {
                'domain': domain,
                'name': domain_info.get('name', domain),
                'description': domain_info.get('description', ''),
                'logic_types': domain_info.get('logic_types', []),
                'applicable_methods': []
            }
            
            # Add domain-specific methods based on analysis results
            if domain == 'modal_praxis' and modal_results.get('applied'):
                pathway['applicable_methods'].extend(['possible_worlds_analysis', 'modal_equivalence_checking'])
            elif domain == 'chrono_praxis' and temporal_results.get('applied'):
                pathway['applicable_methods'].extend(['temporal_sequence_analysis', 'causal_chain_analysis'])
            elif domain == 'gnosi_praxis' and epistemic_results.get('applied'):
                pathway['applicable_methods'].extend(['knowledge_base_reasoning', 'belief_update'])
            elif domain == 'themi_praxis' and deontic_results.get('applied'):
                pathway['applicable_methods'].extend(['obligation_analysis', 'normative_reasoning'])
            
            pathways.append(pathway)
        
        return pathways
    
    def _calculate_domain_compatibility(
        self,
        intent_classification: Dict[str, float],
        selected_domains: List[str]
    ) -> Dict[str, float]:
        """Calculate compatibility scores for each selected domain"""
        
        compatibility = {}
        primary_intent = max(intent_classification.items(), key=lambda x: x[1])[0] if intent_classification else 'unknown'
        
        for domain in selected_domains:
            domain_info = self.available_domains.get(domain, {})
            applicable_intents = domain_info.get('applicable_intents', [])
            
            if primary_intent in applicable_intents:
                base_score = 0.8
            else:
                base_score = 0.4
            
            # Adjust based on domain specialization
            strength_domains = domain_info.get('strength_domains', [])
            if strength_domains:
                base_score += 0.1  # Bonus for specialized domains
            
            compatibility[domain] = min(1.0, base_score)
        
        return compatibility
    
    def _calculate_analysis_depth(self, selected_domains: List[str]) -> str:
        """Calculate the depth of IEL analysis"""
        if len(selected_domains) == 0:
            return 'none'
        elif len(selected_domains) == 1:
            return 'basic'
        elif len(selected_domains) == 2:
            return 'intermediate'
        else:
            return 'comprehensive'


# Global IEL overlay engine instance
iel_overlay_engine = IELOverlayEngine()


@register_uip_handler(UIPStep.STEP_3_IEL_OVERLAY, dependencies=[UIPStep.STEP_2_PXL_COMPLIANCE], timeout=45)
async def handle_iel_overlay(context: UIPContext) -> Dict[str, Any]:
    """
    UIP Step 3: IEL Overlay Handler
    
    Applies Integrated Epistemic Logic frameworks to enhance reasoning
    capabilities for the validated user request.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Apply IEL overlay analysis
        overlay_result = await iel_overlay_engine.apply_iel_overlay(context)
        
        # Convert result to dictionary for context storage
        result_dict = {
            'step': 'iel_overlay_complete',
            'overlay_applied': overlay_result.overlay_applied,
            'selected_domains': overlay_result.selected_domains,
            'modal_analysis': overlay_result.modal_analysis,
            'temporal_analysis': overlay_result.temporal_analysis,
            'epistemic_analysis': overlay_result.epistemic_analysis,
            'deontic_analysis': overlay_result.deontic_analysis,
            'logical_coherence_score': overlay_result.logical_coherence_score,
            'reasoning_pathways': overlay_result.reasoning_pathways,
            'domain_compatibility': overlay_result.domain_compatibility,
            'overlay_metadata': overlay_result.overlay_metadata
        }
        
        # Add overlay insights
        insights = []
        if overlay_result.overlay_applied:
            insights.append(f"Applied {len(overlay_result.selected_domains)} IEL domain(s) for enhanced reasoning.")
            if overlay_result.logical_coherence_score >= 0.8:
                insights.append("High logical coherence detected across applied domains.")
            if overlay_result.reasoning_pathways:
                insights.append(f"Identified {len(overlay_result.reasoning_pathways)} reasoning pathway(s).")
        else:
            insights.append("No IEL overlay applied - proceeding with standard processing.")
        
        result_dict['overlay_insights'] = insights
        
        logger.info(
            f"IEL overlay application completed for {context.correlation_id} "
            f"(applied: {overlay_result.overlay_applied}, "
            f"domains: {len(overlay_result.selected_domains)}, "
            f"coherence: {overlay_result.logical_coherence_score:.3f})"
        )
        
        return result_dict
        
    except Exception as e:
        logger.error(f"IEL overlay application failed for {context.correlation_id}: {e}")
        raise


__all__ = [
    'IELOverlayResult',
    'IELOverlayEngine',
    'iel_overlay_engine', 
    'handle_iel_overlay'
]