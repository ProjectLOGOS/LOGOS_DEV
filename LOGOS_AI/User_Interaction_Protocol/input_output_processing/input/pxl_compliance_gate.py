"""
PXL Compliance Gate - UIP Step 2
=================================

Validates requests against PXL operational constraints.
Ensures user input complies with system capabilities and ethical boundaries.
Gates non-compliant requests with appropriate feedback.
"""

from system_utillities.shared.system_imports import *
from system_utillities.user_interaction.uip_registry import (
    UIPContext,
    UIPStep,
    register_uip_handler,
)


@dataclass
class PXLComplianceResult:
    """Result structure for PXL compliance check"""

    is_compliant: bool
    compliance_score: float  # 0.0 - 1.0
    violations: List[Dict[str, Any]]  # detected violations
    permitted_operations: List[str]  # operations that are allowed
    restricted_operations: List[str]  # operations that are restricted
    ethical_flags: List[str]  # ethical concerns
    safety_flags: List[str]  # safety concerns
    capability_flags: List[str]  # capability limitations
    recommended_alternatives: List[str]  # alternative approaches
    compliance_metadata: Dict[str, Any]


class PXLComplianceEngine:
    """Core PXL compliance validation engine"""

    def __init__(self):
        """Initialize PXL compliance engine"""
        self.logger = logging.getLogger(__name__)
        self.pxl_axioms = None  # Will load from mathematics/pxl/
        self.ethical_constraints = None  # Will load from governance/
        self.safety_constraints = None  # Will load from governance/
        self.capability_constraints = None  # Will load from interfaces/
        self.initialize_constraints()

    def initialize_constraints(self):
        """Initialize PXL constraint systems"""
        try:
            # These will import actual constraint definitions once they exist
            # from mathematics.pxl.axioms import PXLAxioms
            # from governance.ethical_constraints import EthicalConstraints
            # from governance.safety_constraints import SafetyConstraints
            # from interfaces.capability_constraints import CapabilityConstraints

            # Initialize placeholder constraints
            self.ethical_constraints = self._load_ethical_constraints()
            self.safety_constraints = self._load_safety_constraints()
            self.capability_constraints = self._load_capability_constraints()

            self.logger.info(
                "PXL compliance constraints initialized (placeholder mode)"
            )

        except ImportError as e:
            self.logger.warning(f"Some constraint systems not available: {e}")

    def _load_ethical_constraints(self) -> Dict[str, Any]:
        """Load ethical constraint definitions"""
        return {
            "harm_prevention": {
                "description": "Prevent harm to individuals or groups",
                "violation_patterns": [
                    "violence",
                    "harassment",
                    "discrimination",
                    "manipulation",
                    "privacy_violation",
                    "misinformation",
                    "exploitation",
                ],
                "severity_weights": {
                    "physical_harm": 1.0,
                    "psychological_harm": 0.8,
                    "privacy_harm": 0.7,
                    "social_harm": 0.6,
                },
            },
            "fairness": {
                "description": "Ensure fair and equitable treatment",
                "violation_patterns": [
                    "bias",
                    "unfair_advantage",
                    "discriminatory_treatment",
                    "exclusionary_practices",
                ],
            },
            "transparency": {
                "description": "Maintain appropriate transparency in operations",
                "violation_patterns": [
                    "deceptive_practices",
                    "hidden_capabilities",
                    "misleading_information",
                ],
            },
            "autonomy": {
                "description": "Respect user autonomy and agency",
                "violation_patterns": [
                    "coercion",
                    "manipulation",
                    "autonomy_override",
                    "forced_compliance",
                ],
            },
        }

    def _load_safety_constraints(self) -> Dict[str, Any]:
        """Load safety constraint definitions"""
        return {
            "system_integrity": {
                "description": "Maintain system security and integrity",
                "violation_patterns": [
                    "injection_attack",
                    "privilege_escalation",
                    "resource_exhaustion",
                    "system_compromise",
                    "data_corruption",
                ],
            },
            "operational_safety": {
                "description": "Ensure safe operational parameters",
                "violation_patterns": [
                    "unsafe_execution",
                    "uncontrolled_recursion",
                    "infinite_loops",
                    "memory_exhaustion",
                    "file_system_damage",
                ],
            },
            "information_safety": {
                "description": "Protect sensitive information",
                "violation_patterns": [
                    "credential_exposure",
                    "data_leakage",
                    "unauthorized_access",
                    "sensitive_data_processing",
                ],
            },
        }

    def _load_capability_constraints(self) -> Dict[str, Any]:
        """Load capability constraint definitions"""
        return {
            "computational_limits": {
                "description": "Computational resource limitations",
                "constraints": {
                    "max_processing_time": 300,  # seconds
                    "max_memory_usage": 1024,  # MB
                    "max_file_size": 100,  # MB
                    "max_network_requests": 50,
                },
            },
            "functional_scope": {
                "description": "Available functional capabilities",
                "available_domains": [
                    "natural_language_processing",
                    "mathematical_reasoning",
                    "logical_analysis",
                    "information_retrieval",
                    "creative_generation",
                    "problem_solving",
                ],
                "restricted_domains": [
                    "real_time_control_systems",
                    "financial_transactions",
                    "medical_diagnosis",
                    "legal_advice",
                    "physical_world_interaction",
                ],
            },
            "interface_limits": {
                "description": "Interface and integration limitations",
                "available_interfaces": [
                    "text_input_output",
                    "file_processing",
                    "structured_data_analysis",
                ],
                "restricted_interfaces": [
                    "real_time_audio_video",
                    "hardware_control",
                    "network_administration",
                    "system_administration",
                ],
            },
        }

    async def validate_compliance(self, context: UIPContext) -> PXLComplianceResult:
        """
        Perform comprehensive PXL compliance validation

        Args:
            context: UIP processing context with linguistic analysis results

        Returns:
            PXLComplianceResult: Detailed compliance assessment
        """
        self.logger.info(
            f"Starting PXL compliance validation for {context.correlation_id}"
        )

        try:
            # Extract information from linguistic analysis
            linguistic_data = context.step_results.get(
                "linguistic_analysis_complete", {}
            )
            user_input = context.user_input
            intent_classification = linguistic_data.get("intent_classification", {})
            entities = linguistic_data.get("entity_extraction", [])

            # Perform compliance checks
            ethical_violations = await self._check_ethical_compliance(
                user_input, intent_classification, entities
            )

            safety_violations = await self._check_safety_compliance(
                user_input, intent_classification, entities
            )

            capability_violations = await self._check_capability_compliance(
                user_input, intent_classification, entities
            )

            # Consolidate all violations
            all_violations = (
                ethical_violations + safety_violations + capability_violations
            )

            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(all_violations)

            # Determine overall compliance
            is_compliant = compliance_score >= 0.7 and len(all_violations) == 0

            # Analyze permitted vs restricted operations
            permitted_ops, restricted_ops = self._analyze_operations(
                intent_classification, all_violations
            )

            # Generate recommendations
            alternatives = self._generate_alternatives(
                all_violations, intent_classification
            )

            result = PXLComplianceResult(
                is_compliant=is_compliant,
                compliance_score=compliance_score,
                violations=all_violations,
                permitted_operations=permitted_ops,
                restricted_operations=restricted_ops,
                ethical_flags=[v["type"] for v in ethical_violations],
                safety_flags=[v["type"] for v in safety_violations],
                capability_flags=[v["type"] for v in capability_violations],
                recommended_alternatives=alternatives,
                compliance_metadata={
                    "validation_timestamp": time.time(),
                    "constraint_versions": {
                        "ethical": "1.0.0",
                        "safety": "1.0.0",
                        "capability": "1.0.0",
                    },
                    "total_constraints_checked": len(self.ethical_constraints)
                    + len(self.safety_constraints)
                    + len(self.capability_constraints),
                },
            )

            self.logger.info(
                f"PXL compliance validation completed for {context.correlation_id} "
                f"(compliant: {is_compliant}, score: {compliance_score:.3f})"
            )

            return result

        except Exception as e:
            self.logger.error(
                f"PXL compliance validation failed for {context.correlation_id}: {e}"
            )
            raise

    async def _check_ethical_compliance(
        self,
        user_input: str,
        intent_classification: Dict[str, float],
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Check for ethical constraint violations"""
        violations = []

        lower_input = user_input.lower()

        # Check harm prevention constraints
        harm_patterns = [
            ("violence", ["kill", "murder", "assault", "attack", "hurt", "harm"]),
            ("harassment", ["harass", "bully", "threaten", "intimidate", "stalk"]),
            ("discrimination", ["racist", "sexist", "discriminate", "prejudice"]),
            ("manipulation", ["manipulate", "deceive", "trick", "exploit", "coerce"]),
        ]

        for harm_type, patterns in harm_patterns:
            if any(pattern in lower_input for pattern in patterns):
                violations.append(
                    {
                        "type": f"ethical_harm_{harm_type}",
                        "severity": "high",
                        "description": f"Input contains patterns suggesting {harm_type}",
                        "constraint_category": "harm_prevention",
                        "detected_patterns": [p for p in patterns if p in lower_input],
                    }
                )

        # Check privacy violations
        privacy_patterns = [
            "password",
            "ssn",
            "social security",
            "credit card",
            "private key",
        ]
        if any(pattern in lower_input for pattern in privacy_patterns):
            violations.append(
                {
                    "type": "ethical_privacy_violation",
                    "severity": "medium",
                    "description": "Input may contain sensitive information",
                    "constraint_category": "privacy_protection",
                }
            )

        # Check misinformation patterns
        misinfo_patterns = ["fake news", "conspiracy", "hoax", "deliberately false"]
        if any(pattern in lower_input for pattern in misinfo_patterns):
            violations.append(
                {
                    "type": "ethical_misinformation",
                    "severity": "medium",
                    "description": "Input may request generation of misinformation",
                    "constraint_category": "information_integrity",
                }
            )

        return violations

    async def _check_safety_compliance(
        self,
        user_input: str,
        intent_classification: Dict[str, float],
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Check for safety constraint violations"""
        violations = []

        lower_input = user_input.lower()

        # Check for injection attack patterns
        injection_patterns = [
            "exec(",
            "eval(",
            "import os",
            "subprocess",
            "__import__",
            "<script>",
            "javascript:",
            "onload=",
            "onerror=",
            "drop table",
            "delete from",
            "insert into",
            "update set",
        ]

        if any(pattern in lower_input for pattern in injection_patterns):
            violations.append(
                {
                    "type": "safety_injection_risk",
                    "severity": "critical",
                    "description": "Input contains patterns that may indicate injection attacks",
                    "constraint_category": "system_integrity",
                }
            )

        # Check for resource exhaustion risks
        exhaustion_patterns = ["while true", "infinite loop", "recursive", "fork bomb"]
        if any(pattern in lower_input for pattern in exhaustion_patterns):
            violations.append(
                {
                    "type": "safety_resource_exhaustion",
                    "severity": "high",
                    "description": "Input may request operations that could exhaust system resources",
                    "constraint_category": "operational_safety",
                }
            )

        # Check for file system risks
        filesystem_patterns = ["rm -rf", "delete *", "format drive", "system32"]
        if any(pattern in lower_input for pattern in filesystem_patterns):
            violations.append(
                {
                    "type": "safety_filesystem_risk",
                    "severity": "high",
                    "description": "Input may request dangerous file system operations",
                    "constraint_category": "operational_safety",
                }
            )

        return violations

    async def _check_capability_compliance(
        self,
        user_input: str,
        intent_classification: Dict[str, float],
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Check for capability constraint violations"""
        violations = []

        lower_input = user_input.lower()

        # Check for requests beyond functional scope
        restricted_domain_patterns = [
            ("real_time_control", ["control robot", "move robot", "real time control"]),
            (
                "financial_transactions",
                ["transfer money", "buy stocks", "make payment"],
            ),
            (
                "medical_diagnosis",
                ["diagnose illness", "medical diagnosis", "what disease"],
            ),
            ("legal_advice", ["legal advice", "sue someone", "court case"]),
            ("physical_interaction", ["physical world", "move objects", "robot arm"]),
        ]

        for domain, patterns in restricted_domain_patterns:
            if any(pattern in lower_input for pattern in patterns):
                violations.append(
                    {
                        "type": f"capability_{domain}_restricted",
                        "severity": "medium",
                        "description": f"Request involves restricted domain: {domain}",
                        "constraint_category": "functional_scope",
                    }
                )

        # Check for excessive computational requests
        if len(user_input) > 50000:  # Very long input
            violations.append(
                {
                    "type": "capability_input_size_exceeded",
                    "severity": "low",
                    "description": "Input size exceeds recommended processing limits",
                    "constraint_category": "computational_limits",
                }
            )

        # Check for real-time processing requests
        realtime_patterns = [
            "real time",
            "live stream",
            "continuous monitoring",
            "instant updates",
        ]
        if any(pattern in lower_input for pattern in realtime_patterns):
            violations.append(
                {
                    "type": "capability_realtime_not_supported",
                    "severity": "medium",
                    "description": "Real-time processing capabilities not available",
                    "constraint_category": "interface_limits",
                }
            )

        return violations

    def _calculate_compliance_score(self, violations: List[Dict[str, Any]]) -> float:
        """Calculate overall compliance score based on violations"""
        if not violations:
            return 1.0

        # Weight violations by severity
        severity_weights = {"critical": 1.0, "high": 0.7, "medium": 0.5, "low": 0.2}

        total_penalty = sum(
            severity_weights.get(violation["severity"], 0.5) for violation in violations
        )

        # Calculate score (higher penalties = lower score)
        max_penalty = len(violations) * 1.0  # If all were critical
        score = max(0.0, 1.0 - (total_penalty / max(max_penalty, 1.0)))

        return score

    def _analyze_operations(
        self, intent_classification: Dict[str, float], violations: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Analyze which operations are permitted vs restricted"""

        # Base permitted operations (safe defaults)
        permitted = [
            "information_retrieval",
            "text_analysis",
            "creative_writing",
            "educational_content",
            "problem_solving",
            "conversation",
        ]

        # Operations restricted by violations
        restricted = []

        violation_types = {v["type"] for v in violations}

        # Map violation types to restricted operations
        restriction_map = {
            "ethical_harm_violence": ["content_generation", "instruction_provision"],
            "safety_injection_risk": ["code_execution", "system_interaction"],
            "capability_realtime_not_supported": ["real_time_processing"],
            "capability_financial_transactions_restricted": ["financial_operations"],
        }

        for violation_type in violation_types:
            if violation_type in restriction_map:
                restricted.extend(restriction_map[violation_type])

        # Remove duplicates
        restricted = list(set(restricted))

        return permitted, restricted

    def _generate_alternatives(
        self, violations: List[Dict[str, Any]], intent_classification: Dict[str, float]
    ) -> List[str]:
        """Generate alternative approaches for non-compliant requests"""
        alternatives = []

        violation_types = {v["type"] for v in violations}
        primary_intent = (
            max(intent_classification.items(), key=lambda x: x[1])[0]
            if intent_classification
            else "unknown"
        )

        # Generate alternatives based on violation types
        if any("harm" in vt for vt in violation_types):
            alternatives.extend(
                [
                    "Consider rephrasing your request to focus on constructive outcomes",
                    "I can help with educational information about conflict resolution instead",
                    "Let me suggest positive approaches to address your underlying concern",
                ]
            )

        if any("injection" in vt for vt in violation_types):
            alternatives.extend(
                [
                    "I can help explain programming concepts safely without executing code",
                    "Consider using a proper development environment for code testing",
                    "I can provide theoretical explanations of the concepts you're interested in",
                ]
            )

        if any("capability" in vt for vt in violation_types):
            alternatives.extend(
                [
                    "I can provide information and guidance within my available capabilities",
                    "Consider breaking down your request into smaller, manageable parts",
                    "I can help you understand the domain and suggest appropriate resources",
                ]
            )

        # Intent-based alternatives
        if primary_intent == "information_seeking":
            alternatives.append(
                "I can provide factual information from my knowledge base"
            )
        elif primary_intent == "problem_solving":
            alternatives.append(
                "I can help analyze the problem and suggest solution approaches"
            )
        elif primary_intent == "creative_task":
            alternatives.append(
                "I can assist with creative projects within appropriate boundaries"
            )

        return alternatives[:5]  # Limit to top 5 alternatives


# Global compliance engine instance
compliance_engine = PXLComplianceEngine()


@register_uip_handler(
    UIPStep.STEP_2_PXL_COMPLIANCE, dependencies=[UIPStep.STEP_1_LINGUISTIC], timeout=20
)
async def handle_pxl_compliance(context: UIPContext) -> Dict[str, Any]:
    """
    UIP Step 2: PXL Compliance Gate Handler

    Validates user requests against PXL operational constraints including
    ethical guidelines, safety requirements, and capability limitations.
    """
    logger = logging.getLogger(__name__)

    try:
        # Perform PXL compliance validation
        compliance_result = await compliance_engine.validate_compliance(context)

        # Convert result to dictionary for context storage
        result_dict = {
            "step": "pxl_compliance_gate_complete",
            "is_compliant": compliance_result.is_compliant,
            "compliance_score": compliance_result.compliance_score,
            "violations": compliance_result.violations,
            "permitted_operations": compliance_result.permitted_operations,
            "restricted_operations": compliance_result.restricted_operations,
            "ethical_flags": compliance_result.ethical_flags,
            "safety_flags": compliance_result.safety_flags,
            "capability_flags": compliance_result.capability_flags,
            "recommended_alternatives": compliance_result.recommended_alternatives,
            "compliance_metadata": compliance_result.compliance_metadata,
        }

        # Add compliance guidance
        guidance = []
        if not compliance_result.is_compliant:
            guidance.append("Request does not meet PXL compliance requirements.")
            if compliance_result.violations:
                guidance.append(
                    f"Found {len(compliance_result.violations)} constraint violations."
                )
            if compliance_result.recommended_alternatives:
                guidance.append("Alternative approaches are available.")
        else:
            guidance.append("Request meets all PXL compliance requirements.")
            if compliance_result.permitted_operations:
                guidance.append(
                    "Multiple operational modes are available for this request."
                )

        result_dict["compliance_guidance"] = guidance

        logger.info(
            f"PXL compliance validation completed for {context.correlation_id} "
            f"(compliant: {compliance_result.is_compliant}, "
            f"score: {compliance_result.compliance_score:.3f}, "
            f"violations: {len(compliance_result.violations)})"
        )

        return result_dict

    except Exception as e:
        logger.error(
            f"PXL compliance validation failed for {context.correlation_id}: {e}"
        )
        raise


__all__ = [
    "PXLComplianceResult",
    "PXLComplianceEngine",
    "compliance_engine",
    "handle_pxl_compliance",
]
