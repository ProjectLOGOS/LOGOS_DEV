"""
PXL Relation Mapper - UIP Step 2 Component
==========================================

Advanced relation mapping system for PXL (Philosophically Extended Logic) analysis.
Maps Trinity vector relationships, logical dependencies, and semantic coherence patterns.

Integrates with: Trinity vector processor, modal logic validator, ontological validator
Dependencies: V2 framework protocols, mathematical frameworks, formal logic systems
"""

from protocols.shared.system_imports import *
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import numpy as np
from collections import defaultdict
import itertools


class RelationType(Enum):
    """Types of PXL relations"""
    LOGICAL_IMPLICATION = "logical_implication"
    MODAL_ENTAILMENT = "modal_entailment" 
    TRINITY_COHERENCE = "trinity_coherence"
    ONTOLOGICAL_DEPENDENCY = "ontological_dependency"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CAUSAL_RELATION = "causal_relation"
    DEFINITIONAL_RELATION = "definitional_relation"
    ANALOGICAL_RELATION = "analogical_relation"
    HIERARCHICAL_RELATION = "hierarchical_relation"
    COMPOSITIONAL_RELATION = "compositional_relation"


class RelationStrength(Enum):
    """Strength levels of relations"""
    VERY_STRONG = "very_strong"      # 0.8-1.0
    STRONG = "strong"                # 0.6-0.8
    MODERATE = "moderate"            # 0.4-0.6
    WEAK = "weak"                    # 0.2-0.4
    VERY_WEAK = "very_weak"          # 0.0-0.2


class RelationDirection(Enum):
    """Direction of relations"""
    UNIDIRECTIONAL = "unidirectional"
    BIDIRECTIONAL = "bidirectional" 
    ASYMMETRIC = "asymmetric"


@dataclass
class PXLRelation:
    """Individual PXL relation between concepts"""
    source_concept: str
    target_concept: str
    relation_type: RelationType
    strength: float
    direction: RelationDirection
    confidence: float
    trinity_vector_source: Optional[Tuple[float, float, float]] = None
    trinity_vector_target: Optional[Tuple[float, float, float]] = None
    modal_properties: Dict[str, Any] = field(default_factory=dict)
    ontological_category: Optional[str] = None
    evidence_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        direction_symbol = "↔" if self.direction == RelationDirection.BIDIRECTIONAL else "→"
        return f"{self.source_concept} {direction_symbol} {self.target_concept} ({self.relation_type.value})"
    
    def get_strength_category(self) -> RelationStrength:
        """Get categorical strength level"""
        if self.strength >= 0.8:
            return RelationStrength.VERY_STRONG
        elif self.strength >= 0.6:
            return RelationStrength.STRONG
        elif self.strength >= 0.4:
            return RelationStrength.MODERATE
        elif self.strength >= 0.2:
            return RelationStrength.WEAK
        else:
            return RelationStrength.VERY_WEAK
    
    def calculate_trinity_coherence(self) -> float:
        """Calculate Trinity vector coherence between concepts"""
        if not (self.trinity_vector_source and self.trinity_vector_target):
            return 0.0
        
        # Calculate cosine similarity between Trinity vectors
        vec1 = np.array(self.trinity_vector_source)
        vec2 = np.array(self.trinity_vector_target)
        
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return max(0.0, cosine_similarity)  # Ensure non-negative


@dataclass
class RelationCluster:
    """Cluster of related PXL concepts"""
    name: str
    concepts: Set[str]
    internal_relations: List[PXLRelation]
    cluster_coherence: float
    dominant_relation_type: RelationType
    trinity_centroid: Optional[Tuple[float, float, float]] = None
    
    def calculate_cluster_density(self) -> float:
        """Calculate density of relations within cluster"""
        n_concepts = len(self.concepts)
        if n_concepts < 2:
            return 0.0
        
        max_relations = n_concepts * (n_concepts - 1) / 2
        actual_relations = len(self.internal_relations)
        
        return actual_relations / max_relations if max_relations > 0 else 0.0


@dataclass
class RelationMappingResult:
    """Result of PXL relation mapping analysis"""
    total_relations: int
    relation_graph: nx.DiGraph
    relation_clusters: List[RelationCluster]
    strongest_relations: List[PXLRelation]
    weakest_relations: List[PXLRelation]
    trinity_coherence_map: Dict[str, Dict[str, float]]
    ontological_hierarchy: Dict[str, List[str]]
    semantic_networks: Dict[RelationType, List[PXLRelation]]
    global_coherence_score: float
    analysis_metadata: Dict[str, Any]


class TrinityRelationAnalyzer:
    """Trinity vector-based relation analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.trinity_thresholds = {
            'high_coherence': 0.8,
            'moderate_coherence': 0.5,
            'low_coherence': 0.2
        }
    
    def analyze_trinity_relations(
        self, 
        concepts: Dict[str, Tuple[float, float, float]]
    ) -> List[PXLRelation]:
        """Analyze Trinity vector relationships between concepts"""
        
        relations = []
        concept_pairs = itertools.combinations(concepts.items(), 2)
        
        for (concept1, vector1), (concept2, vector2) in concept_pairs:
            # Calculate Trinity coherence
            coherence = self._calculate_trinity_coherence(vector1, vector2)
            
            # Determine relation strength and type
            if coherence >= self.trinity_thresholds['high_coherence']:
                strength = coherence
                relation_type = RelationType.TRINITY_COHERENCE
            elif coherence >= self.trinity_thresholds['moderate_coherence']:
                strength = coherence * 0.8  # Moderate confidence reduction
                relation_type = RelationType.SEMANTIC_SIMILARITY
            else:
                # Low coherence - check for complementary relationships
                complementarity = self._calculate_trinity_complementarity(vector1, vector2)
                if complementarity > 0.6:
                    strength = complementarity * 0.7
                    relation_type = RelationType.ANALOGICAL_RELATION
                else:
                    continue  # Skip weak relations
            
            # Create relation
            relation = PXLRelation(
                source_concept=concept1,
                target_concept=concept2,
                relation_type=relation_type,
                strength=strength,
                direction=RelationDirection.BIDIRECTIONAL,
                confidence=self._calculate_confidence(coherence, vector1, vector2),
                trinity_vector_source=vector1,
                trinity_vector_target=vector2,
                metadata={'trinity_coherence': coherence}
            )
            
            relations.append(relation)
        
        self.logger.debug(f"Analyzed {len(relations)} Trinity-based relations")
        return relations
    
    def _calculate_trinity_coherence(
        self, 
        vector1: Tuple[float, float, float], 
        vector2: Tuple[float, float, float]
    ) -> float:
        """Calculate coherence between two Trinity vectors"""
        
        vec1 = np.array(vector1)
        vec2 = np.array(vector2)
        
        # Cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        
        # Trinity-specific coherence adjustments
        # Higher weight for balanced Trinity dimensions
        balance_factor1 = 1.0 - np.std(vec1) / np.mean(vec1) if np.mean(vec1) > 0 else 0
        balance_factor2 = 1.0 - np.std(vec2) / np.mean(vec2) if np.mean(vec2) > 0 else 0
        balance_bonus = (balance_factor1 + balance_factor2) * 0.1
        
        coherence = max(0.0, cosine_sim + balance_bonus)
        return min(1.0, coherence)
    
    def _calculate_trinity_complementarity(
        self, 
        vector1: Tuple[float, float, float], 
        vector2: Tuple[float, float, float]
    ) -> float:
        """Calculate complementarity (inverse relationship) between Trinity vectors"""
        
        vec1 = np.array(vector1)
        vec2 = np.array(vector2)
        
        # Check for complementary patterns (high in different dimensions)
        dimension_complements = []
        
        for i in range(3):  # E, G, T dimensions
            if vec1[i] > 0.7 and vec2[i] < 0.3:  # One high, other low
                dimension_complements.append(1.0)
            elif vec1[i] < 0.3 and vec2[i] > 0.7:  # Inverse pattern
                dimension_complements.append(1.0)
            else:
                dimension_complements.append(0.0)
        
        complementarity = sum(dimension_complements) / 3.0
        return complementarity
    
    def _calculate_confidence(
        self, 
        coherence: float, 
        vector1: Tuple[float, float, float], 
        vector2: Tuple[float, float, float]
    ) -> float:
        """Calculate confidence in relation analysis"""
        
        # Base confidence from coherence
        base_confidence = coherence
        
        # Boost confidence for well-defined vectors (avoid zeros)
        vec1_definition = sum(1 for v in vector1 if v > 0.1) / 3.0
        vec2_definition = sum(1 for v in vector2 if v > 0.1) / 3.0
        definition_bonus = (vec1_definition + vec2_definition) * 0.1
        
        # Penalty for extreme vectors (too perfect or too weak)
        vec1_extremity = sum(1 for v in vector1 if v > 0.95 or v < 0.05) / 3.0
        vec2_extremity = sum(1 for v in vector2 if v > 0.95 or v < 0.05) / 3.0
        extremity_penalty = (vec1_extremity + vec2_extremity) * 0.1
        
        confidence = base_confidence + definition_bonus - extremity_penalty
        return max(0.0, min(1.0, confidence))


class LogicalRelationAnalyzer:
    """Logical dependency and implication analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logical_patterns = self._initialize_logical_patterns()
    
    def _initialize_logical_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for logical relation detection"""
        return {
            'implication_indicators': {
                'patterns': ['implies', 'entails', 'follows from', 'therefore', 'thus'],
                'relation_type': RelationType.LOGICAL_IMPLICATION,
                'strength_base': 0.8
            },
            'equivalence_indicators': {
                'patterns': ['equivalent to', 'if and only if', 'identical to'],
                'relation_type': RelationType.DEFINITIONAL_RELATION,
                'strength_base': 0.9
            },
            'causal_indicators': {
                'patterns': ['causes', 'results in', 'leads to', 'produces'],
                'relation_type': RelationType.CAUSAL_RELATION,
                'strength_base': 0.7
            }
        }
    
    def analyze_logical_relations(
        self, 
        concept_definitions: Dict[str, str]
    ) -> List[PXLRelation]:
        """Analyze logical relations from concept definitions"""
        
        relations = []
        
        for source_concept, definition in concept_definitions.items():
            # Find logical connections in definition text
            for target_concept in concept_definitions.keys():
                if source_concept != target_concept and target_concept.lower() in definition.lower():
                    
                    # Analyze relation type based on context
                    relation_type, strength, confidence = self._analyze_definition_context(
                        definition, source_concept, target_concept
                    )
                    
                    if relation_type and strength > 0.2:
                        relation = PXLRelation(
                            source_concept=source_concept,
                            target_concept=target_concept,
                            relation_type=relation_type,
                            strength=strength,
                            direction=RelationDirection.UNIDIRECTIONAL,
                            confidence=confidence,
                            evidence_sources=[f"Definition of {source_concept}"],
                            metadata={'definition_context': definition}
                        )
                        relations.append(relation)
        
        self.logger.debug(f"Analyzed {len(relations)} logical relations")
        return relations
    
    def _analyze_definition_context(
        self, 
        definition: str, 
        source_concept: str, 
        target_concept: str
    ) -> Tuple[Optional[RelationType], float, float]:
        """Analyze definitional context for relation type and strength"""
        
        definition_lower = definition.lower()
        target_lower = target_concept.lower()
        
        # Check for logical patterns
        for pattern_name, pattern_info in self.logical_patterns.items():
            for pattern in pattern_info['patterns']:
                if pattern in definition_lower:
                    # Find position relative to target concept
                    pattern_pos = definition_lower.find(pattern)
                    target_pos = definition_lower.find(target_lower)
                    
                    if abs(pattern_pos - target_pos) < 50:  # Within 50 characters
                        return (
                            pattern_info['relation_type'],
                            pattern_info['strength_base'],
                            0.8
                        )
        
        # Default: semantic similarity if target is mentioned
        if target_lower in definition_lower:
            return RelationType.SEMANTIC_SIMILARITY, 0.4, 0.6
        
        return None, 0.0, 0.0


class SemanticNetworkBuilder:
    """Semantic network construction and analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def build_semantic_networks(
        self, 
        relations: List[PXLRelation]
    ) -> Dict[RelationType, List[PXLRelation]]:
        """Build semantic networks by relation type"""
        
        networks = defaultdict(list)
        
        for relation in relations:
            networks[relation.relation_type].append(relation)
        
        # Sort each network by strength
        for relation_type in networks:
            networks[relation_type].sort(key=lambda r: r.strength, reverse=True)
        
        self.logger.debug(f"Built {len(networks)} semantic networks")
        return dict(networks)
    
    def identify_relation_clusters(
        self, 
        relations: List[PXLRelation], 
        min_cluster_size: int = 3
    ) -> List[RelationCluster]:
        """Identify clusters of strongly related concepts"""
        
        # Build adjacency graph
        graph = nx.Graph()
        
        for relation in relations:
            if relation.strength > 0.4:  # Only strong relations
                graph.add_edge(
                    relation.source_concept, 
                    relation.target_concept, 
                    weight=relation.strength,
                    relation=relation
                )
        
        # Find communities/clusters
        try:
            import networkx.algorithms.community as community
            communities = community.greedy_modularity_communities(graph)
        except:
            # Fallback: simple connected components
            communities = list(nx.connected_components(graph))
        
        clusters = []
        
        for i, community_nodes in enumerate(communities):
            if len(community_nodes) >= min_cluster_size:
                
                # Get relations within cluster
                cluster_relations = []
                for relation in relations:
                    if (relation.source_concept in community_nodes and 
                        relation.target_concept in community_nodes):
                        cluster_relations.append(relation)
                
                if cluster_relations:
                    # Calculate cluster properties
                    avg_strength = sum(r.strength for r in cluster_relations) / len(cluster_relations)
                    
                    # Find dominant relation type
                    relation_types = [r.relation_type for r in cluster_relations]
                    dominant_type = max(set(relation_types), key=relation_types.count)
                    
                    cluster = RelationCluster(
                        name=f"Cluster_{i+1}",
                        concepts=community_nodes,
                        internal_relations=cluster_relations,
                        cluster_coherence=avg_strength,
                        dominant_relation_type=dominant_type
                    )
                    
                    clusters.append(cluster)
        
        self.logger.debug(f"Identified {len(clusters)} relation clusters")
        return clusters


class PXLRelationMapper:
    """Main PXL relation mapping engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize analyzers
        self.trinity_analyzer = TrinityRelationAnalyzer()
        self.logical_analyzer = LogicalRelationAnalyzer()
        self.network_builder = SemanticNetworkBuilder()
        
        # Configuration
        self.min_relation_strength = self.config.get('min_relation_strength', 0.2)
        self.max_relations_per_concept = self.config.get('max_relations_per_concept', 20)
        
        # Caching
        self._mapping_cache: Dict[str, RelationMappingResult] = {}
        
        self.logger.info("PXL relation mapper initialized")
    
    def map_concept_relations(
        self,
        concepts: Dict[str, Any],
        trinity_vectors: Optional[Dict[str, Tuple[float, float, float]]] = None,
        concept_definitions: Optional[Dict[str, str]] = None
    ) -> RelationMappingResult:
        """
        Map relations between PXL concepts
        
        Args:
            concepts: Dictionary of concepts with metadata
            trinity_vectors: Trinity vector representations of concepts
            concept_definitions: Textual definitions of concepts
            
        Returns:
            RelationMappingResult: Comprehensive relation mapping analysis
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(concepts, trinity_vectors, concept_definitions)
            if cache_key in self._mapping_cache:
                return self._mapping_cache[cache_key]
            
            all_relations = []
            
            # Trinity-based relation analysis
            if trinity_vectors:
                trinity_relations = self.trinity_analyzer.analyze_trinity_relations(trinity_vectors)
                all_relations.extend(trinity_relations)
                self.logger.debug(f"Added {len(trinity_relations)} Trinity-based relations")
            
            # Logical relation analysis
            if concept_definitions:
                logical_relations = self.logical_analyzer.analyze_logical_relations(concept_definitions)
                all_relations.extend(logical_relations)
                self.logger.debug(f"Added {len(logical_relations)} logical relations")
            
            # Filter by minimum strength
            filtered_relations = [
                r for r in all_relations 
                if r.strength >= self.min_relation_strength
            ]
            
            # Build relation graph
            relation_graph = self._build_relation_graph(filtered_relations)
            
            # Identify clusters
            relation_clusters = self.network_builder.identify_relation_clusters(filtered_relations)
            
            # Build semantic networks
            semantic_networks = self.network_builder.build_semantic_networks(filtered_relations)
            
            # Analyze strongest and weakest relations
            sorted_relations = sorted(filtered_relations, key=lambda r: r.strength, reverse=True)
            strongest_relations = sorted_relations[:10]
            weakest_relations = sorted_relations[-10:] if len(sorted_relations) > 10 else []
            
            # Build Trinity coherence map
            trinity_coherence_map = self._build_trinity_coherence_map(filtered_relations)
            
            # Build ontological hierarchy
            ontological_hierarchy = self._build_ontological_hierarchy(filtered_relations, concepts)
            
            # Calculate global coherence
            global_coherence = self._calculate_global_coherence(filtered_relations, relation_clusters)
            
            # Analysis metadata
            analysis_metadata = {
                'total_concepts': len(concepts),
                'total_relations_found': len(all_relations),
                'filtered_relations': len(filtered_relations),
                'relation_density': len(filtered_relations) / max(len(concepts) * (len(concepts) - 1), 1),
                'avg_relation_strength': sum(r.strength for r in filtered_relations) / max(len(filtered_relations), 1),
                'processing_timestamp': time.time()
            }
            
            result = RelationMappingResult(
                total_relations=len(filtered_relations),
                relation_graph=relation_graph,
                relation_clusters=relation_clusters,
                strongest_relations=strongest_relations,
                weakest_relations=weakest_relations,
                trinity_coherence_map=trinity_coherence_map,
                ontological_hierarchy=ontological_hierarchy,
                semantic_networks=semantic_networks,
                global_coherence_score=global_coherence,
                analysis_metadata=analysis_metadata
            )
            
            # Cache result
            self._mapping_cache[cache_key] = result
            
            self.logger.info(f"Mapped {len(filtered_relations)} relations across {len(concepts)} concepts")
            return result
            
        except Exception as e:
            self.logger.error(f"Relation mapping failed: {e}")
            raise
    
    def _build_relation_graph(self, relations: List[PXLRelation]) -> nx.DiGraph:
        """Build NetworkX graph from relations"""
        
        graph = nx.DiGraph()
        
        for relation in relations:
            if relation.direction == RelationDirection.BIDIRECTIONAL:
                graph.add_edge(
                    relation.source_concept,
                    relation.target_concept,
                    weight=relation.strength,
                    relation_type=relation.relation_type.value,
                    relation=relation
                )
                graph.add_edge(
                    relation.target_concept,
                    relation.source_concept,
                    weight=relation.strength,
                    relation_type=relation.relation_type.value,
                    relation=relation
                )
            else:
                graph.add_edge(
                    relation.source_concept,
                    relation.target_concept,
                    weight=relation.strength,
                    relation_type=relation.relation_type.value,
                    relation=relation
                )
        
        return graph
    
    def _build_trinity_coherence_map(self, relations: List[PXLRelation]) -> Dict[str, Dict[str, float]]:
        """Build Trinity coherence mapping between concepts"""
        
        coherence_map = defaultdict(dict)
        
        for relation in relations:
            if relation.trinity_vector_source and relation.trinity_vector_target:
                coherence = relation.calculate_trinity_coherence()
                coherence_map[relation.source_concept][relation.target_concept] = coherence
                
                if relation.direction == RelationDirection.BIDIRECTIONAL:
                    coherence_map[relation.target_concept][relation.source_concept] = coherence
        
        return dict(coherence_map)
    
    def _build_ontological_hierarchy(
        self, 
        relations: List[PXLRelation], 
        concepts: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Build ontological hierarchy from hierarchical relations"""
        
        hierarchy = defaultdict(list)
        
        hierarchical_relations = [
            r for r in relations 
            if r.relation_type == RelationType.HIERARCHICAL_RELATION
        ]
        
        for relation in hierarchical_relations:
            # Source is typically the superordinate concept
            hierarchy[relation.source_concept].append(relation.target_concept)
        
        return dict(hierarchy)
    
    def _calculate_global_coherence(
        self, 
        relations: List[PXLRelation], 
        clusters: List[RelationCluster]
    ) -> float:
        """Calculate global coherence score for the relation network"""
        
        if not relations:
            return 0.0
        
        # Average relation strength
        avg_strength = sum(r.strength for r in relations) / len(relations)
        
        # Cluster coherence bonus
        if clusters:
            avg_cluster_coherence = sum(c.cluster_coherence for c in clusters) / len(clusters)
            cluster_bonus = avg_cluster_coherence * 0.2
        else:
            cluster_bonus = 0.0
        
        # Trinity coherence bonus
        trinity_relations = [r for r in relations if r.relation_type == RelationType.TRINITY_COHERENCE]
        if trinity_relations:
            trinity_coherence = sum(r.strength for r in trinity_relations) / len(trinity_relations)
            trinity_bonus = trinity_coherence * 0.3
        else:
            trinity_bonus = 0.0
        
        global_coherence = avg_strength + cluster_bonus + trinity_bonus
        return min(1.0, global_coherence)
    
    def _generate_cache_key(
        self,
        concepts: Dict[str, Any],
        trinity_vectors: Optional[Dict[str, Tuple[float, float, float]]],
        concept_definitions: Optional[Dict[str, str]]
    ) -> str:
        """Generate cache key for relation mapping"""
        
        import hashlib
        
        # Create hash from input parameters
        key_data = {
            'concepts': sorted(concepts.keys()),
            'trinity_vectors': sorted(trinity_vectors.keys()) if trinity_vectors else [],
            'definitions': sorted(concept_definitions.keys()) if concept_definitions else []
        }
        
        key_string = str(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()


# Global PXL relation mapper instance
pxl_relation_mapper = PXLRelationMapper()


__all__ = [
    'RelationType',
    'RelationStrength',
    'RelationDirection',
    'PXLRelation',
    'RelationCluster',
    'RelationMappingResult',
    'TrinityRelationAnalyzer',
    'LogicalRelationAnalyzer',
    'SemanticNetworkBuilder',
    'PXLRelationMapper',
    'pxl_relation_mapper'
]