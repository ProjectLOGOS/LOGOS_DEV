"""Ontological Node Implementation

Provides the core OntologicalNode class for THŌNOC system, integrating
Lambda Logos with Mandelbrot fractal space mapping. Nodes store ontological
properties, Lambda expressions, and fractal positioning coordinates.

Key components:
- Fractal position calculation
- EGT indexing
- Lambda state storage
- Mandelbrot orbit analysis

Dependencies: cmath, typing, uuid
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Set
import cmath
import math
import uuid
import json
import time
from enum import Enum

# Import Lambda Logos core (adjust import path as needed)
try:
    from lambda_logos_core import LogosExpr, LambdaLogosEngine
except ImportError:
    # Mock classes for standalone use
    class LogosExpr:
        def to_dict(self) -> Dict[str, Any]: return {}
    
    class LambdaLogosEngine:
        def evaluate(self, expr): return expr

class CategoryType(Enum):
    """Node category types."""
    MATERIAL = "MATERIAL"
    METAPHYSICAL = "METAPHYSICAL"

class DomainType(Enum):
    """Ontological domain types."""
    LOGICAL = "LOGICAL"
    TRANSCENDENTAL = "TRANSCENDENTAL"

class TrinityIndex(Enum):
    """Trinity dimension indices."""
    EXISTENCE = "E"
    GOODNESS = "G"
    TRUTH = "T"

class OntologicalNode:
    """Node in ontological fractal space."""
    
    def __init__(self, c_value: complex):
        """Initialize ontological node.
        
        Args:
            c_value: Complex position in Mandelbrot space
        """
        self.c = c_value
        self.node_id = self._generate_id(c_value)
        
        # Core categorization
        self.category = CategoryType.MATERIAL if self.c.imag == 0 else CategoryType.METAPHYSICAL
        self.trinitarian_domain = DomainType.LOGICAL if self.category == CategoryType.MATERIAL else DomainType.TRANSCENDENTAL
        self.invariant_value = 3 if self.trinitarian_domain == DomainType.LOGICAL else 1
        
        # Orbital properties
        self.orbit_properties = self._calculate_orbit_properties(self.c)
        self.calculation_depth = self.orbit_properties.get("depth", 0)
        
        # Trinity indexing
        self.index_E = 0 if self.category == CategoryType.MATERIAL else 1
        self.index_G = 0 if self.trinitarian_domain == DomainType.LOGICAL else 1
        self.index_T = self._calculate_T_index(self.orbit_properties)
        
        # Trinity vector (existence, goodness, truth)
        self.trinity_vector = (
            0.5 + (0.5 * self.index_E),
            0.5 + (0.5 * self.index_G),
            0.5 + (0.5 * self.index_T)
        )
        
        # Data payload
        self.data_payload = {
            "label": None,
            "semantic_props": {},
            "lambda_state": {
                "environment": {},
                "current_term": None,
                "evaluation_result": None
            }
        }
        
        # Node relationships
        self.relationships = []
        self.modal_status = {"necessary": [], "possible": []}
        self.timestamps = {"created": time.time(), "updated": time.time()}
        
    def _generate_id(self, c_value: complex) -> str:
        """Generate unique ID for node based on complex coordinates.
        
        Args:
            c_value: Complex position
            
        Returns:
            Unique node identifier
        """
        base = f"node_{c_value.real:.6f}_{c_value.imag:.6f}"
        return f"{base}_{uuid.uuid4().hex[:8]}"
    
    def _calculate_orbit_properties(self, c: complex) -> Dict[str, Any]:
        """Calculate Mandelbrot orbit properties for complex value.
        
        Args:
            c: Complex parameter
            
        Returns:
            Orbit properties dictionary
        """
        max_iter = 500 if self.category == CategoryType.METAPHYSICAL else 100
        escape_radius = 2.0
        
        # Calculate orbit
        z = complex(0, 0)
        orbit = []
        
        for i in range(max_iter):
            orbit.append(z)
            z = z * z + c
            if abs(z) > escape_radius:
                break
        
        # Determine if in Mandelbrot set
        in_set = i == max_iter - 1
        
        # Calculate properties
        orbit_type = "COMPLEX_ORBIT" if i > 20 else "SIMPLE_ORBIT"
        
        return {
            "depth": i,
            "in_set": in_set,
            "orbit": orbit[:10],  # Store first 10 points only
            "type": orbit_type,
            "escape_value": abs(z),
            "final_z": (z.real, z.imag)
        }
    
    def _calculate_T_index(self, orbit_props: Dict[str, Any]) -> int:
        """Calculate Truth index from orbit properties.
        
        Args:
            orbit_props: Orbit properties dictionary
            
        Returns:
            Truth index (0 or 1)
        """
        # Simple mapping based on orbit type and depth
        if orbit_props.get("type") == "COMPLEX_ORBIT" or orbit_props.get("in_set", False):
            return 1
        return 0
    
    def update_lambda_term(self, term: LogosExpr, env: Dict[str, Any] = None) -> None:
        """Update node's Lambda term and environment.
        
        Args:
            term: Lambda expression
            env: Environment bindings (optional)
        """
        self.data_payload["lambda_state"]["current_term"] = term
        
        if env:
            self.data_payload["lambda_state"]["environment"].update(env)
        
        self.timestamps["updated"] = time.time()
    
    def evaluate_lambda_term(self, engine: Optional[LambdaLogosEngine] = None) -> Any:
        """Evaluate current Lambda term.
        
        Args:
            engine: Lambda engine instance (optional)
            
        Returns:
            Evaluation result
        """
        term = self.data_payload["lambda_state"]["current_term"]
        
        if not term:
            return None
        
        # Use provided engine or basic evaluation
        if engine:
            result = engine.evaluate(term)
        else:
            # Basic fallback (ideally we'd use a real engine)
            result = term
        
        self.data_payload["lambda_state"]["evaluation_result"] = result
        return result
    
    def add_relationship(self, relation_type: str, target_node_id: str, metadata: Dict[str, Any] = None) -> None:
        """Add relationship to another node.
        
        Args:
            relation_type: Type of relation
            target_node_id: Target node ID
            metadata: Optional relationship metadata
        """
        rel = (relation_type, target_node_id, metadata or {})
        self.relationships.append(rel)
        self.timestamps["updated"] = time.time()
    
    def update_modal_status(self, status: str, proposition_id: str) -> None:
        """Update modal status for a proposition.
        
        Args:
            status: Modal status ("necessary" or "possible")
            proposition_id: Proposition identifier
        """
        if status in self.modal_status:
            if proposition_id not in self.modal_status[status]:
                self.modal_status[status].append(proposition_id)
                self.timestamps["updated"] = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        lambda_state = self.data_payload["lambda_state"].copy()
        
        # Convert Lambda term to dict if present
        if lambda_state["current_term"]:
            try:
                lambda_state["current_term"] = lambda_state["current_term"].to_dict()
            except AttributeError:
                lambda_state["current_term"] = str(lambda_state["current_term"])
        
        # Convert evaluation result to dict if present
        if lambda_state["evaluation_result"]:
            try:
                lambda_state["evaluation_result"] = lambda_state["evaluation_result"].to_dict()
            except AttributeError:
                lambda_state["evaluation_result"] = str(lambda_state["evaluation_result"])
        
        return {
            "node_id": self.node_id,
            "c_value": {
                "real": self.c.real,
                "imag": self.c.imag
            },
            "category": self.category.value,
            "domain": self.trinitarian_domain.value,
            "invariant_value": self.invariant_value,
            "calculation_depth": self.calculation_depth,
            "trinity_indices": {
                "E": self.index_E,
                "G": self.index_G,
                "T": self.index_T
            },
            "trinity_vector": self.trinity_vector,
            "data_payload": {
                "label": self.data_payload["label"],
                "semantic_props": self.data_payload["semantic_props"],
                "lambda_state": lambda_state
            },
            "relationships": self.relationships,
            "modal_status": self.modal_status,
            "timestamps": self.timestamps
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OntologicalNode':
        """Create node from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Ontological node
        """
        # Create base node
        c_value = complex(
            data.get("c_value", {}).get("real", 0),
            data.get("c_value", {}).get("imag", 0)
        )
        node = cls(c_value)
        
        # Update fields
        node.node_id = data.get("node_id", node.node_id)
        node.data_payload["label"] = data.get("data_payload", {}).get("label")
        node.data_payload["semantic_props"] = data.get("data_payload", {}).get("semantic_props", {})
        
        # Lambda state requires special handling
        lambda_state = data.get("data_payload", {}).get("lambda_state", {})
        node.data_payload["lambda_state"]["environment"] = lambda_state.get("environment", {})
        
        # Note: Actual Lambda expression objects would need proper reconstruction
        # This is a placeholder - in reality, we'd use LogosExpr.from_dict()
        node.data_payload["lambda_state"]["current_term"] = lambda_state.get("current_term")
        node.data_payload["lambda_state"]["evaluation_result"] = lambda_state.get("evaluation_result")
        
        # Other fields
        node.relationships = data.get("relationships", [])
        node.modal_status = data.get("modal_status", {"necessary": [], "possible": []})
        node.timestamps = data.get("timestamps", {"created": time.time(), "updated": time.time()})
        
        return node

# Example usage
if __name__ == "__main__":
    # Create node in material domain (real number)
    real_node = OntologicalNode(complex(0.5, 0))
    print(f"Material node: E={real_node.index_E}, G={real_node.index_G}, T={real_node.index_T}")
    
    # Create node in metaphysical domain (complex number)
    complex_node = OntologicalNode(complex(-0.5, 0.6))
    print(f"Metaphysical node: E={complex_node.index_E}, G={complex_node.index_G}, T={complex_node.index_T}")
    
    # Add relationship
    complex_node.add_relationship("derivation", real_node.node_id, {"strength": 0.8})
    
    # Serialize and deserialize
    node_dict = complex_node.to_dict()
    print(f"Serialized node ID: {node_dict['node_id']}")
    
    # Recreate from dict
    recreated = OntologicalNode.from_dict(node_dict)
    print(f"Recreated node ID: {recreated.node_id}")
    print(f"Relationship count: {len(recreated.relationships)}")
    
    # Trinity vector representation
    print(f"Trinity vector: ({complex_node.trinity_vector[0]}, "
          f"{complex_node.trinity_vector[1]}, {complex_node.trinity_vector[2]})")