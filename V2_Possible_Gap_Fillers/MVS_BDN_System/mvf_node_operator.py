"""Ontological Fractal Database

Persistent storage system for fractal knowledge representation with
dimensional indexing (𝔼-𝔾-𝕋) and O(log n) access to ontological nodes.

Key Features:
- Persistent node storage
- Multi-dimensional indexing
- Fractal position tracking
- Entailment chain preservation

Dependencies: numpy, scipy, sqlalchemy
"""

from typing import Dict, List, Tuple, Optional, Union, Set, Any, NamedTuple, TypeVar, Generic
import numpy as np
import sqlite3
import json
import time
import math
import hashlib
from enum import Enum
from collections import defaultdict
import heapq
from dataclasses import dataclass, field, asdict

T = TypeVar('T')

class TrinityDimension(Enum):
    """Ontological dimensions of the trinitarian framework."""
    EXISTENCE = "existence"  # 𝔼
    GOODNESS = "goodness"    # 𝔾
    TRUTH = "truth"          # 𝕋

@dataclass
class FractalPosition:
    """Position in Mandelbrot fractal space."""
    c_real: float
    c_imag: float
    iterations: int
    in_set: bool
    escape_radius: float = 2.0
    
    @property
    def complex(self) -> complex:
        """Get position as complex number."""
        return complex(self.c_real, self.c_imag)
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'FractalPosition':
        """Create instance from serialized data."""
        return cls(**data)

@dataclass
class TrinityVector:
    """Trinity vector (𝔼-𝔾-𝕋)."""
    existence: float
    goodness: float
    truth: float
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """Get as tuple (existence, goodness, truth)."""
        return (self.existence, self.goodness, self.truth)
    
    def serialize(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return asdict(self)
    
    @classmethod
    def deserialize(cls, data: Dict[str, float]) -> 'TrinityVector':
        """Create instance from serialized data."""
        return cls(**data)
    
    def __getitem__(self, dimension: TrinityDimension) -> float:
        """Get value for specific dimension."""
        if dimension == TrinityDimension.EXISTENCE:
            return self.existence
        elif dimension == TrinityDimension.GOODNESS:
            return self.goodness
        elif dimension == TrinityDimension.TRUTH:
            return self.truth
        raise ValueError(f"Unknown dimension: {dimension}")

@dataclass
class OntologicalNode:
    """Node in ontological knowledge database."""
    id: str
    query: str
    trinity_vector: TrinityVector
    fractal_position: FractalPosition
    created_at: float
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = {
            "id": self.id,
            "query": self.query,
            "trinity_vector": self.trinity_vector.serialize(),
            "fractal_position": self.fractal_position.serialize(),
            "created_at": self.created_at,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata
        }
        return data
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'OntologicalNode':
        """Create instance from serialized data."""
        # Handle nested objects
        data["trinity_vector"] = TrinityVector.deserialize(data["trinity_vector"])
        data["fractal_position"] = FractalPosition.deserialize(data["fractal_position"])
        return cls(**data)

class KDNode:
    """Node for k-d tree implementation."""
    
    def __init__(self, node_id: str, point: List[float]):
        """Initialize k-d tree node.
        
        Args:
            node_id: Ontological node ID
            point: k-dimensional point
        """
        self.node_id = node_id
        self.point = point
        self.left = None
        self.right = None

class KDTree:
    """k-d tree implementation for efficient spatial indexing."""
    
    def __init__(self, k: int = 3):
        """Initialize k-d tree.
        
        Args:
            k: Number of dimensions
        """
        self.k = k
        self.root = None
    
    def insert(self, node_id: str, point: List[float]) -> None:
        """Insert node into k-d tree.
        
        Args:
            node_id: Ontological node ID
            point: k-dimensional point
        """
        self.root = self._insert(self.root, node_id, point, 0)
    
    def _insert(self, node: Optional[KDNode], node_id: str, point: List[float], depth: int) -> KDNode:
        """Internal recursive insertion function."""
        if node is None:
            return KDNode(node_id, point)
        
        axis = depth % self.k
        
        if point[axis] < node.point[axis]:
            node.left = self._insert(node.left, node_id, point, depth + 1)
        else:
            node.right = self._insert(node.right, node_id, point, depth + 1)
            
        return node
    
    def nearest_neighbor(self, point: List[float]) -> Optional[str]:
        """Find nearest neighbor to query point.
        
        Args:
            point: Query point
            
        Returns:
            ID of nearest node or None if tree is empty
        """
        if self.root is None:
            return None
            
        best = [None, float('inf')]  # [node_id, distance]
        self._nearest_neighbor(self.root, point, 0, best)
        return best[0]
    
    def _nearest_neighbor(self, node: KDNode, point: List[float], depth: int, best: List[Any]) -> None:
        """Internal recursive nearest neighbor search."""
        if node is None:
            return
            
        # Calculate distance to current node
        dist = sum((a - b) ** 2 for a, b in zip(node.point, point))
        
        # Update best if current is closer
        if dist < best[1]:
            best[0] = node.node_id
            best[1] = dist
        
        # Recursively search subtrees
        axis = depth % self.k
        diff = point[axis] - node.point[axis]
        
        # Search closest subtree first
        if diff < 0:
            self._nearest_neighbor(node.left, point, depth + 1, best)
            # Only search other subtree if it could contain closer points
            if diff ** 2 < best[1]:
                self._nearest_neighbor(node.right, point, depth + 1, best)
        else:
            self._nearest_neighbor(node.right, point, depth + 1, best)
            if diff ** 2 < best[1]:
                self._nearest_neighbor(node.left, point, depth + 1, best)
    
    def k_nearest_neighbors(self, point: List[float], k: int) -> List[Tuple[str, float]]:
        """Find k nearest neighbors to query point.
        
        Args:
            point: Query point
            k: Number of neighbors to return
            
        Returns:
            List of (node_id, distance) tuples
        """
        if self.root is None:
            return []
            
        nearest = []  # Priority queue of (-distance, node_id)
        self._k_nearest_neighbors(self.root, point, 0, nearest, k)
        
        # Convert to list of (node_id, distance) tuples
        result = [(nid, -dist) for dist, nid in nearest]
        result.sort(key=lambda x: x[1])
        return result
    
    def _k_nearest_neighbors(self, node: KDNode, point: List[float], depth: int, 
                            nearest: List[Tuple[float, str]], k: int) -> None:
        """Internal recursive k-nearest neighbors search."""
        if node is None:
            return
            
        # Calculate distance to current node
        dist = sum((a - b) ** 2 for a, b in zip(node.point, point))
        
        # Update nearest if current node qualifies
        if len(nearest) < k:
            heapq.heappush(nearest, (-dist, node.node_id))
        elif -dist > nearest[0][0]:  # If current dist is smaller than largest in heap
            heapq.heapreplace(nearest, (-dist, node.node_id))
        
        # Recursively search subtrees
        axis = depth % self.k
        diff = point[axis] - node.point[axis]
        
        # Search closest subtree first
        if diff < 0:
            self._k_nearest_neighbors(node.left, point, depth + 1, nearest, k)
            # Only search other subtree if it could contain closer points
            if len(nearest) < k or diff ** 2 < -nearest[0][0]:
                self._k_nearest_neighbors(node.right, point, depth + 1, nearest, k)
        else:
            self._k_nearest_neighbors(node.right, point, depth + 1, nearest, k)
            if len(nearest) < k or diff ** 2 < -nearest[0][0]:
                self._k_nearest_neighbors(node.left, point, depth + 1, nearest, k)

class FractalKnowledgeDatabase:
    """Main database class for ontological knowledge storage."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize ontological database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or ":memory:"
        self.conn = self._initialize_database()
        
        # In-memory indices
        self.trinity_index = KDTree(k=3)
        self.complex_index = KDTree(k=2)
        self.nodes = {}  # Cache of loaded nodes
    
    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        
        # Create tables if not exist
        with conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                trinity_vector TEXT NOT NULL,
                fractal_position TEXT NOT NULL,
                created_at REAL NOT NULL,
                parent_id TEXT,
                metadata TEXT
            )
            ''')
            
            conn.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL NOT NULL,
                metadata TEXT,
                PRIMARY KEY (source_id, target_id, relation_type)
            )
            ''')
            
            # Create indices
            conn.execute('CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id)')
        
        return conn
    
    def store_node(self, node: OntologicalNode) -> None:
        """Store ontological node in database.
        
        Args:
            node: Node to store
        """
        # Serialize node data
        with self.conn:
            self.conn.execute(
                'INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)',
                (
                    node.id,
                    node.query,
                    json.dumps(node.trinity_vector.serialize()),
                    json.dumps(node.fractal_position.serialize()),
                    node.created_at,
                    node.parent_id,
                    json.dumps(node.metadata)
                )
            )
        
        # Update parent-child relationship if parent exists
        if node.parent_id:
            parent = self.get_node(node.parent_id)
            if parent and node.id not in parent.children_ids:
                parent.children_ids.append(node.id)
                self.store_node(parent)
        
        # Update indices
        point_trinity = list(node.trinity_vector.as_tuple())
        point_complex = [node.fractal_position.c_real, node.fractal_position.c_imag]
        
        self.trinity_index.insert(node.id, point_trinity)
        self.complex_index.insert(node.id, point_complex)
        
        # Update cache
        self.nodes[node.id] = node
    
    def get_node(self, node_id: str) -> Optional[OntologicalNode]:
        """Retrieve node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node or None if not found
        """
        # Check cache first
        if node_id in self.nodes:
            return self.nodes[node_id]
        
        # Query database
        cursor = self.conn.execute(
            'SELECT query, trinity_vector, fractal_position, created_at, parent_id, metadata FROM nodes WHERE id = ?',
            (node_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return None
            
        # Deserialize data
        query, trinity_vector_json, fractal_position_json, created_at, parent_id, metadata_json = row
        
        trinity_vector = TrinityVector.deserialize(json.loads(trinity_vector_json))
        fractal_position = FractalPosition.deserialize(json.loads(fractal_position_json))
        metadata = json.loads(metadata_json) if metadata_json else {}
        
        # Find children
        children_cursor = self.conn.execute(
            'SELECT id FROM nodes WHERE parent_id = ?',
            (node_id,)
        )
        children_ids = [row[0] for row in children_cursor.fetchall()]
        
        # Create node
        node = OntologicalNode(
            id=node_id,
            query=query,
            trinity_vector=trinity_vector,
            fractal_position=fractal_position,
            created_at=created_at,
            parent_id=parent_id,
            children_ids=children_ids,
            metadata=metadata
        )
        
        # Update cache
        self.nodes[node_id] = node
        
        return node
    
    def create_node(self, 
                   query: str, 
                   trinity_vector: TrinityVector,
                   fractal_position: FractalPosition,
                   parent_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> OntologicalNode:
        """Create and store new ontological node.
        
        Args:
            query: Original query string
            trinity_vector: Trinity vector (𝔼-𝔾-𝕋)
            fractal_position: Position in fractal space
            parent_id: Optional parent node ID
            metadata: Optional additional metadata
            
        Returns:
            Created node
        """
        # Generate ID
        node_id = self._generate_id(query)
        
        # Create node
        node = OntologicalNode(
            id=node_id,
            query=query,
            trinity_vector=trinity_vector,
            fractal_position=fractal_position,
            created_at=time.time(),
            parent_id=parent_id,
            children_ids=[],
            metadata=metadata or {}
        )
        
        # Store node
        self.store_node(node)
        
        return node
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID for node based on content.
        
        Args:
            text: Text to hash
            
        Returns:
            Unique ID string
        """
        hash_input = f"{text}:{time.time()}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def add_relation(self, 
                    source_id: str, 
                    target_id: str, 
                    relation_type: str,
                    weight: float = 1.0,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add relation between nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relation
            weight: Relation strength
            metadata: Optional relation metadata
            
        Returns:
            True if successful, False otherwise
        """
        # Verify nodes exist
        if not self.get_node(source_id) or not self.get_node(target_id):
            return False
        
        # Store relation
        with self.conn:
            self.conn.execute(
                'INSERT OR REPLACE INTO relations VALUES (?, ?, ?, ?, ?)',
                (
                    source_id,
                    target_id,
                    relation_type,
                    weight,
                    json.dumps(metadata or {})
                )
            )
        
        return True
    
    def get_relations(self, 
                     node_id: str, 
                     relation_type: Optional[str] = None,
                     direction: str = "outgoing") -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Get node relations.
        
        Args:
            node_id: Node ID
            relation_type: Optional relation type filter
            direction: "outgoing", "incoming" or "both"
            
        Returns:
            List of (node_id, relation_type, weight, metadata) tuples
        """
        relations = []
        
        # Query outgoing relations
        if direction in ["outgoing", "both"]:
            query = 'SELECT target_id, relation_type, weight, metadata FROM relations WHERE source_id = ?'
            params = [node_id]
			"""Ontological Fractal Database

Persistent storage system for fractal knowledge representation with
dimensional indexing (𝔼-𝔾-𝕋) and O(log n) access to ontological nodes.

Key Features:
- Persistent node storage
- Multi-dimensional indexing
- Fractal position tracking
- Entailment chain preservation

Dependencies: numpy, scipy, sqlalchemy
"""

from typing import Dict, List, Tuple, Optional, Union, Set, Any, NamedTuple, TypeVar, Generic
import numpy as np
import sqlite3
import json
import time
import math
import hashlib
from enum import Enum
from collections import defaultdict
import heapq
from dataclasses import dataclass, field, asdict

T = TypeVar('T')

class TrinityDimension(Enum):
    """Ontological dimensions of the trinitarian framework."""
    EXISTENCE = "existence"  # 𝔼
    GOODNESS = "goodness"    # 𝔾
    TRUTH = "truth"          # 𝕋

@dataclass
class FractalPosition:
    """Position in Mandelbrot fractal space."""
    c_real: float
    c_imag: float
    iterations: int
    in_set: bool
    escape_radius: float = 2.0
    
    @property
    def complex(self) -> complex:
        """Get position as complex number."""
        return complex(self.c_real, self.c_imag)
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'FractalPosition':
        """Create instance from serialized data."""
        return cls(**data)

@dataclass
class TrinityVector:
    """Trinity vector (𝔼-𝔾-𝕋)."""
    existence: float
    goodness: float
    truth: float
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """Get as tuple (existence, goodness, truth)."""
        return (self.existence, self.goodness, self.truth)
    
    def serialize(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return asdict(self)
    
    @classmethod
    def deserialize(cls, data: Dict[str, float]) -> 'TrinityVector':
        """Create instance from serialized data."""
        return cls(**data)
    
    def __getitem__(self, dimension: TrinityDimension) -> float:
        """Get value for specific dimension."""
        if dimension == TrinityDimension.EXISTENCE:
            return self.existence
        elif dimension == TrinityDimension.GOODNESS:
            return self.goodness
        elif dimension == TrinityDimension.TRUTH:
            return self.truth
        raise ValueError(f"Unknown dimension: {dimension}")

@dataclass
class OntologicalNode:
    """Node in ontological knowledge database."""
    id: str
    query: str
    trinity_vector: TrinityVector
    fractal_position: FractalPosition
    created_at: float
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = {
            "id": self.id,
            "query": self.query,
            "trinity_vector": self.trinity_vector.serialize(),
            "fractal_position": self.fractal_position.serialize(),
            "created_at": self.created_at,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata
        }
        return data
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'OntologicalNode':
        """Create instance from serialized data."""
        # Handle nested objects
        data["trinity_vector"] = TrinityVector.deserialize(data["trinity_vector"])
        data["fractal_position"] = FractalPosition.deserialize(data["fractal_position"])
        return cls(**data)

class KDNode:
    """Node for k-d tree implementation."""
    
    def __init__(self, node_id: str, point: List[float]):
        """Initialize k-d tree node.
        
        Args:
            node_id: Ontological node ID
            point: k-dimensional point
        """
        self.node_id = node_id
        self.point = point
        self.left = None
        self.right = None

class KDTree:
    """k-d tree implementation for efficient spatial indexing."""
    
    def __init__(self, k: int = 3):
        """Initialize k-d tree.
        
        Args:
            k: Number of dimensions
        """
        self.k = k
        self.root = None
    
    def insert(self, node_id: str, point: List[float]) -> None:
        """Insert node into k-d tree.
        
        Args:
            node_id: Ontological node ID
            point: k-dimensional point
        """
        self.root = self._insert(self.root, node_id, point, 0)
    
    def _insert(self, node: Optional[KDNode], node_id: str, point: List[float], depth: int) -> KDNode:
        """Internal recursive insertion function."""
        if node is None:
            return KDNode(node_id, point)
        
        axis = depth % self.k
        
        if point[axis] < node.point[axis]:
            node.left = self._insert(node.left, node_id, point, depth + 1)
        else:
            node.right = self._insert(node.right, node_id, point, depth + 1)
            
        return node
    
    def nearest_neighbor(self, point: List[float]) -> Optional[str]:
        """Find nearest neighbor to query point.
        
        Args:
            point: Query point
            
        Returns:
            ID of nearest node or None if tree is empty
        """
        if self.root is None:
            return None
            
        best = [None, float('inf')]  # [node_id, distance]
        self._nearest_neighbor(self.root, point, 0, best)
        return best[0]
    
    def _nearest_neighbor(self, node: KDNode, point: List[float], depth: int, best: List[Any]) -> None:
        """Internal recursive nearest neighbor search."""
        if node is None:
            return
            
        # Calculate distance to current node
        dist = sum((a - b) ** 2 for a, b in zip(node.point, point))
        
        # Update best if current is closer
        if dist < best[1]:
            best[0] = node.node_id
            best[1] = dist
        
        # Recursively search subtrees
        axis = depth % self.k
        diff = point[axis] - node.point[axis]
        
        # Search closest subtree first
        if diff < 0:
            self._nearest_neighbor(node.left, point, depth + 1, best)
            # Only search other subtree if it could contain closer points
            if diff ** 2 < best[1]:
                self._nearest_neighbor(node.right, point, depth + 1, best)
        else:
            self._nearest_neighbor(node.right, point, depth + 1, best)
            if diff ** 2 < best[1]:
                self._nearest_neighbor(node.left, point, depth + 1, best)
    
    def k_nearest_neighbors(self, point: List[float], k: int) -> List[Tuple[str, float]]:
        """Find k nearest neighbors to query point.
        
        Args:
            point: Query point
            k: Number of neighbors to return
            
        Returns:
            List of (node_id, distance) tuples
        """
        if self.root is None:
            return []
            
        nearest = []  # Priority queue of (-distance, node_id)
        self._k_nearest_neighbors(self.root, point, 0, nearest, k)
        
        # Convert to list of (node_id, distance) tuples
        result = [(nid, -dist) for dist, nid in nearest]
        result.sort(key=lambda x: x[1])
        return result
    
    def _k_nearest_neighbors(self, node: KDNode, point: List[float], depth: int, 
                            nearest: List[Tuple[float, str]], k: int) -> None:
        """Internal recursive k-nearest neighbors search."""
        if node is None:
            return
            
        # Calculate distance to current node
        dist = sum((a - b) ** 2 for a, b in zip(node.point, point))
        
        # Update nearest if current node qualifies
        if len(nearest) < k:
            heapq.heappush(nearest, (-dist, node.node_id))
        elif -dist > nearest[0][0]:  # If current dist is smaller than largest in heap
            heapq.heapreplace(nearest, (-dist, node.node_id))
        
        # Recursively search subtrees
        axis = depth % self.k
        diff = point[axis] - node.point[axis]
        
        # Search closest subtree first
        if diff < 0:
            self._k_nearest_neighbors(node.left, point, depth + 1, nearest, k)
            # Only search other subtree if it could contain closer points
            if len(nearest) < k or diff ** 2 < -nearest[0][0]:
                self._k_nearest_neighbors(node.right, point, depth + 1, nearest, k)
        else:
            self._k_nearest_neighbors(node.right, point, depth + 1, nearest, k)
            if len(nearest) < k or diff ** 2 < -nearest[0][0]:
                self._k_nearest_neighbors(node.left, point, depth + 1, nearest, k)

class FractalKnowledgeDatabase:
    """Main database class for ontological knowledge storage."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize ontological database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or ":memory:"
        self.conn = self._initialize_database()
        
        # In-memory indices
        self.trinity_index = KDTree(k=3)
        self.complex_index = KDTree(k=2)
        self.nodes = {}  # Cache of loaded nodes
    
    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        
        # Create tables if not exist
        with conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                trinity_vector TEXT NOT NULL,
                fractal_position TEXT NOT NULL,
                created_at REAL NOT NULL,
                parent_id TEXT,
                metadata TEXT
            )
            ''')
            
            conn.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL NOT NULL,
                metadata TEXT,
                PRIMARY KEY (source_id, target_id, relation_type)
            )
            ''')
            
            # Create indices
            conn.execute('CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id)')
        
        return conn
    
    def store_node(self, node: OntologicalNode) -> None:
        """Store ontological node in database.
        
        Args:
            node: Node to store
        """
        # Serialize node data
        with self.conn:
            self.conn.execute(
                'INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)',
                (
                    node.id,
                    node.query,
                    json.dumps(node.trinity_vector.serialize()),
                    json.dumps(node.fractal_position.serialize()),
                    node.created_at,
                    node.parent_id,
                    json.dumps(node.metadata)
                )
            )
        
        # Update parent-child relationship if parent exists
        if node.parent_id:
            parent = self.get_node(node.parent_id)
            if parent and node.id not in parent.children_ids:
                parent.children_ids.append(node.id)
                self.store_node(parent)
        
        # Update indices
        point_trinity = list(node.trinity_vector.as_tuple())
        point_complex = [node.fractal_position.c_real, node.fractal_position.c_imag]
        
        self.trinity_index.insert(node.id, point_trinity)
        self.complex_index.insert(node.id, point_complex)
        
        # Update cache
        self.nodes[node.id] = node
    
    def get_node(self, node_id: str) -> Optional[OntologicalNode]:
        """Retrieve node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node or None if not found
        """
        # Check cache first
        if node_id in self.nodes:
            return self.nodes[node_id]
        
        # Query database
        cursor = self.conn.execute(
            'SELECT query, trinity_vector, fractal_position, created_at, parent_id, metadata FROM nodes WHERE id = ?',
            (node_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return None
            
        # Deserialize data
        query, trinity_vector_json, fractal_position_json, created_at, parent_id, metadata_json = row
        
        trinity_vector = TrinityVector.deserialize(json.loads(trinity_vector_json))
        fractal_position = FractalPosition.deserialize(json.loads(fractal_position_json))
        metadata = json.loads(metadata_json) if metadata_json else {}
        
        # Find children
        children_cursor = self.conn.execute(
            'SELECT id FROM nodes WHERE parent_id = ?',
            (node_id,)
        )
        children_ids = [row[0] for row in children_cursor.fetchall()]
        
        # Create node
        node = OntologicalNode(
            id=node_id,
            query=query,
            trinity_vector=trinity_vector,
            fractal_position=fractal_position,
            created_at=created_at,
            parent_id=parent_id,
            children_ids=children_ids,
            metadata=metadata
        )
        
        # Update cache
        self.nodes[node_id] = node
        
        return node
    
    def create_node(self, 
                   query: str, 
                   trinity_vector: TrinityVector,
                   fractal_position: FractalPosition,
                   parent_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> OntologicalNode:
        """Create and store new ontological node.
        
        Args:
            query: Original query string
            trinity_vector: Trinity vector (𝔼-𝔾-𝕋)
            fractal_position: Position in fractal space
            parent_id: Optional parent node ID
            metadata: Optional additional metadata
            
        Returns:
            Created node
        """
        # Generate ID
        node_id = self._generate_id(query)
        
        # Create node
        node = OntologicalNode(
            id=node_id,
            query=query,
            trinity_vector=trinity_vector,
            fractal_position=fractal_position,
            created_at=time.time(),
            parent_id=parent_id,
            children_ids=[],
            metadata=metadata or {}
        )
        
        # Store node
        self.store_node(node)
        
        return node
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID for node based on content.
        
        Args:
            text: Text to hash
            
        Returns:
            Unique ID string
        """
        hash_input = f"{text}:{time.time()}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def add_relation(self, 
                    source_id: str, 
                    target_id: str, 
                    relation_type: str,
                    weight: float = 1.0,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add relation between nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relation
            weight: Relation strength
            metadata: Optional relation metadata
            
        Returns:
            True if successful, False otherwise
        """
        # Verify nodes exist
        if not self.get_node(source_id) or not self.get_node(target_id):
            return False
        
        # Store relation
        with self.conn:
            self.conn.execute(
                'INSERT OR REPLACE INTO relations VALUES (?, ?, ?, ?, ?)',
                (
                    source_id,
                    target_id,
                    relation_type,
                    weight,
                    json.dumps(metadata or {})
                )
            )
        
        return True
    
    def get_relations(self, 
                     node_id: str, 
                     relation_type: Optional[str] = None,
                     direction: str = "outgoing") -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Get node relations.
        
        Args:
            node_id: Node ID
            relation_type: Optional relation type filter
            direction: "outgoing", "incoming" or "both"
            
        Returns:
            List of (node_id, relation_type, weight, metadata) tuples
        """
        relations = []
        
        # Query outgoing relations
        if direction in ["outgoing", "both"]:
            query = 'SELECT target_id, relation_type, weight, metadata FROM relations WHERE source_id = ?'
            params = [node_id]
            
            if relation_type:
                query += ' AND relation_type = ?'
                params.append(relation_type)
            
            cursor = self.conn.execute(query, params)
            for row in cursor:
                target_id, rel_type, weight, metadata_json = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                relations.append((target_id, rel_type, weight, metadata))
        
        # Query incoming relations
        if direction in ["incoming", "both"]:
            query = 'SELECT source_id, relation_type, weight, metadata FROM relations WHERE target_id = ?'
            params = [node_id]
            
            if relation_type:
                query += ' AND relation_type = ?'
                params.append(relation_type)
            
            cursor = self.conn.execute(query, params)
            for row in cursor:
                source_id, rel_type, weight, metadata_json = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                relations.append((source_id, rel_type, weight, metadata))
        
        return relations
    
    def find_nearest_by_trinity(self, vector: TrinityVector, k: int = 5) -> List[Tuple[str, float]]:
        """Find nearest nodes by trinity vector.
        
        Args:
            vector: Trinity vector (𝔼-𝔾-𝕋)
            k: Number of nearest neighbors to return
            
        Returns:
            List of (node_id, distance) tuples
        """
        point = list(vector.as_tuple())
        return self.trinity_index.k_nearest_neighbors(point, k)
    
    def find_nearest_by_position(self, position: FractalPosition, k: int = 5) -> List[Tuple[str, float]]:
        """Find nearest nodes by fractal position.
        
        Args:
            position: Fractal position
            k: Number of nearest neighbors to return
            
        Returns:
            List of (node_id, distance) tuples
        """
        point = [position.c_real, position.c_imag]
        return self.complex_index.k_nearest_neighbors(point, k)
    
    def find_by_query(self, query_term: str, limit: int = 10) -> List[str]:
        """Find nodes by query text.
        
        Args:
            query_term: Search text
            limit: Maximum number of results
            
        Returns:
            List of matching node IDs
        """
        cursor = self.conn.execute(
            'SELECT id FROM nodes WHERE query LIKE ? LIMIT ?',
            (f'%{query_term}%', limit)
        )
        return [row[0] for row in cursor.fetchall()]
    
    def get_children(self, node_id: str) -> List[OntologicalNode]:
        """Get direct child nodes.
        
        Args:
            node_id: Parent node ID
            
        Returns:
            List of child nodes
        """
        cursor = self.conn.execute(
            'SELECT id FROM nodes WHERE parent_id = ?',
            (node_id,)
        )
        child_ids = [row[0] for row in cursor.fetchall()]
        return [self.get_node(child_id) for child_id in child_ids if self.get_node(child_id)]
    
    def get_ancestors(self, node_id: str, max_depth: int = -1) -> List[OntologicalNode]:
        """Get ancestor nodes.
        
        Args:
            node_id: Starting node ID
            max_depth: Maximum depth to traverse (-1 for unlimited)
            
        Returns:
            List of ancestor nodes
        """
        ancestors = []
        current_id = node_id
        depth = 0
        
        while True:
            node = self.get_node(current_id)
            if not node or not node.parent_id or (max_depth >= 0 and depth >= max_depth):
                break
                
            parent = self.get_node(node.parent_id)
            if parent:
                ancestors.append(parent)
                current_id = parent.id
                depth += 1
            else:
                break
                
        return ancestors
    
    def get_descendants(self, node_id: str, max_depth: int = -1) -> List[OntologicalNode]:
        """Get descendant nodes.
        
        Args:
            node_id: Starting node ID
            max_depth: Maximum depth to traverse (-1 for unlimited)
            
        Returns:
            List of descendant nodes
        """
        descendants = []
        to_visit = [(node_id, 0)]  # (node_id, depth)
        visited = set()
        
        while to_visit:
            current_id, depth = to_visit.pop(0)
            
            if current_id in visited or (max_depth >= 0 and depth > max_depth):
                continue
                
            visited.add(current_id)
            
            # Add children to visit queue
            children = self.get_children(current_id)
            descendants.extend(children)
            
            for child in children:
                to_visit.append((child.id, depth + 1))
                
        return descendants
    
    def create_entailment_chain(self, source_id: str, target_id: str, strength: float) -> bool:
        """Create logical entailment relation between nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            strength: Entailment strength
            
        Returns:
            True if successful, False otherwise
        """
        return self.add_relation(
            source_id=source_id,
            target_id=target_id,
            relation_type="entails",
            weight=strength,
            metadata={"creation_time": time.time()}
        )
    
    def get_entailments(self, node_id: str, recursive: bool = False, depth: int = 1) -> List[Tuple[str, float]]:
        """Get entailed nodes.
        
        Args:
            node_id: Starting node ID
            recursive: Whether to traverse entailment chains recursively
            depth: Recursion depth (only used if recursive=True)
            
        Returns:
            List of (node_id, strength) tuples
        """
        relations = self.get_relations(
            node_id=node_id,
            relation_type="entails",
            direction="outgoing"
        )
        
        entailments = [(rel[0], rel[2]) for rel in relations]  # (node_id, weight)
        
        if recursive and depth > 1:
            for ent_id, _ in entailments:
                deeper_entailments = self.get_entailments(ent_id, True, depth - 1)
                for deep_id, deep_strength in deeper_entailments:
                    # Avoid duplicates
                    if not any(e[0] == deep_id for e in entailments):
                        entailments.append((deep_id, deep_strength))
                        
        return entailments
    
    def apply_banach_tarski_decomposition(self, 
                                         node_id: str, 
                                         pieces: int = 2,
                                         relation_type: str = "decomposition") -> List[str]:
        """Apply Banach-Tarski decomposition to create paradoxical pieces.
        
        Args:
            node_id: Node to decompose
            pieces: Number of pieces
            relation_type: Relation type
            
        Returns:
            List of created piece node IDs
        """
        node = self.get_node(node_id)
        if not node:
            return []
            
        # Generate pieces through rotations
        piece_ids = []
        for i in range(pieces):
            angle = (i / pieces) * 2 * math.pi
            rotation = complex(math.cos(angle), math.sin(angle))
            
            # Rotate complex position
            original_c = node.fractal_position.complex
            rotated_c = original_c * rotation
            
            # Create new position
            new_position = FractalPosition(
                c_real=rotated_c.real,
                c_imag=rotated_c.imag,
                iterations=node.fractal_position.iterations,
                in_set=node.fractal_position.in_set,
                escape_radius=node.fractal_position.escape_radius
            )
            
            # Create piece node
            piece_node = self.create_node(
                query=f"Piece {i+1} of {node.query}",
                trinity_vector=node.trinity_vector,
                fractal_position=new_position,
                parent_id=node.id,
                metadata={
                    "decomposition_source": node_id,
                    "piece_index": i,
                    "rotation_angle": angle,
                    "is_decomposition_piece": True
                }
            )
            
            # Create relation
            self.add_relation(
                source_id=node_id,
                target_id=piece_node.id,
                relation_type=relation_type,
                weight=1.0,
                metadata={"piece_index": i, "angle": angle}
            )
            
            piece_ids.append(piece_node.id)
            
        return piece_ids
    
    def apply_banach_tarski_reassembly(self, 
                                      piece_ids: List[str], 
                                      copies: int = 2,
                                      relation_type: str = "reassembly") -> List[str]:
        """Apply Banach-Tarski reassembly to create copies.
        
        Args:
            piece_ids: Piece node IDs
            copies: Number of copies to create
            relation_type: Relation type
            
        Returns:
            List of created copy node IDs
        """
        # Verify all pieces exist
        pieces = [self.get_node(pid) for pid in piece_ids]
        if not all(pieces):
            return []
        
        # Find original node (parent of first piece)
        original_id = pieces[0].parent_id
        original = self.get_node(original_id) if original_id else None
        
        if not original:
            return []
            
        # Create copies with different rotations
        copy_ids = []
        for i in range(copies):
            angle = (i / copies) * 2 * math.pi
            rotation = complex(math.cos(angle), math.sin(angle))
            
            # Rotate complex position
            original_c = original.fractal_position.complex
            rotated_c = original_c * rotation
            
            # Create new position
            new_position = FractalPosition(
                c_real=rotated_c.real,
                c_imag=rotated_c.imag,
                iterations=original.fractal_position.iterations,
                in_set=original.fractal_position.in_set,
                escape_radius=original.fractal_position.escape_radius
            )
            
            # Create copy node
            copy_node = self.create_node(
                query=f"Copy {i+1} of {original.query}",
                trinity_vector=original.trinity_vector,
                fractal_position=new_position,
                parent_id=original.id,
                metadata={
                    "reassembly_source": original_id,
                    "copy_index": i,
                    "rotation_angle": angle,
                    "is_reassembly_copy": True,
                    "source_pieces": piece_ids
                }
            )
            
            # Create relation from original
            self.add_relation(
                source_id=original_id,
                target_id=copy_node.id,
                relation_type=relation_type,
                weight=1.0,
                metadata={
                    "copy_index": i, 
                    "angle": angle,
                    "source_pieces": piece_ids
                }
            )
            
            # Create relations from pieces
            for j, piece_id in enumerate(piece_ids):
                self.add_relation(
                    source_id=piece_id,
                    target_id=copy_node.id,
                    relation_type="contributes_to",
                    weight=1.0 / len(piece_ids),
                    metadata={"piece_index": j, "copy_index": i}
                )
            
            copy_ids.append(copy_node.id)
            
        return copy_ids
    
    def close(self) -> None:
        """Close database connection."""
        self.conn.close()


# Usage example
if __name__ == "__main__":
    # Initialize database
    db = FractalKnowledgeDatabase("thonoc_knowledge.db")
    
    # Create ontological nodes
    v1 = TrinityVector(existence=0.8, goodness=0.7, truth=0.9)
    p1 = FractalPosition(c_real=0.3, c_imag=0.2, iterations=50, in_set=False)
    node1 = db.create_node("Does goodness require existence?", v1, p1)
    
    v2 = TrinityVector(existence=0.9, goodness=0.6, truth=0.8)
    p2 = FractalPosition(c_real=0.31, c_imag=0.21, iterations=45, in_set=False)
    node2 = db.create_node("Can something exist without being good?", v2, p2)
    
    # Create entailment relation
    db.create_entailment_chain(node1.id, node2.id, 0.7)
    
    # Apply Banach-Tarski decomposition
    piece_ids = db.apply_banach_tarski_decomposition(node1.id)
    
    # Apply reassembly to create copies
    copy_ids = db.apply_banach_tarski_reassembly(piece_ids)
    
    # Find nearest nodes
    nearest = db.find_nearest_by_trinity(v1)
    
    print(f"Created node: {node1.id}")
    print(f"Entailed node: {node2.id}")
    print(f"Banach-Tarski pieces: {piece_ids}")
    print(f"Reassembled copies: {copy_ids}")
    print(f"Nearest nodes by trinity vector: {nearest}")
    
    # Close database
    db.close()