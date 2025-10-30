# logos_system/services/database/db_service.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from .persistence_manager import PersistenceManager
from ...core.data_structures import TrinityVector, FractalPosition
from .db_core_logic import FractalDB, OntologicalNode

db_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    global db_instance
    db_instance = FractalDB(db_path="data/logos_knowledge.db")
    persistence = PersistenceManager(db_instance)
    persistence.populate_on_startup()
    yield
    # This code runs on shutdown
    persistence.save_on_shutdown()
    db_instance.conn.close()

app = FastAPI(title="ARCHON Database Service", version="1.0", lifespan=lifespan)

@app.post("/node/store")
def store_node_endpoint(node_data: dict):
    try:
        node = OntologicalNode.deserialize(node_data)
        db_instance.store(node)
        return {"status": "success", "node_id": node.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store node: {e}")

@app.get("/node/get/{node_id}")
def get_node_endpoint(node_id: str):
    node = db_instance.get(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node.serialize()

@app.post("/search/trinity")
def find_nearest_by_trinity_endpoint(search_data: dict):
    vector_data = search_data.get("vector")
    k = search_data.get("k", 5)
    if not vector_data:
        raise HTTPException(status_code=400, detail="Search requires a 'vector' field.")
    
    vector = TrinityVector.deserialize(vector_data)
    results = db_instance.trinity_idx.k_nearest(list(vector.as_tuple()), k)
    return {"status": "success", "results": results}

# Add other endpoints for search, relations, etc. as needed