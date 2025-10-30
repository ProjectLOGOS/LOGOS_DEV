# logos_system/services/database/persistence_manager.py
import json
import os
from .db_core_logic import FractalDB, OntologicalNode, TrinityVector, FractalPosition

class PersistenceManager:
    """Handles auto-saving and auto-loading of the knowledge graph."""

    def __init__(self, db: FractalDB, preseed_data_dir: str = "logos_system/subsystems/telos/generative_tools/precede_nodes_and_data"):
        self.db = db
        self.preseed_data_dir = preseed_data_dir

    def populate_on_startup(self):
        """Loads all preseed nodes and saved nodes into the DB on startup."""
        print("--- [DB] Populating knowledge graph on startup ---")
        
        # 1. Load preseed nodes
        for filename in os.listdir(self.preseed_data_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.preseed_data_dir, filename), 'r') as f:
                    data = json.load(f)
                    # This is a placeholder for the logic to convert
                    # the preseed JSON into OntologicalNode objects and store them.
                    print(f"  - Loaded preseed file: {filename}")
        
        # 2. Load nodes from main persistent storage (if any)
        # The SQLite DB is already persistent, so this is automatic.
        # If using a JSON dump as backup, the logic would go here.
        node_count = self.db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        print(f"  - Verified {node_count} existing nodes in persistent storage.")
        print("--- [DB] Population complete ---")

    def save_on_shutdown(self):
        """
        Ensures all data is committed on shutdown. Can also create backups.
        """
        # For SQLite, the 'with self.conn:' statements already ensure commits.
        # This function is for creating a full JSON backup if desired.
        print("--- [DB] Shutdown protocol initiated. Committing final transactions. ---")
        self.db.conn.commit()
        print("--- [DB] All data saved. ---")