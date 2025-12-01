import sqlite3
from rdkit import Chem
from threading import Lock
import os
from src.data import utils

try:
    CHEFNMR_DIR = os.environ["CHEFNMR_DIR"]
    GLOBAL_DB_PATH = f"{CHEFNMR_DIR}/meta/MOL_IDX.db"
except KeyError:
    raise EnvironmentError("Please set the CHEFNMR_DIR environment variable to the root directory of the ChefNMR project.")

class MOLIDXMapper:
    def __init__(self, db_path: str = GLOBAL_DB_PATH):
        # Check if the directory exists, create it if it doesn't
        db_dir = os.path.dirname(db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        self.conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        # WAL mode for concurrent readers/writer
        self.conn.execute("PRAGMA journal_mode = WAL;")
        # single table for mol_idx↔smiles
        self.conn.execute("""
          CREATE TABLE IF NOT EXISTS molecules (
            mol_idx INTEGER PRIMARY KEY AUTOINCREMENT,
            smiles  TEXT    UNIQUE NOT NULL
          );
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_smiles ON molecules(smiles);")
        self._lock = Lock()
        # in-memory cache
        self._smiles_to_mol_idx = {}
        self._mol_idx_to_smiles = {}
        self._load_cache()

    def _load_cache(self):
        cur = self.conn.execute("SELECT mol_idx, smiles FROM molecules;")
        for mol_idx, smiles in cur:
            self._mol_idx_to_smiles[mol_idx] = smiles
            self._smiles_to_mol_idx[smiles] = mol_idx

    def lookup_smiles_in_db(self, smiles: str) -> int | None:
        """Look up the canonical SMILES in the database directly, bypassing the cache."""
        can = utils.canonicalize(smiles)
        if not can:
            return None
        cur = self.conn.execute("SELECT mol_idx FROM molecules WHERE smiles = ?", (can,))
        row = cur.fetchone()
        return row[0] if row else None

    def add_smiles(self, smiles: str) -> int | None:
        can = utils.canonicalize(smiles)
        if not can:
            return None, None
        # fast in-memory check
        if can in self._smiles_to_mol_idx:
            return self._smiles_to_mol_idx[can], can

        # look up in db before insert
        mol_idx = self.lookup_smiles_in_db(can)
        # print("Lookup result for", can, ":", mol_idx)  # Debug print
        if mol_idx is not None:
            # update cache for future lookups
            self._smiles_to_mol_idx[can] = mol_idx
            self._mol_idx_to_smiles[mol_idx] = can
            return mol_idx, can

        with self._lock:  
            try:
                cur = self.conn.execute(
                    "INSERT INTO molecules(smiles) VALUES(?)", (can,)
                )
                mol_idx = cur.lastrowid
                self._smiles_to_mol_idx[can]   = mol_idx
                self._mol_idx_to_smiles[mol_idx] = can
                return mol_idx, can
            except sqlite3.IntegrityError:
                # another thread/process beat us to it
                return self.get_id(can), can

    def get_id(self, smiles: str) -> int | None:
        can = utils.canonicalize(smiles)
        if not can:
            return None
        mol_idx = self._smiles_to_mol_idx.get(can)
        if mol_idx is not None:
            return mol_idx
        # Not in cache, look up in db
        mol_idx = self.lookup_smiles_in_db(can)
        if mol_idx is not None:
            self._smiles_to_mol_idx[can] = mol_idx
            self._mol_idx_to_smiles[mol_idx] = can
        return mol_idx

    def lookup_idx_in_db(self, mol_idx: int) -> str | None:
        """Look up the SMILES for a given mol_idx in the database directly, bypassing the cache."""
        cur = self.conn.execute("SELECT smiles FROM molecules WHERE mol_idx = ?", (mol_idx,))
        row = cur.fetchone()
        return row[0] if row else None

    def get_smiles(self, mol_idx: int) -> str | None:
        smiles = self._mol_idx_to_smiles.get(mol_idx)
        if smiles is not None:
            return smiles
        # Not in cache, look up in db
        smiles = self.lookup_idx_in_db(mol_idx)
        if smiles is not None:
            self._mol_idx_to_smiles[mol_idx] = smiles
            self._smiles_to_mol_idx[smiles] = mol_idx
        return smiles

    def delete_by_id(self, mol_idx: int) -> bool:
        if mol_idx not in self._mol_idx_to_smiles:
            return False
        smiles = self._mol_idx_to_smiles.pop(mol_idx)
        self._smiles_to_mol_idx.pop(smiles, None)
        self.conn.execute("DELETE FROM molecules WHERE mol_idx = ?", (mol_idx,))
        return True

    def delete_by_smiles(self, smiles: str) -> bool:
        mol_idx = self.get_id(smiles)
        return self.delete_by_id(mol_idx) if mol_idx is not None else False
    
    def close(self):
        """Close the database connection and commit any pending transactions."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.commit()
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        """Context manager entry point."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point that ensures the database is closed properly."""
        self.close()
        
    def __del__(self):
        """Destructor to ensure database connection is closed."""
        self.close()