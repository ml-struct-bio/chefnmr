import pathlib
import os
import h5py
import numpy as np
import multiprocessing
from functools import partial
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm

from src.data.factory.mapper import MOLIDXMapper
from src.data import utils
from .models import SplitData
from .storage import HDF5StorageManager
from .processors import DataProcessor
from .validators import DataValidator
from .display import MoleculeDisplayer
from .computation import ComputationManager


class MoleculeFactory:
    """
    Main molecule factory with hierarchical HDF5 storage.
    
    This factory provides a clean interface for:
    - Querying molecules and metadata
    - Adding/updating molecules
    - Managing dataset splits
    - Computing dataset statistics
    
    Organizes data as follows:
    - molecules.h5                      
    - ├── <mol_idx> (as zero-padded 7-digit strings)
    - │   ├── scalar attributes: smiles (bytes), skeleton_smiles (bytes)
    - │   ├── spectra
    - │   │   ├── h_10k     (n_spectra_h, 10000) float32
    - │   │   ├── h_28k     (n_spectra_h, 28000) float32
    - │   │   ├── h_solvent list of (str) with len n_spectra_h
    - │   │   ├── c_80      (n_spectra_c, 80) float32
    - │   │   ├── c_10k     (n_spectra_c, 10000) float32
    - │   │   └── c_solvent list of (str) with len n_spectra_c
    - │   ├── atom_features
    - │   │   ├── attributes: max_iterations (int)
    - │   │   ├── atom_decoder (list of str) e.g. ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    - │   │   ├── atom_coords  (n_confs, max_n_atoms, 3) float32
    - │   │   ├── atom_one_hot (max_n_atoms, n_atom_types) float32
    - │   │   ├── atom_charges (max_n_atoms) float32
    - │   │   └── atom_mask    (max_n_atoms) float32
    - ├── valid_indices_h (int32)
    - ├── valid_indices_c (int32)
    - ├── split_indices{suffix1}
    - │   ├── train (n_train,) int32
    - │   ├── val   (n_val,)   int32
    - │   ├── test  (n_test,)  int32
    - │   └── attributes: sigma_data (float32), max_n_atoms (int32)
    - ├── split_indices{suffix2} ...
    """
    
    def __init__(self, datadir: str, mode: str = 'r', swmr: bool = True, db_path: str = None):
        """Initialize the molecule factory."""
        self.datadir = pathlib.Path(datadir)
        self.filepath = self.datadir / 'molecules.h5'
        self.mode = mode
        self.swmr = swmr
        self.mol_idx_mapper = MOLIDXMapper(db_path=db_path) if db_path else MOLIDXMapper()
        
        # Initialize components
        self._setup_hdf5_file()
        self.storage = HDF5StorageManager(self.file)
        self.processor = DataProcessor()
        self.validator = DataValidator()
        self.displayer = MoleculeDisplayer(self)
        self.computation = ComputationManager(self)
    
    def _setup_hdf5_file(self):
        """Setup HDF5 file with appropriate mode."""
        libver = 'latest'
        if self.mode == 'r':
            if not self.datadir.exists():
                raise FileNotFoundError(f"Data directory {self.datadir} does not exist.")
            self.file = h5py.File(self.filepath, self.mode, swmr=self.swmr, libver=libver)
        else:
            os.makedirs(self.datadir, exist_ok=True)
            self.file = h5py.File(self.filepath, self.mode, libver=libver)
            if self.swmr and hasattr(self.file, 'swmr_mode'):
                try:
                    self.file.swmr_mode = True
                except Exception:
                    self.swmr = False
    
    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, 'file') and self.file:
            self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        self.close()
    
    # Core utility methods
    def format_mol_idx(self, mol_idx) -> str:
        """Format molecule index as zero-padded 7-digit string."""
        if isinstance(mol_idx, str):
            return mol_idx
        elif isinstance(mol_idx, int):
            return f"{mol_idx:07d}"
        else:
            raise ValueError(f"Invalid molecule index: {mol_idx} with type {type(mol_idx)}")
    
    def int_mol_idx(self, mol_idx: Union[int, str]) -> int:
        """Convert molecule index to integer."""
        if isinstance(mol_idx, int):
            return mol_idx
        elif isinstance(mol_idx, str) and mol_idx.isdigit():
            return int(mol_idx)
        elif isinstance(mol_idx, np.integer):
            return int(mol_idx)
        else:
            raise ValueError(f"Invalid molecule index: {mol_idx} with type {type(mol_idx)}")
    
    def _validate_write_access(self):
        """Validate that the factory has write access."""
        if self.mode == 'r':
            raise ValueError(f"Cannot perform write operation in read-only mode ({self.mode})")
    
    def _flush_if_needed(self):
        """Flush HDF5 file if SWMR mode is enabled."""
        if self.swmr and hasattr(self.file, 'flush'):
            self.file.flush()
    
    # ===================
    # Query Methods
    # ===================
    
    def get_molecule_ids(self) -> List[str]:
        """Get list of all molecule IDs in the dataset."""
        return [key for key in self.file.keys() 
                if not key.startswith('split_indices') and not key.startswith('valid_indices')]
    
    def count_molecules(self) -> int:
        """Return the number of molecules in the dataset."""
        return len(self.get_molecule_ids())
    
    def molecule_exists(self, mol_idx: Union[int, str]) -> bool:
        """Check if a molecule exists."""
        if isinstance(mol_idx, int):
            mol_idx = self.format_mol_idx(mol_idx)
        return mol_idx in self.file
    
    def get_molecule_data_by_idx(self, mol_idx: Union[int, str], include_skeleton_smiles: bool = True,
                          include_spectra: bool = True, include_atom_features: bool = True) -> Dict[str, Any]:
        """Get molecule data by index."""
        mol_idx = self.format_mol_idx(self.int_mol_idx(mol_idx))
        mol_data = self.storage.load_molecule_data(mol_idx, include_skeleton_smiles,
                                                   include_spectra, include_atom_features)
        return self.processor.standardize_data_types(mol_data)
    
    def get_molecule_data_by_smiles(self, smiles: str, include_skeleton_smiles: bool = True,
                                    include_spectra: bool = True, include_atom_features: bool = True) -> Dict[str, Any]:
        
        mol_idx = self.mol_idx_mapper.lookup_smiles_in_db(smiles)
        if mol_idx is None:
            return {}
        return self.get_molecule_data_by_idx(mol_idx, include_skeleton_smiles, include_spectra, include_atom_features)
    
    # ===================
    # Add/Update Methods
    # ===================
    
    def add_molecule(self, smiles: str, atom_features: Optional[Dict[str, Any]] = None,
                    spectra: Optional[Dict[str, Any]] = None, flush: bool = True, 
                    operation: str = 'a') -> Optional[str]:
        """
        Add a molecule to the dataset.
        
        Args:
            operation: 'w' or 'a'
                - 'w': add new molecule or rewrite existing data
                - 'a': add new molecule or append new atom coords or spectra to existing data
        """
        self._validate_write_access()
        
        # Get or create molecule index
        mol_idx, canonical_smiles = self.mol_idx_mapper.add_smiles(smiles)
        if canonical_smiles is None:
            print(f"Invalid SMILES: {smiles}")
            return None
        
        mol_idx_str = self.format_mol_idx(mol_idx)
        molecule_exists = mol_idx_str in self.file
        
        if operation not in ['w', 'a']:
            raise ValueError(f"Invalid operation: {operation}. Use 'w' for write or 'a' for append.")
        
        if not molecule_exists:
            skeleton_smiles = utils.canonicalize(canonical_smiles, remove_stereo=True)
            mol_group = self.storage.create_molecule_group(mol_idx_str, canonical_smiles, skeleton_smiles)
        else:
            mol_group = self.file[mol_idx_str]
            existing_smiles = self.storage._read_str(mol_group.attrs.get('smiles', b''))
            if canonical_smiles != existing_smiles:
                raise ValueError(f"Mol ({mol_idx_str}, {existing_smiles}) exists in the dataset. Different from new ({mol_idx_str}, {canonical_smiles})")
        
        if atom_features is not None:
            processed_features = self.processor.process_atom_features(atom_features)
            self.storage.store_atom_features(mol_group, processed_features, operation=operation)
        
        if spectra is not None:
            processed_spectra = self.processor.process_spectra(spectra)
            self.storage.store_spectra(mol_group, processed_spectra, operation=operation)
        
        if flush:
            self._flush_if_needed()
        
        return mol_idx_str
    
    def add_molecules_batch(self, molecules: List[Dict[str, Any]], operation: str) -> List[Optional[str]]:
        """Add multiple molecules in batch."""
        self._validate_write_access()
        
        results = []
        for mol in molecules:
            mol_idx = self.add_molecule(
                mol.get('smiles'),
                mol.get('atom_features'),
                mol.get('spectra'),
                flush=False,
                operation=operation
            )
            results.append(mol_idx)
        
        self._flush_if_needed()
        return results
    
    def delete_molecule_by_idx(self, mol_idx: Union[int, str], flush: bool = True) -> bool:
        self._validate_write_access()
        
        mol_idx_str = self.format_mol_idx(mol_idx)
        molecule_exists = mol_idx_str in self.file
        
        if not molecule_exists:
            return False
        else:
            mol_group = self.file[mol_idx_str]
            state = self.storage.delete_molecule_group(mol_group)
            
        if flush:
            self._flush_if_needed()
            
        return state
        
    def delete_molecules_batch(self, mol_idxs: List[Union[int, str]], flush: bool = True) -> List[bool]:
        """Delete multiple molecules in batch."""
        self._validate_write_access()
        
        results = []
        for mol_idx in mol_idxs:
            state = self.delete_molecule_by_idx(mol_idx, flush=False)
            results.append(state)
        
        if flush:
            self._flush_if_needed()
        
        return results
    
    # ===================
    # Valid Molecule Idx Management
    # ===================
    def validate_molecule_data_by_idx(self, mol_idx: Union[int, str]):
        mol_data = self.get_molecule_data_by_idx(mol_idx)
        return self.validator.validate_molecule_data(mol_data)
        
    def set_valid_indices(self, num_workers: Optional[int] = None) -> None:
        """Set valid indices for atom features and spectra."""
        self._validate_write_access()
        
        mol_ids = self.get_molecule_ids()
        if not mol_ids:
            print("No molecules found in dataset")
            return
        
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
        
        if num_workers == 1:
            # Single-threaded processing
            valid_indices_h = []
            valid_indices_c = []
            for mol_id in tqdm(mol_ids, desc="Setting valid indices", unit="molecules"):
                is_valid_atom, is_valid_spectra_h, is_valid_spectra_c = self.validate_molecule_data_by_idx(mol_id)
                if not is_valid_atom:
                    continue
                if is_valid_spectra_h:
                    valid_indices_h.append(self.int_mol_idx(mol_id))
                if is_valid_spectra_c:
                    valid_indices_c.append(self.int_mol_idx(mol_id))
        else:
            # Multi-threaded processing
            chunk_size = max(1, len(mol_ids) // (num_workers * 4))
            chunks = [mol_ids[i:i + chunk_size] for i in range(0, len(mol_ids), chunk_size)]
            
            from .workers import process_chunk_validation
            worker_func = partial(
                process_chunk_validation,
                factory_datadir=str(self.datadir),
                factory_mode='r',
                factory_swmr=True,
                factory_db_path=getattr(self.mol_idx_mapper, 'db_path', None)
            )
            
            with multiprocessing.Pool(num_workers) as pool:
                results = list(tqdm(
                    pool.imap(worker_func, chunks),
                    total=len(chunks),
                    desc="Setting valid indices",
                    unit="chunks"
                ))
            
            # Combine results from all chunks
            valid_indices_h = []
            valid_indices_c = []
            for result in results:
                valid_indices_h.extend(result['h'])
                valid_indices_c.extend(result['c'])
        
        # Store valid indices
        valid_indices_h = np.array(valid_indices_h, dtype=np.int32)
        valid_indices_c = np.array(valid_indices_c, dtype=np.int32)
        print(f"Found {len(valid_indices_h)} valid molecules with H spectra and {len(valid_indices_c)} valid molecules with C spectra.")
        self.storage.store_valid_indices('valid_indices_h', valid_indices_h)
        self.storage.store_valid_indices('valid_indices_c', valid_indices_c)
            
    
    def get_valid_indices(self, spectra_type: str = 'h') -> np.ndarray:
        """Get valid (atom feature and spectra) indices for a given spectra type ('h', 'c', or 'both')."""
        if spectra_type == 'h':
            return self.storage.load_valid_indices('valid_indices_h')
        elif spectra_type == 'c':
            return self.storage.load_valid_indices('valid_indices_c')
        elif spectra_type == 'both':
            valid_h = self.storage.load_valid_indices('valid_indices_h')
            valid_c = self.storage.load_valid_indices('valid_indices_c')
            return np.intersect1d(valid_h, valid_c)
        else:
            raise ValueError(f"Invalid spectra type: {spectra_type}. Use 'h', 'c', or 'both'.")
        
    
    # ===================
    # Split Management
    # ===================
    
    def get_split_names(self) -> List[str]:
        """Get all split names."""
        return [key for key in self.file.keys() if key.startswith('split_indices')]
    
    def get_split(self, suffix: str = '', recompute_sigma_data: bool = False,
                 recompute_max_n_atoms: bool = False) -> Dict[str, Any]:
        """Get dataset split."""
        split_name = f'split_indices{suffix}'
        split_data = self.storage.load_split(split_name)
        
        result = {
            'train': split_data.train,
            'val': split_data.val,
            'test': split_data.test,
        }
        
        # Compute or retrieve sigma_data
        if recompute_sigma_data or split_data.sigma_data is None:
            mol_indices = np.concatenate((split_data.train, split_data.val, split_data.test))
            result['sigma_data'] = round(self.computation.compute_sigma_data(mol_indices), 2)
        else:
            result['sigma_data'] = round(float(split_data.sigma_data), 2)
        
        # Compute or retrieve max_n_atoms
        if recompute_max_n_atoms or split_data.max_n_atoms is None:
            mol_indices = np.concatenate((split_data.train, split_data.val, split_data.test))
            result['max_n_atoms'] = self.computation.compute_max_n_atoms(mol_indices)
        else:
            result['max_n_atoms'] = int(split_data.max_n_atoms)
        
        return result
    
    def set_split(self, train_indices: np.ndarray, val_indices: np.ndarray,
                 test_indices: np.ndarray, suffix: str = '') -> None:
        """Set dataset split."""
        self._validate_write_access()
        
        split_data = SplitData(train_indices, val_indices, test_indices)
        
        # Compute metrics if requested
        mol_indices = np.concatenate((train_indices, val_indices, test_indices))
        split_data.sigma_data = round(self.computation.compute_sigma_data(mol_indices), 2)
        split_data.max_n_atoms = self.computation.compute_max_n_atoms(mol_indices)
        
        split_name = f'split_indices{suffix}'
        self.storage.store_split(split_name, split_data)
        self._flush_if_needed()
    
    def in_which_split(self, mol_idx: Union[int, str], suffix: str = '') -> str:
        """Check if a molecule index is in the specified split."""
        mol_idx = self.int_mol_idx(mol_idx)
        split_name = f'split_indices{suffix}'
        
        if split_name not in self.file:
            raise ValueError(f"Split '{split_name}' does not exist.")
        
        split_data = self.storage.load_split(split_name)
        
        if mol_idx in split_data.train:
            return 'train'
        elif mol_idx in split_data.val:
            return 'val'
        elif mol_idx in split_data.test:
            return 'test'
        else:
            return 'none'
    
    # ===================
    # Computation Methods (Delegated)
    # ===================
    
    def compute_sigma_data(self, mol_indices: List[int], num_workers: Optional[int] = None) -> float:
        """Compute sigma data using multiprocessing."""
        return self.computation.compute_sigma_data(mol_indices, num_workers)
    
    def compute_max_n_atoms(self, mol_indices: List[int], num_workers: Optional[int] = None, remove_h: bool = False) -> int:
        """Compute maximum number of atoms using multiprocessing."""
        return self.computation.compute_max_n_atoms(mol_indices, num_workers, remove_h)
    
    def compute_n_atoms_distribution(self, mol_indices: List[int], num_workers: Optional[int] = None, remove_h: bool = False) -> Dict[str, Any]:
        """Compute the distribution of number of atoms using multiprocessing."""
        return self.computation.compute_n_atoms_distribution(mol_indices, num_workers, remove_h)
    
    def compute_atom_types(self, mol_indices: List[int], num_workers: Optional[int] = None) -> List[str]:
        """Compute unique atom types across molecules using multiprocessing."""
        return self.computation.compute_atom_types(mol_indices, num_workers)
    
    def compute_solvents_distribution(self, mol_indices: List[int], num_workers: Optional[int] = None, 
                                     solvent_type: Optional[str] = None) -> Dict[str, Any]:
        """Compute the distribution of solvents using multiprocessing."""
        return self.computation.compute_solvents_distribution(mol_indices, num_workers, solvent_type)
    
    # ===================
    # Display Methods (Delegated)
    # ===================
    
    def display_molecule_by_idx(self, mol_idx: Union[int, str], verbose: bool = True) -> None:
        """Display molecule information."""
        self.displayer.display_molecule_by_idx(mol_idx, verbose)
    
    def display_valid_indices(self) -> None:
        self.displayer.display_valid_indices()
        
    def display_split(self, suffix: str = '') -> None:
        """Display split information."""
        self.displayer.display_split(suffix)
    
    def display_statistics(self, verbose: bool = True) -> None:
        """Display dataset statistics."""
        self.displayer.display_statistics(verbose)
    
    def visualize_molecule_by_idx(self, mol_idx: Union[int, str], save_dir: str = None) -> None:
        """Visualize molecule by index."""
        self.displayer.visualize_molecule_by_idx(mol_idx, save_dir)

