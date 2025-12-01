import numpy as np
from typing import List, Optional, Dict, Any

def process_chunk_sigma_data(mol_idx_chunk: List[int], factory_datadir: str, 
                           factory_mode: str, factory_swmr: bool, 
                           factory_db_path: Optional[str]) -> np.ndarray:
    """Worker function to process a chunk of molecules for sigma data computation."""
    from .factory import MoleculeFactory  # Import here to avoid circular imports
    
    with MoleculeFactory(datadir=factory_datadir, mode=factory_mode, 
                        swmr=factory_swmr, db_path=factory_db_path) as factory:
        chunk_coords = np.empty(0, dtype=np.float32)
        for mol_idx in mol_idx_chunk:
            try:
                mol_data = factory.get_molecule_data_by_idx(
                    mol_idx, 
                    include_skeleton_smiles=False, 
                    include_spectra=False, 
                    include_atom_features=True
                )
                if 'atom_features' in mol_data and 'atom_coords' in mol_data['atom_features']:
                    atom_coords = mol_data['atom_features']['atom_coords']
                    chunk_coords = np.concatenate((chunk_coords, atom_coords.flatten()))
            except Exception as e:
                print(f"Error processing molecule {mol_idx}: {str(e)}")
        return chunk_coords

def process_chunk_max_n_atoms(mol_idx_chunk: List[int], factory_datadir: str,
                             factory_mode: str, factory_swmr: bool,
                             factory_db_path: Optional[str], remove_h: bool) -> int:
    """Worker function to process a chunk of molecules for max atoms computation."""
    from .factory import MoleculeFactory  # Import here to avoid circular imports
    from src.data import utils
    
    with MoleculeFactory(datadir=factory_datadir, mode=factory_mode, 
                        swmr=factory_swmr, db_path=factory_db_path) as factory:
        max_atoms_in_chunk = 0
        for mol_idx in mol_idx_chunk:
            try:
                mol_data = factory.get_molecule_data_by_idx(
                    mol_idx, 
                    include_skeleton_smiles=False, 
                    include_spectra=False, 
                    include_atom_features=False
                )
                smiles = mol_data['smiles']
                n_atoms = utils.n_atoms_in_smiles(smiles, remove_h=remove_h)
                max_atoms_in_chunk = max(max_atoms_in_chunk, n_atoms)
            except Exception as e:
                print(f"Error processing molecule {mol_idx}: {str(e)}")
        return max_atoms_in_chunk

def process_chunk_validation(mol_idx_chunk: List[str], factory_datadir: str,
                           factory_mode: str, factory_swmr: bool,
                           factory_db_path: Optional[str]) -> Dict[str, List[int]]:
    """Worker function to process a chunk of molecules for validation."""
    from .factory import MoleculeFactory  # Import here to avoid circular imports
    
    with MoleculeFactory(datadir=factory_datadir, mode=factory_mode, 
                        swmr=factory_swmr, db_path=factory_db_path) as factory:
        valid_indices_h = []
        valid_indices_c = []
        
        for mol_id in mol_idx_chunk:
            try:
                mol_data = factory.get_molecule_data_by_idx(mol_id)
                is_valid_atom, is_valid_spectra_h, is_valid_spectra_c = factory.validator.validate_molecule_data(mol_data)
                if not is_valid_atom:
                    continue
                if is_valid_spectra_h:
                    valid_indices_h.append(factory.int_mol_idx(mol_id))
                if is_valid_spectra_c:
                    valid_indices_c.append(factory.int_mol_idx(mol_id))
            except Exception as e:
                print(f"Error validating molecule {mol_id}: {str(e)}")
        
        return {'h': valid_indices_h, 'c': valid_indices_c}

def process_chunk_n_atoms_distribution(mol_idx_chunk: List[int], factory_datadir: str,
                                     factory_mode: str, factory_swmr: bool,
                                     factory_db_path: Optional[str], remove_h: bool) -> List[int]:
    """Worker function to process a chunk of molecules for n_atoms distribution computation."""
    from .factory import MoleculeFactory  # Import here to avoid circular imports
    from src.data import utils
    
    with MoleculeFactory(datadir=factory_datadir, mode=factory_mode, 
                        swmr=factory_swmr, db_path=factory_db_path) as factory:
        n_atoms_list = []
        for mol_idx in mol_idx_chunk:
            try:
                mol_data = factory.get_molecule_data_by_idx(
                    mol_idx, 
                    include_skeleton_smiles=False, 
                    include_spectra=False, 
                    include_atom_features=False
                )
                smiles = mol_data['smiles']
                n_atoms = utils.n_atoms_in_smiles(smiles, remove_h=remove_h)
                n_atoms_list.append(n_atoms)
            except Exception as e:
                print(f"Error processing molecule {mol_idx}: {str(e)}")
        return n_atoms_list

def process_chunk_atom_types(mol_idx_chunk: List[int], factory_datadir: str,
                           factory_mode: str, factory_swmr: bool,
                           factory_db_path: Optional[str]) -> List[str]:
    """Worker function to process a chunk of molecules for atom types computation."""
    from .factory import MoleculeFactory  # Import here to avoid circular imports
    from rdkit import Chem
    
    with MoleculeFactory(datadir=factory_datadir, mode=factory_mode, 
                        swmr=factory_swmr, db_path=factory_db_path) as factory:
        atom_types = set()
        for mol_idx in mol_idx_chunk:
            try:
                mol_data = factory.get_molecule_data_by_idx(
                    mol_idx, 
                    include_skeleton_smiles=False, 
                    include_spectra=False, 
                    include_atom_features=False
                )
                smiles = mol_data['smiles']
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mol = Chem.AddHs(mol)  # Add hydrogens
                    for atom in mol.GetAtoms():
                        atom_types.add(atom.GetSymbol())
                        
            except Exception as e:
                print(f"Error processing molecule {mol_idx}: {str(e)}")
        return list(atom_types)

def process_chunk_solvents(mol_idx_chunk: List[int], factory_datadir: str,
                         factory_mode: str, factory_swmr: bool,
                         factory_db_path: Optional[str], solvent_type: Optional[str] = None) -> List[str]:
    """Worker function to process a chunk of molecules for solvents computation."""
    from .factory import MoleculeFactory  # Import here to avoid circular imports
    
    with MoleculeFactory(datadir=factory_datadir, mode=factory_mode, 
                        swmr=factory_swmr, db_path=factory_db_path) as factory:
        solvents = []  # Use list to preserve duplicates for counting
        for mol_idx in mol_idx_chunk:
            try:
                mol_data = factory.get_molecule_data_by_idx(
                    mol_idx, 
                    include_skeleton_smiles=False, 
                    include_spectra=True, 
                    include_atom_features=False
                )
                if 'spectra' in mol_data:
                    spectra = mol_data['spectra']
                    # Check for solvent information in spectra data
                    for key, value in spectra.items():
                        # Filter by solvent_type if specified
                        if solvent_type and not key.endswith(solvent_type):
                            continue
                        if key.endswith('solvent') and value:
                            if isinstance(value, list):
                                solvents.extend(value)
                            else:
                                solvents.append(str(value))
            except Exception as e:
                print(f"Error processing molecule {mol_idx}: {str(e)}")
        return solvents
