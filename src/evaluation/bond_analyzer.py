import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import torch
from rdkit import RDLogger, Chem
from rdkit.Chem import rdDetermineBonds
from src.data import utils as data_utils

def get_smiles_from_mol(mol: Chem.Mol, remove_stereo: bool) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts canonical SMILES and the SMILES of the largest fragment from an RDKit molecule.

    Args:
        mol (Chem.Mol): The RDKit molecule object.
        remove_stereo (bool): Whether to remove stereochemistry information.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing:
            - full_smiles: The canonical SMILES of the full molecule.
            - largest_fragment_smiles: The canonical SMILES of the largest fragment (if applicable).
            Returns (None, None) if an error occurs.
    """
    try:
        mol_no_h = Chem.RemoveHs(mol)
        if remove_stereo:
            Chem.RemoveStereochemistry(mol_no_h)
        
        full_smiles = Chem.MolToSmiles(mol_no_h)
        full_smiles = Chem.CanonSmiles(full_smiles)
        
        fragments = Chem.rdmolops.GetMolFrags(mol_no_h, asMols=True, sanitizeFrags=True)
        if len(fragments) > 1:
            largest_fragment = max(fragments, key=lambda m: m.GetNumAtoms())
            Chem.SanitizeMol(largest_fragment)
            largest_fragment_smiles = Chem.MolToSmiles(largest_fragment)
            largest_fragment_smiles = Chem.CanonSmiles(largest_fragment_smiles)
            return (None, largest_fragment_smiles)
        else:
            return (full_smiles, full_smiles)
    except Exception:
        return (None, None)


def determine_bond_with_timeout(mol_block: str, remove_stereo: bool, timeout_seconds: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Determines bonds for a molecule from a MolBlock with a timeout safeguard.
    
    This function runs RDKit's DetermineBonds in a separate process to prevent
    hanging on very complex or malformed molecules.
    
    Note: This is a temporary workaround with expensive process overhead.
    TODO: Upgrade RDKit to new version with timeout support. See https://github.com/rdkit/rdkit/pull/8548 

    Args:
        mol_block (str): The molecule block representation.
        remove_stereo (bool): Whether to remove stereochemistry.
        timeout_seconds (int): Maximum execution time in seconds.

    Returns:
        Tuple[Optional[str], Optional[str]]: (full_smiles, largest_fragment_smiles) or (None, None) on failure/timeout.
    """
    def worker_target(result_queue, mol_block, remove_stereo):
        """Internal worker function to run in a separate process."""
        try:
            mol = Chem.MolFromMolBlock(mol_block, sanitize=False)
            rdDetermineBonds.DetermineBonds(mol)
            result = get_smiles_from_mol(mol, remove_stereo)
            result_queue.put(result)
        except Exception:
            result_queue.put((None, None))
    
    # Create a queue to receive the result
    result_queue = mp.Queue()
    worker_process = mp.Process(target=worker_target, args=(result_queue, mol_block, remove_stereo))
    
    # Start the process and wait for the specified timeout
    worker_process.start()
    worker_process.join(timeout=timeout_seconds)
    
    if worker_process.is_alive():
        # If the process is still running, terminate it
        worker_process.terminate()
        worker_process.join(timeout=1)  # Allow grace period for cleanup
        
        if worker_process.is_alive():
            worker_process.kill() # Force kill if necessary
            worker_process.join()
            
        return (None, None)
    
    # Retrieve the result from the queue
    try:
        result = result_queue.get_nowait()
        return result
    except Exception:
        return (None, None)


def atom_features_to_smiles_one(args: Tuple) -> Tuple[int, Optional[str], Optional[str]]:
    """
    Worker function to process a single molecule: converts atom features to SMILES.

    Args:
        args (Tuple): A tuple containing all necessary parameters:
            (molecule_index, atom_coords, atom_one_hot, atom_mask, dataset_name, 
             remove_h, remove_stereo, atom_decoder, timeout_seconds)

    Returns:
        Tuple[int, Optional[str], Optional[str]]: (molecule_index, full_smiles, largest_fragment_smiles)
    """
    (molecule_index, atom_coords, atom_one_hot, atom_mask, dataset_name, 
     remove_h, remove_stereo, atom_decoder, timeout_seconds) = args
    
    # Reconstruct the molecule object
    molecule = data_utils.Molecule(
        atom_coords=atom_coords,
        atom_one_hot=atom_one_hot,
        atom_mask=atom_mask,
        atom_decoder=atom_decoder,
        remove_h=remove_h,
        collapse=True
    )
    
    if remove_h:
        # Currently not supported because bond determination requires hydrogens
        raise NotImplementedError("Remove_h=True case is not supported for SMILES reconstruction.")
    else:
        if timeout_seconds == 0:
            # Fast path: Direct processing in the current process
            try:
                rdkit_mol = molecule.to_rdkit_molecule()
                rdDetermineBonds.DetermineBonds(rdkit_mol)
                reconstructed_smiles, largest_fragment_smiles = get_smiles_from_mol(rdkit_mol, remove_stereo)
            except Exception:
                reconstructed_smiles = None
                largest_fragment_smiles = None
        else:
            # Safe path: Process with timeout protection for potentially problematic geometries
            try:
                rdkit_mol = molecule.to_rdkit_molecule()
                mol_block = Chem.MolToMolBlock(rdkit_mol)
                reconstructed_smiles, largest_fragment_smiles = determine_bond_with_timeout(
                    mol_block, remove_stereo, timeout_seconds
                )
            except Exception:
                reconstructed_smiles, largest_fragment_smiles = (None, None)
    
    return molecule_index, reconstructed_smiles, largest_fragment_smiles


def atom_features_to_smiles(
    dataset_name: str,
    atom_features: Dict[str, Any], 
    multiplicity: int,
    remove_h: bool,
    remove_stereo: bool,
    atom_decoder: List[str],
    timeout_seconds: int = 2,
    num_workers: Optional[int] = None
) -> Tuple[List[Optional[str]], List[Optional[str]]]:
    """
    Converts a batch of atom features into SMILES strings using parallel processing.
    
    Handles bond determination for molecules with hydrogens, including timeout management
    for complex cases.

    Args:
        dataset_name (str): Name of the dataset (affects processing strategy).
        atom_features (Dict[str, Any]): Dictionary containing 'atom_coords', 'atom_one_hot', 'atom_mask'.
        multiplicity (int): Number of conformers/views per molecule to process.
        remove_h (bool): Whether hydrogens were removed in the input features.
        remove_stereo (bool): Whether to strip stereochemistry from output SMILES.
        atom_decoder (List[str]): Mapping from one-hot indices to atom types.
        timeout_seconds (int, optional): Timeout for bond determination per molecule. Defaults to 2.
        num_workers (int, optional): Number of parallel workers. Defaults to CPU count.

    Returns:
        Tuple[List[Optional[str]], List[Optional[str]]]: 
            - List of reconstructed full SMILES.
            - List of reconstructed largest fragment SMILES.
    """
    # Helper to convert tensor/list to numpy array
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.cpu().detach().numpy()
        return np.array(data)

    atom_coords_list = to_numpy(atom_features["atom_coords"])
    atom_one_hot_list = to_numpy(atom_features["atom_one_hot"])
    atom_mask_list = to_numpy(atom_features["atom_mask"])
    
    # Repeat features for multiplicity (e.g., if multiple samples per molecule)
    atom_one_hot_list = np.repeat(atom_one_hot_list, multiplicity, axis=0)
    atom_mask_list = np.repeat(atom_mask_list, multiplicity, axis=0)
    
    # Suppress RDKit logs during bulk processing
    RDLogger.DisableLog('rdApp.*')
    
    # Prepare arguments for each molecule
    args_list = [
        (i, coords, one_hot, mask, dataset_name, remove_h, remove_stereo, atom_decoder, timeout_seconds)
        for i, (coords, one_hot, mask) in enumerate(zip(atom_coords_list, atom_one_hot_list, atom_mask_list))
    ]
    
    # Determine number of workers
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    
    # Initialize results containers
    reconstructed_smiles_list = [None] * len(args_list)
    reconstructed_largest_fragment_smiles_list = [None] * len(args_list)
    
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            # Calculate chunksize for better distribution
            chunksize = max(1, len(args_list) // (num_workers * 4))
            
            results = list(pool.map(atom_features_to_smiles_one, args_list, chunksize=chunksize))
            
            # Aggregate results
            for i, reconstructed_smiles, largest_frag_smiles in results:
                reconstructed_smiles_list[i] = reconstructed_smiles
                reconstructed_largest_fragment_smiles_list[i] = largest_frag_smiles
                
    except Exception as e:
        raise
    
    return reconstructed_smiles_list, reconstructed_largest_fragment_smiles_list