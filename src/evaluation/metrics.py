from typing import List, Tuple, Optional, Union
import rdkit.Chem as Chem
import torch
import numpy as np
from src.data import utils as data_utils
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from einops import einsum

def compute_matching_accuracy(smiles_list: List[Optional[str]]) -> float:
    """
    Compute the percentage of successfully reconstructed SMILES strings.

    Args:
        smiles_list (List[Optional[str]]): List of SMILES strings or None.

    Returns:
        float: Percentage of non-None SMILES.
    """
    matched_smiles_list = [smiles for smiles in smiles_list if smiles is not None]
    if not smiles_list:
        return 0.0
    matching_accuracy = len(matched_smiles_list) * 100.0 / len(smiles_list)
    return matching_accuracy

def compute_validity(smiles_list: List[Optional[str]]) -> float:
    """
    Compute the validity of SMILES strings.

    Args:
        smiles_list (List[Optional[str]]): List of SMILES strings.

    Returns:
        float: Percentage of valid (non-None) SMILES.
    """
    # compute the number of non-None SMILES strings
    valid_smiles_list = [smiles for smiles in smiles_list if smiles is not None]
    if not smiles_list:
        return 0.0
    validity = len(valid_smiles_list) * 100.0 / len(smiles_list)
    return validity

def match_smiles(
    target_smiles: str,
    reconstructed_smiles: str,
    remove_stereo: bool = True,
) -> bool:
    """
    Match target SMILES with reconstructed SMILES, allowing for relaxed matching.

    Args:
        target_smiles (str): Ground truth SMILES.
        reconstructed_smiles (str): Predicted SMILES.
        remove_stereo (bool): If True, ignores stereochemistry during matching.

    Returns:
        bool: True if molecules match, False otherwise.
    """
    try:
        target_skeleton_smiles = data_utils.canonicalize(target_smiles, remove_stereo=True)
        reconstructed_skeleton_smiles = data_utils.canonicalize(reconstructed_smiles, remove_stereo=True)
        
        if target_skeleton_smiles != reconstructed_skeleton_smiles:
            return False
            
        if remove_stereo:
            return True
        else: 
            # check full match including stereochemistry
            mol_target = Chem.MolFromSmiles(target_smiles)
            mol_reconstructed = Chem.MolFromSmiles(reconstructed_smiles)
            if mol_reconstructed.HasSubstructMatch(mol_target, useChirality=True):
                return True
    except Exception:
        return False
    return False

def smiles_to_similarity(
    target_smiles: List[str],
    reconstructed_smiles: List[str],
    multiplicity: int,
) -> List[float]:
    """
    Compute Tanimoto similarity between target and reconstructed molecules using Morgan fingerprints.

    Args:
        target_smiles (List[str]): List of target SMILES strings.
        reconstructed_smiles (List[str]): List of reconstructed SMILES strings.
        multiplicity (int): Number of reconstructions per target.

    Returns:
        List[float]: List of maximum similarity scores for each target.
    """
    target_smiles_arr = np.array(target_smiles)
    reconstructed_smiles_arr = np.array(reconstructed_smiles).reshape(-1, multiplicity)
    
    similarity_list = np.zeros((len(target_smiles_arr), multiplicity))
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    for i, target in enumerate(target_smiles_arr):
        target_mol = Chem.MolFromSmiles(target)
        if target_mol is None:
            continue
        target_fp = mfpgen.GetFingerprint(target_mol)
        
        for j, reconstructed in enumerate(reconstructed_smiles_arr[i]):
            try:
                reconstructed_mol = Chem.MolFromSmiles(reconstructed)
                if reconstructed_mol is None:
                    similarity_list[i, j] = 0.0
                    continue
                reconstructed_fp = mfpgen.GetFingerprint(reconstructed_mol)
                similarity = DataStructs.TanimotoSimilarity(target_fp, reconstructed_fp)
                similarity_list[i, j] = similarity
            except Exception:
                similarity_list[i, j] = 0.0
    
    # Compute the max similarity for each target (flattened list)
    return list(similarity_list.reshape(-1))

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: np.ndarray, first vector
        vec2: np.ndarray, second vector
        
    Returns:
        float, cosine similarity
    """
    if len(vec1) == 0 or len(vec2) == 0:
        return 0.0
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same shape for cosine similarity.")
    
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return float(np.dot(vec1, vec2) / (norm_vec1 * norm_vec2))

def compute_rmsd(
    predicted_coords: torch.Tensor,
    all_target_coords: torch.Tensor,
    conformer_mask: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_one_hot: torch.Tensor,
    atom_decoder: List[str],
    multiplicity: int,
    num_workers: int = 12,
) -> Tuple[List[float], List[float]]:
    """
    Compute RMSD between target and predicted coordinates.
    
    Args:
        predicted_coords: shape [batch_size * multiplicity, n_atoms, 3]
        all_target_coords: shape [batch_size, n_conformers, n_atoms, 3] - all ground truth conformers
        conformer_mask: shape [batch_size, n_conformers] - mask indicating valid conformers
        atom_mask: shape [batch_size, n_atoms]
        atom_one_hot: shape [batch_size, n_atoms, n_atom_types]
        atom_decoder: List of atom types mapping indices to symbols
        multiplicity: int, number of predicted samples per molecule
        num_workers: int, unused in current implementation but kept for API compatibility
        
    Returns:
        rmsd_list: List[float], RMSD for each sample (Average Minimum RMSD if all_target_coords provided)
        rmsd_wo_h_list: List[float], RMSD without hydrogens for each sample
    """
    batch_size = all_target_coords.shape[0]
    n_atoms = all_target_coords.shape[2]
    predicted_coords = predicted_coords.reshape(-1, multiplicity, n_atoms, 3)
    
    hydrogen_index = atom_decoder.index("H") if "H" in atom_decoder else -1
    atom_type = torch.argmax(atom_one_hot, dim=2)
    
    rmsd_list = torch.zeros((batch_size, multiplicity))
    rmsd_wo_h_list = torch.zeros((batch_size, multiplicity))
    
    for i in range(batch_size):
        atom_mask_i = atom_mask[i]
        atom_type_i = atom_type[i]
        hydrogen_mask = (atom_type_i == hydrogen_index).float()
        heavy_atom_mask = atom_mask_i * (1 - hydrogen_mask)
        
        # Get all ground truth conformers for this molecule
        target_conformers = all_target_coords[i]  # [n_conformers, n_atoms, 3]
        valid_conformers_mask = conformer_mask[i].bool() # [n_conformers]
        target_conformers = target_conformers[valid_conformers_mask]  # Only use valid conformers
        
        for j, predicted in enumerate(predicted_coords[i]):
            # Compute RMSD against all ground truth conformers and take minimum
            min_rmsd = float('inf')
            min_rmsd_wo_h = float('inf')
            
            for target_conformer in target_conformers:
                # Compute RMSD with all atoms
                # ! TODO: support batch alignment to speed up
                aligned_target = weighted_rigid_align(
                    target_conformer.unsqueeze(0),
                    predicted.unsqueeze(0),
                    torch.ones_like(atom_mask_i.unsqueeze(0)),
                    atom_mask_i.unsqueeze(0) 
                )
                # Remove batch dimension to match predicted shape
                aligned_target = aligned_target.squeeze(0)
                
                # Calculate squared differences
                diff = (aligned_target - predicted) ** 2
                # Sum over coordinates (x, y, z)
                diff = torch.sum(diff, dim=1)
                # Apply atom mask to only consider valid atoms
                masked_diff = diff * atom_mask_i
                # Calculate RMSD: sqrt(mean of masked squared differences)
                n_valid_atoms = torch.sum(atom_mask_i)
                rmsd = torch.sqrt(torch.sum(masked_diff) / n_valid_atoms)
                min_rmsd = min(min_rmsd, rmsd.item())
                
                # Compute RMSD without hydrogens
                aligned_target_wo_h = weighted_rigid_align(
                    target_conformer.unsqueeze(0),
                    predicted.unsqueeze(0),
                    torch.ones_like(heavy_atom_mask.unsqueeze(0)),
                    heavy_atom_mask.unsqueeze(0)
                )
                # Remove batch dimension to match predicted shape
                aligned_target_wo_h = aligned_target_wo_h.squeeze(0)
                
                # Calculate squared differences for heavy atoms only
                diff_wo_h = (aligned_target_wo_h - predicted) ** 2
                diff_wo_h = torch.sum(diff_wo_h, dim=1)
                masked_diff_wo_h = diff_wo_h * heavy_atom_mask
                n_valid_heavy_atoms = torch.sum(heavy_atom_mask)
                rmsd_wo_h = torch.sqrt(torch.sum(masked_diff_wo_h) / n_valid_heavy_atoms)
                min_rmsd_wo_h = min(min_rmsd_wo_h, rmsd_wo_h.item())
            
            # Store the minimum RMSD across all conformers
            rmsd_list[i, j] = min_rmsd
            rmsd_wo_h_list[i, j] = min_rmsd_wo_h
    
    # Flatten and return as list
    return rmsd_list.reshape(-1).tolist(), rmsd_wo_h_list.reshape(-1).tolist()

# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang
def weighted_rigid_align(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute weighted alignment between two sets of coordinates.

    Args:
        true_coords (torch.Tensor): The ground truth atom coordinates [batch, n_points, 3].
        pred_coords (torch.Tensor): The predicted atom coordinates [batch, n_points, 3].
        weights (torch.Tensor): The weights for alignment [batch, n_points].
        mask (torch.Tensor): The atoms mask [batch, n_points].

    Returns:
        torch.Tensor: Aligned coordinates [batch, n_points, 3].
    """

    batch_size, num_points, dim = true_coords.shape
    weights = (mask * weights).unsqueeze(-1)

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix
    cov_matrix = einsum(
        weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
    )

    # Compute the SVD of the covariance matrix, required float32 for svd and determinant
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    U, S, V = torch.linalg.svd(
        cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
    )
    V = V.mH

    # # Catch ambiguous rotation by checking the magnitude of singular values
    # if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
    #     print(
    #         "Warning: Excessively low rank of "
    #         + "cross-correlation between aligned point clouds. "
    #         + "`WeightedRigidAlign` cannot return a unique rotation."
    #     )

    # Compute the rotation matrix
    rot_matrix = torch.einsum("b i j, b k j -> b i k", U, V).to(dtype=torch.float32)

    # Ensure proper rotation matrix with determinant 1
    F = torch.eye(dim, dtype=cov_matrix_32.dtype, device=cov_matrix.device)[
        None
    ].repeat(batch_size, 1, 1)
    F[:, -1, -1] = torch.det(rot_matrix)
    rot_matrix = einsum(U, F, V, "b i j, b j k, b l k -> b i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    # Apply the rotation and translation
    aligned_coords = (
        einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j")
        + pred_centroid
    )
    aligned_coords.detach_()

    return aligned_coords
