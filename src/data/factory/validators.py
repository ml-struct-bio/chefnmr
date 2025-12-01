import numpy as np
from typing import Any, Dict

class DataValidator:
    """Validates molecule data before storage."""
    
    @staticmethod
    def validate_atom_features(atom_features: Dict[str, Any]) -> bool:
        """Validate atom features data."""
        required_keys = [
            'atom_decoder', 'atom_coords',
            'atom_one_hot', 'atom_charges', 'atom_mask'
        ]
        for key in required_keys:
            if key not in atom_features:
                return False

        # atom_decoder: list of str
        atom_decoder = atom_features['atom_decoder']
        if not (isinstance(atom_decoder, list) and all(isinstance(x, str) for x in atom_decoder)):
            return False

        # atom_coords: (n_confs, max_n_atoms, 3) float32
        coords = atom_features['atom_coords']
        if not (isinstance(coords, np.ndarray) and coords.dtype == np.float32 and coords.ndim == 3 and coords.shape[0] > 0 and coords.shape[2] == 3 and not np.isnan(coords).any() and not np.isinf(coords).any() and not np.isneginf(coords).any()):
            return False
        # Check that each conformer is not all zeros (or below epsilon)
        if np.any(np.sum(np.abs(coords), axis=(1, 2)) < 1e-6):
            return False
        n_confs, max_n_atoms = coords.shape[0], coords.shape[1]

        # atom_one_hot: (max_n_atoms, n_atom_types) float32
        one_hot = atom_features['atom_one_hot']
        if not (isinstance(one_hot, np.ndarray) and one_hot.dtype == np.float32 and one_hot.ndim == 2 and one_hot.shape[0] == max_n_atoms and not np.isnan(one_hot).any()):
            return False

        # atom_charges: (max_n_atoms,) float32
        charges = atom_features['atom_charges']
        if not (isinstance(charges, np.ndarray) and charges.dtype == np.float32 and charges.ndim == 1 and charges.shape[0] == max_n_atoms and not np.isnan(charges).any()):
            return False

        # atom_mask: (max_n_atoms,) float32
        mask = atom_features['atom_mask']
        if not (isinstance(mask, np.ndarray) and mask.dtype == np.float32 and mask.ndim == 1 and mask.shape[0] == max_n_atoms and not np.isnan(mask).any()):
            return False

        return True
    
    @staticmethod
    def validate_spectra(spectra: Dict[str, Any]):
        """Validate spectra data. Returns (h_valid, c_valid)."""
        h_valid = True
        c_valid = True

        # H spectra
        h_keys = ['h_10k', 'h_28k', 'h_solvent', 'h_100']
        h_10k = spectra.get('h_10k', None)
        h_28k = spectra.get('h_28k', None)
        h_100 = spectra.get('h_100', None)
        h_solvent = spectra.get('h_solvent', None)
        valid_h_10k = isinstance(h_10k, np.ndarray) and h_10k.dtype == np.float32 and h_10k.ndim == 2 and h_10k.shape[0] > 0 and h_10k.shape[1] == 10000 and not np.isnan(h_10k).any()
        valid_h_28k = isinstance(h_28k, np.ndarray) and h_28k.dtype == np.float32 and h_28k.ndim == 2 and h_28k.shape[0] > 0 and h_28k.shape[1] == 28000 and not np.isnan(h_28k).any()
        valid_h_100 = isinstance(h_100, np.ndarray) and h_100.dtype == np.float32 and h_100.ndim == 2 and h_100.shape[0] > 0 and h_100.shape[1] == 100 and not np.isnan(h_100).any()
        if not (valid_h_10k or valid_h_28k or valid_h_100):
            h_valid = False

        # C spectra
        c_keys = ['c_80', 'c_10k', 'c_solvent']
        c_80 = spectra.get('c_80', None)
        c_10k = spectra.get('c_10k', None)
        c_solvent = spectra.get('c_solvent', None)
        valid_c_80 = isinstance(c_80, np.ndarray) and c_80.dtype == np.float32 and c_80.ndim == 2 and c_80.shape[0] > 0 and c_80.shape[1] == 80 and not np.isnan(c_80).any()
        valid_c_10k = isinstance(c_10k, np.ndarray) and c_10k.dtype == np.float32 and c_10k.ndim == 2 and c_10k.shape[0] > 0 and c_10k.shape[1] == 10000 and not np.isnan(c_10k).any()
        if not (valid_c_80 or valid_c_10k):
            c_valid = False

        return h_valid, c_valid
    
    @staticmethod
    def validate_molecule_data(mol_data: Dict[str, Any]):
        """Validate complete molecule data."""
        # Check SMILES
        if 'smiles' not in mol_data or not mol_data['smiles']:
            return False, False, False

        # Check atom features
        atom_valid = True
        if 'atom_features' not in mol_data or not mol_data['atom_features'] or not DataValidator.validate_atom_features(mol_data['atom_features']):
            atom_valid = False

        # Check spectra
        if 'spectra' not in mol_data or not mol_data['spectra']:
            h_valid, c_valid = False, False
        else:
            h_valid, c_valid = DataValidator.validate_spectra(mol_data['spectra'])
            
        return atom_valid, h_valid, c_valid
