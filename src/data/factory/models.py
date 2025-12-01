from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np

@dataclass
class AtomFeatures:
    """Data class for atom features."""
    atom_coords: np.ndarray = None
    atom_one_hot: np.ndarray = None
    atom_charges: np.ndarray = None
    atom_mask: np.ndarray = None
    atom_decoder: List[str] = None
    max_iterations: Optional[int] = None
    
    def __post_init__(self):
        """Validate and standardize data types."""
        self.atom_coords = np.array(self.atom_coords, dtype=np.float32)
        self.atom_one_hot = np.array(self.atom_one_hot, dtype=np.float32)
        self.atom_charges = np.array(self.atom_charges, dtype=np.float32)
        self.atom_mask = np.array(self.atom_mask, dtype=np.float32)
        
        if self.atom_coords.ndim == 2:
            self.atom_coords = self.atom_coords.reshape((1, self.atom_coords.shape[0], 3))

@dataclass
class SpectraData:
    """Data class for spectral data."""
    h_10k: Optional[np.ndarray] = None
    h_28k: Optional[np.ndarray] = None
    h_100: Optional[np.ndarray] = None
    h_solvent: Optional[List[str]] = None
    c_80: Optional[np.ndarray] = None
    c_10k: Optional[np.ndarray] = None
    c_solvent: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate and standardize data types."""
        for key in ['h_10k', 'h_28k', 'h_100', 'c_80', 'c_10k']:
            value = getattr(self, key)
            if value is not None:
                value = np.array(value, dtype=np.float32)
                if value.ndim == 1:
                    value = value.reshape((1, -1))
                setattr(self, key, value)

@dataclass
class MoleculeData:
    """Data class for complete molecule data."""
    smiles: str
    skeleton_smiles: Optional[str] = None
    atom_features: Optional[AtomFeatures] = None
    spectra: Optional[SpectraData] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {'smiles': self.smiles}
        if self.skeleton_smiles:
            result['skeleton_smiles'] = self.skeleton_smiles
        if self.atom_features:
            result['atom_features'] = {
                'atom_coords': self.atom_features.atom_coords,
                'atom_one_hot': self.atom_features.atom_one_hot,
                'atom_charges': self.atom_features.atom_charges,
                'atom_mask': self.atom_features.atom_mask,
                'atom_decoder': self.atom_features.atom_decoder,
                'max_iterations': self.atom_features.max_iterations
            }
        if self.spectra:
            result['spectra'] = {k: v for k, v in self.spectra.__dict__.items() if v is not None}
        return result

@dataclass
class SplitData:
    """Data class for dataset splits."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    sigma_data: Optional[float] = None
    max_n_atoms: Optional[int] = None
    
    def __post_init__(self):
        """Ensure arrays are int32."""
        self.train = np.array(self.train, dtype=np.int32)
        self.val = np.array(self.val, dtype=np.int32)
        self.test = np.array(self.test, dtype=np.int32)
