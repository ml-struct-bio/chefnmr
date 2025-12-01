import numpy as np
from typing import Any, Dict, Optional
from src.data import utils
from .models import AtomFeatures, SpectraData

class DataProcessor:
    """Processes and transforms molecule data."""
    
    @staticmethod
    def process_atom_features(atom_features_dict: Dict[str, Any]) -> AtomFeatures:
        """Process raw atom features into AtomFeatures object."""
        atom_features = AtomFeatures(**atom_features_dict)
        
        # Apply collapse operation to remove masked atoms
        if atom_features.atom_coords is not None and atom_features.atom_coords.size > 0:
            collapsed_dict = utils.collapse_atom_features(atom_features.__dict__)
            atom_features = AtomFeatures(**collapsed_dict)
        
        return atom_features
    
    @staticmethod
    def process_spectra(spectra_dict: Dict[str, Any]) -> SpectraData:
        """Process raw spectra into SpectraData object."""
        return SpectraData(**spectra_dict)
    
    @staticmethod
    def standardize_data_types(data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize data types according to specifications."""
        result = data
        
        # Standardize spectra data
        if 'spectra' in result:
            for key, value in result['spectra'].items():
                if key.endswith('solvent'):
                    continue  # Skip solvent spectra
                else:
                    # Ensure numeric spectra are float32 arrays
                    if isinstance(value, np.ndarray) and value.dtype != np.float32:
                        result['spectra'][key] = value.astype(np.float32)
                    elif not isinstance(value, np.ndarray):
                        result['spectra'][key] = np.array(value, dtype=np.float32)
        
        # Standardize atom features
        if 'atom_features' in result:
            for key, value in result['atom_features'].items():
                if key in ['atom_decoder', 'max_iterations']:
                    continue
                elif key in ['atom_coords', 'atom_one_hot', 'atom_charges', 'atom_mask']:
                    if isinstance(value, np.ndarray) and value.dtype != np.float32:
                        result['atom_features'][key] = value.astype(np.float32)
                    elif not isinstance(value, np.ndarray):
                        result['atom_features'][key] = np.array(value, dtype=np.float32)
        
        return result
