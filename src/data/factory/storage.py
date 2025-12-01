import h5py
import numpy as np
from typing import Any, Dict, List, Optional, Union
from .models import AtomFeatures, SpectraData, SplitData

class HDF5StorageManager:
    """Manages HDF5 storage operations."""
    
    def __init__(self, file_handle: h5py.File):
        self.file = file_handle
    
    def _str_to_bytes(self, val) -> bytes:
        """Convert a string to bytes."""
        if isinstance(val, str):
            return val.encode("utf-8")
        elif isinstance(val, bytes):
            return val
        elif val is None:
            return b''
        else:
            raise TypeError(f"Expected str or bytes, got {type(val)}")
    
    def _read_str(self, val) -> str:
        """Convert bytes to string."""
        if isinstance(val, bytes):
            return val.decode("utf-8")
        if isinstance(val, str):
            return val
        raise TypeError(f"Expected bytes or str, got {type(val)}")
    
    def create_molecule_group(self, mol_idx: str, smiles: str, skeleton_smiles: str) -> h5py.Group:
        """Create a new molecule group."""
        mol_group = self.file.create_group(mol_idx)
        mol_group.attrs['smiles'] = self._str_to_bytes(smiles)
        mol_group.attrs['skeleton_smiles'] = self._str_to_bytes(skeleton_smiles)
        return mol_group
    
    def delete_molecule_group(self, mol_group_or_idx: Union[str, h5py.Group]) -> bool:
        """
        Delete a molecule group and all its sub-datasets.
        
        Args:
            mol_group_or_idx: Either the molecule index/identifier (string) or the HDF5 group object to delete
            
        Returns:
            bool: True if the group was deleted, False if it didn't exist
        """
        if isinstance(mol_group_or_idx, h5py.Group):
            # If passed a group object, get its name and delete from parent
            group_name = mol_group_or_idx.name.split('/')[-1]  # Get the last part of the path
            if group_name in self.file:
                del self.file[group_name]
                return True
            return False
        else:
            # If passed a string identifier
            if mol_group_or_idx not in self.file:
                return False
            
            del self.file[mol_group_or_idx]
            return True
    
    def store_atom_features(self, mol_group: h5py.Group, atom_features: AtomFeatures, operation: str):
        """
        Store atom features in molecule group.
        
        Args:
            operation: 'w' or 'a'
                - 'w': create new atom features group or reset existing data
                - 'a': append new atom coordinates or reset max_iterations
        """
        
        if 'atom_features' not in mol_group:
            features_group = mol_group.create_group('atom_features')
            self._reset_atom_features(features_group, atom_features) # Regardless of operation, we create a new group
        else:
            features_group = mol_group['atom_features']
            if operation == 'w':
                self._reset_atom_features(features_group, atom_features)
            elif operation == 'a':
                self._append_atom_features(features_group, atom_features)
            else:
                raise ValueError(f"Invalid operation: {operation}. Use 'w' to reset or 'a' to append.")
        
    
    def _reset_atom_features(self, features_group: h5py.Group, atom_features: AtomFeatures):
        """Reset (replace) atom features data."""
        for key, value in atom_features.__dict__.items():
            if value is None:
                continue
        
            # Store atom coordinates, one-hot encodings, charges, and masks
            if key in ['atom_coords', 'atom_one_hot', 'atom_charges', 'atom_mask']:
                if key in features_group:
                    if features_group[key].shape == value.shape and features_group[key].dtype == value.dtype:
                        features_group[key][...] = value
                    else:
                        del features_group[key]
                        features_group.create_dataset(key, data=value)
                else:
                    features_group.create_dataset(key, data=value)
        
            # Store atom decoder
            if key == 'atom_decoder':
                decoder_bytes = [self._str_to_bytes(s) for s in value]
                if 'atom_decoder' in features_group:
                    features_group['atom_decoder'][...] = decoder_bytes
                else:
                    features_group.create_dataset('atom_decoder', data=decoder_bytes)
            
            # Store max_iterations as attribute
            if key == 'max_iterations':
                features_group.attrs['max_iterations'] = int(value)
    
    def _append_atom_features(self, features_group: h5py.Group, atom_features: AtomFeatures):
        """Append new conformations to existing atom coords, and reset max_iterations."""
        new_coords = atom_features.atom_coords
        
        if new_coords is not None:
            if 'atom_coords' in features_group:
                existing_coords = features_group['atom_coords'][()]
                # Check shape compatibility (should have same max_n_atoms and 3D coords)
                if existing_coords.shape[1:] != new_coords.shape[1:]:
                    raise ValueError(f"Shape mismatch: existing {existing_coords.shape[1:]} vs new {new_coords.shape[1:]}")
                
                # Concatenate along conformation axis
                combined_coords = np.concatenate([existing_coords, new_coords], axis=0)
                del features_group['atom_coords']
                features_group.create_dataset('atom_coords', data=combined_coords)
            else:
                features_group.create_dataset('atom_coords', data=new_coords)
        
        # For other features except for max_iterations, we don't change
        # For max_iterations, we reset it if provided
        if atom_features.max_iterations is not None:
            features_group.attrs['max_iterations'] = int(atom_features.max_iterations)

    def store_spectra(self, mol_group: h5py.Group, spectra: SpectraData, operation: str):
        """
        Store spectra in molecule group.
        
        Args:
            operation: 'w' or 'a'
                - 'w': create new spectra group or reset existing data
                - 'a': append new spectra
        """
        if 'spectra' not in mol_group:
            spectra_group = mol_group.create_group('spectra')
            self._reset_spectra(spectra_group, spectra)
        else:
            spectra_group = mol_group['spectra']
            if operation == 'w':
                self._reset_spectra(spectra_group, spectra)
            elif operation == 'a':
                self._append_spectra(spectra_group, spectra)
            else:
                raise ValueError(f"Invalid operation: {operation}. Use 'w' to reset or 'a' to append.")
    
    def _reset_spectra(self, spectra_group: h5py.Group, spectra: SpectraData):
        """Reset (replace) spectra data."""
        for key, value in spectra.__dict__.items():
            if value is None:
                continue
                
            if key.endswith('solvent'):
                value_bytes = [self._str_to_bytes(s) for s in value]
                if key in spectra_group:
                    del spectra_group[key]
                spectra_group.create_dataset(key, data=value_bytes)
            else:
                if key in spectra_group:
                    if spectra_group[key].shape == value.shape and spectra_group[key].dtype == value.dtype:
                        spectra_group[key][...] = value
                    else:
                        del spectra_group[key]
                        spectra_group.create_dataset(key, data=value)
                else:
                    spectra_group.create_dataset(key, data=value)
    
    def _append_spectra(self, spectra_group: h5py.Group, spectra: SpectraData):
        """Append new spectra to existing data."""
        for key, value in spectra.__dict__.items():
            if value is None:
                continue
            
            if key in spectra_group:
                existing_data = spectra_group[key][()]
                
                if key.endswith('solvent'):
                    # Append solvent list
                    existing_solvents = [self._read_str(s) for s in existing_data]
                    new_solvents = existing_solvents + list(value)
                    value_bytes = [self._str_to_bytes(s) for s in new_solvents]
                    del spectra_group[key]
                    spectra_group.create_dataset(key, data=value_bytes)
                else:
                    # Append spectral data along first axis (n_spectra)
                    if existing_data.shape[1:] != value.shape[1:]:
                        raise ValueError(f"Shape mismatch for {key}: existing {existing_data.shape[1:]} vs new {value.shape[1:]}")
                    
                    combined_data = np.concatenate([existing_data, value], axis=0)
                    del spectra_group[key]
                    spectra_group.create_dataset(key, data=combined_data)
            else:
                # First time adding this type of spectra
                if key.endswith('solvent'):
                    value_bytes = [self._str_to_bytes(s) for s in value]
                    spectra_group.create_dataset(key, data=value_bytes)
                else:
                    spectra_group.create_dataset(key, data=value)

    def load_molecule_data(self, mol_idx: str, include_skeleton_smiles: bool = True,
                          include_spectra: bool = True, include_atom_features: bool = True) -> Dict[str, Any]:
        """Load molecule data from HDF5."""
        if mol_idx not in self.file:
            return {}
        
        mol_group = self.file[mol_idx]
        result = {
            'smiles': self._read_str(mol_group.attrs.get('smiles', b'')),
        }
        
        if include_skeleton_smiles:
            result['skeleton_smiles'] = self._read_str(mol_group.attrs.get('skeleton_smiles', b''))
        
        if include_spectra and 'spectra' in mol_group:
            result['spectra'] = {}
            spectra_group = mol_group['spectra']
            for key in ['h_10k', 'h_28k', 'h_100', 'h_solvent', 'c_80', 'c_10k', 'c_solvent']:
                if key in spectra_group:
                    if key.endswith('solvent'):
                        result['spectra'][key] = [self._read_str(s) for s in spectra_group[key][()]]
                    else:
                        result['spectra'][key] = spectra_group[key][()]
        
        if include_atom_features and 'atom_features' in mol_group:
            features_group = mol_group['atom_features']
            result['atom_features'] = {}
            if 'atom_decoder' in features_group:
                result['atom_features']['atom_decoder'] = [self._read_str(s) for s in features_group['atom_decoder'][()]]
            for key in ['atom_coords', 'atom_one_hot', 'atom_charges', 'atom_mask']:
                if key in features_group:
                    result['atom_features'][key] = features_group[key][()]
            if 'max_iterations' in features_group.attrs:
                result['atom_features']['max_iterations'] = features_group.attrs['max_iterations']
        
        return result
    
    def store_split(self, split_name: str, split_data: SplitData):
        """Store dataset split."""
        if split_name in self.file:
            del self.file[split_name]
        
        split_group = self.file.create_group(split_name)
        split_group.create_dataset('train', data=split_data.train)
        split_group.create_dataset('val', data=split_data.val)
        split_group.create_dataset('test', data=split_data.test)
        
        if split_data.sigma_data is not None:
            split_group.attrs['sigma_data'] = round(float(split_data.sigma_data), 2)
        if split_data.max_n_atoms is not None:
            split_group.attrs['max_n_atoms'] = int(split_data.max_n_atoms)
    
    def load_split(self, split_name: str) -> SplitData:
        """Load dataset split."""
        if split_name not in self.file:
            raise KeyError(f"Split {split_name} not found")
        
        split_group = self.file[split_name]
        return SplitData(
            train=split_group['train'][()],
            val=split_group['val'][()],
            test=split_group['test'][()],
            sigma_data=split_group.attrs.get('sigma_data'),
            max_n_atoms=split_group.attrs.get('max_n_atoms')
        )
    
    def store_valid_indices(self, name: str, indices: np.ndarray):
        """
        Store valid indices array in the HDF5 file.
        Overwrites if the dataset already exists.
        """
        if name not in ['valid_indices_h', 'valid_indices_c']:
            raise ValueError("Name must be either 'valid_indices_h' or 'valid_indices_c'")
        if name in self.file:
            del self.file[name]
        self.file.create_dataset(name, data=indices)

    def load_valid_indices(self, name: str) -> np.ndarray:
        """
        Load valid indices array from the HDF5 file.
        Raises KeyError if the dataset does not exist.
        """
        if name not in ['valid_indices_h', 'valid_indices_c']:
            raise ValueError("Name must be either 'valid_indices_h' or 'valid_indices_c'")
        if name not in self.file:
            raise KeyError(f"Valid indices {name} not found in dataset")
        return self.file[name][()]