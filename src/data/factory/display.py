import numpy as np
from typing import Any, Dict, List, Union
from src.data import utils
import os
class MoleculeDisplayer:
    """Handles display and statistics for molecules."""
    
    def __init__(self, factory):
        self.factory = factory
    
    def display_molecule_by_idx(self, mol_idx: Union[int, str], verbose: bool = True) -> None:
        """Display a formatted representation of a molecule."""
        mol_data = self.factory.get_molecule_data_by_idx(mol_idx)
        lines = []
        lines.append("-" * 50)
        lines.append(f"Molecule ID: {mol_idx}")
        lines.append(f"  SMILES: {mol_data['smiles']}")
        
        if mol_data.get('skeleton_smiles'):
            lines.append(f"  Skeleton SMILES: {mol_data['skeleton_smiles']}")
        
        # Format atom features
        if 'atom_features' in mol_data and mol_data['atom_features']:
            lines.append("  Atom Features:")
            for key, value in mol_data['atom_features'].items():
                if key in ['atom_decoder', 'max_iterations']:
                    lines.append(f"    {key}: {value}")
                elif isinstance(value, np.ndarray):
                    shape_str = f"shape={value.shape}, dtype={value.dtype}"
                    if verbose and value.size > 0:
                        if value.ndim == 1:
                            lines.append(f"    {key}: {shape_str}, value[:5]={value[:5]}")
                        elif value.ndim == 2:
                            lines.append(f"    {key}: {shape_str}, value[0]={value[0]}")
                        elif value.ndim == 3:
                            lines.append(f"    {key}: {shape_str}, value[0, 0]={value[0, 0]}")
                    else:
                        lines.append(f"    {key}: {shape_str}")
                else:
                    lines.append(f"    {key}: {type(value)}")
        
        # Format spectra
        if 'spectra' in mol_data and mol_data['spectra']:
            lines.append("  Spectra:")
            for key, value in mol_data['spectra'].items():
                if key.endswith('solvent'):
                    lines.append(f"    {key}: {value}")
                elif isinstance(value, np.ndarray):
                    shape_str = f"shape={value.shape}, dtype={value.dtype}"
                    if verbose and value.size > 0:
                        if value.ndim == 1:
                            lines.append(f"    {key}: {shape_str}, value[:5]={value[:5]}")
                        elif value.ndim == 2:
                            lines.append(f"    {key}: {shape_str}, value[0]={value[0]}")
                    else:
                        lines.append(f"    {key}: {shape_str}")
                else:
                    lines.append(f"    {key}: {type(value)}")
        
        lines.append("-" * 50)
        print("\n".join(lines))
    
    def visualize_molecule_by_idx(self, mol_idx: Union[int, str], save_dir: str = None) -> None:
        """Visualize a molecule SMILES and spectra by index."""
        mol_data = self.factory.get_molecule_data_by_idx(mol_idx)
        smiles = mol_data.get('smiles', '')
        spectra = mol_data.get('spectra', {})
        atom_features = mol_data.get('atom_features', {})
        
        if not smiles:
            print(f"No SMILES found for molecule ID {mol_idx}")
            return
        
        # Prepare save path
        os.makedirs(save_dir, exist_ok=True)
        
        # Visualize SMILES
        utils.smiles_to_svg(smiles, svg_path=os.path.join(save_dir, f"{mol_idx}_smiles.svg"))
        
        # Visualize spectra if available
        ppm_range_map = {
            'h_10k': (10, -2),
            'h_28k': (12, -2),
            'h_100': (10, -2),
            'c_80': (231.3, 3.42),
            'c_10k': (230, -20)
        }
        color_map = {
            'h_10k': '#225391',
            'h_28k': '#225391',
            'h_100': '#225391',
            'c_80': '#39836f',
            'c_10k': '#39836f'
        }
        if spectra:
            for key in ['h_10k', 'h_28k', 'h_100', 'c_80', 'c_10k']:
                if key in spectra:
                    spectrum = spectra[key]
                    ppm_range = ppm_range_map.get(key, (0, 0))
                    color = color_map.get(key, 'black')
                    for i in range(spectrum.shape[0]):
                        spectra_type = key.split('_')[0]
                        solvent = spectra.get(f"{spectra_type}_solvent", [])
                        if solvent and i < len(solvent):
                            solvent_name = solvent[i]
                        else:
                            solvent_name = ''
                        utils.plot_nmr_spectrum(
                            spectrum[i],
                            ppm_range=ppm_range,
                            xlabel='Chemical Shift (ppm)',
                            ascending=False,
                            figsize=(6, 4),
                            dpi=300,
                            color=color,
                            is_carbon=('c' in key),
                            save_path=os.path.join(save_dir, f"{mol_idx}_{key}_{i}_{solvent_name}.png") 
                        )
                        
        if atom_features:
            mol = utils.Molecule(
                atom_coords=atom_features.get('atom_coords')[0],
                atom_one_hot=atom_features.get('atom_one_hot'),
                atom_charges=None,
                atom_mask=atom_features.get('atom_mask'),
                atom_decoder=atom_features.get('atom_decoder'),
                remove_h='H' not in atom_features.get('atom_decoder', []),
                collapse=True
            )
            
            # Save the molecule as an .xyz file
            xyz_file_path = os.path.join(save_dir, f"{mol_idx}_conf0.xyz")
            mol.save_as_xyz_file(xyz_file_path)
    
    def display_split(self, suffix: str = '') -> None:
        """Display split information."""
        try:
            result = self.factory.get_split(suffix)
            if not result:
                print(f"No split data found for suffix '{suffix}'")
                return
            
            lines = []
            lines.append(f"split_indices{suffix}:")
            size = result['train'].shape[0] + result['val'].shape[0] + result['test'].shape[0]
            lines.append(f"  Total size: {size}")
            for key, value in result.items():
                if key in ['sigma_data', 'max_n_atoms']:
                    lines.append(f"  {key}: {value}")
                else:
                    lines.append(f"  {key}: {value.shape}, value[:10]={value[:10] if len(value) > 10 else value}")
            print("\n".join(lines))
        except KeyError as e:
            print(f"Split not found: {e}")
    
    def display_statistics(self, verbose: bool = True) -> None:
        """Display comprehensive statistics about the dataset."""
        print("-" * 50)
        print("MoleculeFactory HDF5 Statistics:")
        print(f"File: {self.factory.filepath}")
        print(f"Number of molecules: {self.factory.count_molecules()}")
        self.display_valid_indices()
        
        split_names = self.factory.get_split_names()
        print(f"Splits: {split_names if split_names else 'None'}")
        
        if split_names:
            print("Sample splits:")
            for name in split_names:
                sample_splits = name
                sample_suffix = sample_splits[len('split_indices'):]
                self.display_split(sample_suffix)
        
        molecule_ids = self.factory.get_molecule_ids()
        if molecule_ids:
            print("Sample molecules:")
            for mol_idx in molecule_ids[:2]:
                self.display_molecule_by_idx(mol_idx, verbose=verbose)
        else:
            print("  No molecule groups found.")
    
    def display_valid_indices(self) -> None:
        """Display valid indices information."""
        try:
            lines = []
            lines.append("-" * 50)
            lines.append("Valid Indices Information:")
            
            # Try to get H spectra valid indices
            try:
                valid_h = self.factory.get_valid_indices('h')
                lines.append(f"  valid_indices_h: {valid_h.shape}, value[:50]={valid_h[:50] if len(valid_h) > 50 else valid_h}")
            except KeyError:
                lines.append("  valid_indices_h: None")
            
            # Try to get C spectra valid indices
            try:
                valid_c = self.factory.get_valid_indices('c')
                lines.append(f"  valid_indices_c: {valid_c.shape}, value[:50]={valid_c[:50] if len(valid_c) > 50 else valid_c}")
            except KeyError:
                lines.append("  valid_indices_c: None")
            
            # Try to get both
            try:
                valid_both = self.factory.get_valid_indices('both')
                lines.append(f"  valid_indices_both: {valid_both.shape}, value[:50]={valid_both[:50] if len(valid_both) > 50 else valid_both}")
            except KeyError:
                lines.append("  valid_indices_both: None")
            
            lines.append("-" * 50)
            print("\n".join(lines))
            
        except Exception as e:
            print(f"Error displaying valid indices: {e}")
