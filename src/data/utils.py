import numpy as np
from rdkit import Chem
import torch
from typing import Union, Dict, List, Optional

def smiles_to_svg(smiles, svg_path=None, image_size=(300, 300)):
    """
    Convert a SMILES string to SVG and optionally PNG representation of the molecule.
    
    Args:
        smiles (str): SMILES string of the molecule
        svg_path (str): Path to save the SVG file
        image_size (tuple): Size of the SVG image as (width, height)
    
    Returns:
        mol: RDKit molecule object with coordinates
    """
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Warning: Could not create molecule from SMILES: {smiles}")
        return None
    
    # Generate improved 2D coordinates
    try:
        rdCoordGen.AddCoords(mol)
    except Exception as coord_e:
        print(f"Warning: rdCoordGen failed: {coord_e}. Falling back to default coordinates.")
    
    # Generate SVG
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(image_size[0], image_size[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg_data = drawer.GetDrawingText()
    
    if svg_path:
        # Save the SVG to a file
        with open(svg_path, 'w') as f:
            f.write(svg_data)
        # print(f"SVG file saved to: {svg_path}")

def n_atoms_in_smiles(smiles: str, remove_h: bool = False) -> int:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 0
    if not remove_h:
        m = Chem.AddHs(m)
    return m.GetNumAtoms()

def canonicalize(
    smiles: str, 
    remove_stereo: bool = False,
) -> Optional[str]:
    """
    Canonicalize a SMILES string using RDKit.

    Args:
        smiles (str): The input SMILES string.
        remove_stereo (bool): Whether to remove stereochemistry information. Defaults to False.

    Returns:
        Optional[str]: The canonicalized SMILES string, or None if invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        isomeric_smiles = not remove_stereo
        smiles_out = Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles)
        if smiles_out is None:
            return None
        return Chem.CanonSmiles(smiles_out)
    except Exception as e:
        return None

class Molecule:
    """
    A class representing a molecule in 3D space.
    
    Handles conversion between tensor/numpy formats, masking, and export to RDKit or XYZ.
    """
    def __init__(self, 
                 atom_coords: Union[np.ndarray, torch.Tensor], 
                 atom_one_hot: Union[np.ndarray, torch.Tensor], 
                 atom_mask: Union[np.ndarray, torch.Tensor], 
                 atom_decoder: Union[Dict[int, str], List[str]], 
                 remove_h: bool, 
                 collapse: bool = False):
        """
        Initialize the Molecule.

        Args:
            atom_coords: Coordinates of atoms. Shape (n_atoms, 3).
            atom_one_hot: One-hot encoding of atom types. Shape (n_atoms, n_atom_types).
            atom_mask: Boolean mask indicating valid atoms. Shape (n_atoms,).
            atom_decoder: Mapping from integer index to atom symbol (e.g., ['C', 'H']).
            remove_h: Whether hydrogens are considered removed (affects validation).
            collapse: If True, immediately filters out masked atoms.
        """
        # Handle atom coordinates
        if isinstance(atom_coords, np.ndarray):
            self.atom_coords = atom_coords.astype(np.float64)
        elif isinstance(atom_coords, torch.Tensor):
            self.atom_coords = atom_coords.to(dtype=torch.float64)
        else:
            raise TypeError("atom_coords must be numpy.ndarray or torch.Tensor")
            
        self.atom_coords = self.atom_coords.reshape(-1, 3)  # Ensure shape is (n_atoms, 3)
        
        # Handle atom types
        self.atom_one_hot = atom_one_hot
        if isinstance(atom_one_hot, torch.Tensor):
            # Convert to numpy for index handling if needed, or use torch.argmax
            # Here we store atom_types as numpy array of ints for compatibility
            self.atom_types = torch.argmax(atom_one_hot, dim=1).detach().cpu().numpy().astype(int)
        else:
            self.atom_types = np.argmax(atom_one_hot, axis=1).astype(int)

        # Handle decoder
        self.atom_decoder = list(atom_decoder)

        # Handle mask
        if isinstance(atom_mask, np.ndarray):
            self.atom_mask = atom_mask.astype(bool)
        elif isinstance(atom_mask, torch.Tensor):
            self.atom_mask = atom_mask.to(dtype=torch.bool)
        else:
            raise TypeError("atom_mask must be either a numpy.ndarray or a torch.Tensor")
            
        self.remove_h = remove_h
        
        if not self.remove_h:
            assert 'H' in self.atom_decoder, "remove_h=False - H atom type must be present in the atom decoder."
        
        if collapse:
            self.collapse()
        
    def collapse(self):
        """
        Filters the molecule data to remove masked (invalid) atoms.
        Modifies the object in-place.
        """
        # Calculate number of valid atoms
        if isinstance(self.atom_mask, torch.Tensor):
            n_atoms = self.atom_mask.sum().item()
        else:
            n_atoms = np.sum(self.atom_mask)
            
        # Apply mask
        self.atom_coords = self.atom_coords[self.atom_mask] if self.atom_coords is not None else None
        self.atom_one_hot = self.atom_one_hot[self.atom_mask] if self.atom_one_hot is not None else None
        
        # atom_types is always numpy array based on __init__ logic above
        # We need to ensure mask is compatible if it's a tensor
        mask_np = self.atom_mask.detach().cpu().numpy() if isinstance(self.atom_mask, torch.Tensor) else self.atom_mask
        self.atom_types = self.atom_types[mask_np].astype(int) if self.atom_types is not None else None
        
        # Reset mask to all True since we've collapsed
        self.atom_mask = np.ones(int(n_atoms), dtype=bool)
        
    def to_rdkit_molecule(self) -> Chem.Mol:
        """
        Converts the Molecule object to an RDKit molecule.
        
        Note: This creates a molecule with atoms and 3D coordinates but NO bonds.
        
        Returns:
            Chem.Mol: The RDKit molecule object.
        """
        mol = Chem.RWMol()
        
        # Add atoms
        for atom_idx in self.atom_types:
            atom_symbol = self.atom_decoder[atom_idx]
            a = Chem.Atom(atom_symbol)
            mol.AddAtom(a)
        
        # Add 3D coordinates to atoms
        # Ensure coordinates are numpy float64 for RDKit
        coords = self.atom_coords
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()
            
        conf = Chem.Conformer(len(coords))
        for i, pos in enumerate(coords):
            # pos must be float64, not float32
            # float32 will raise an error: cannot extract desired type from sequence
            conf.SetAtomPosition(i, pos.astype(np.float64)) 
        
        mol.AddConformer(conf)
        return mol
    
    def save_as_xyz_file(self, file_path: str):
        """
        Save the molecule as an .xyz file.
        
        Warning: This method calls collapse(), which modifies the object in-place.

        Args:
            file_path (str): Path to save the .xyz file.
        """
        self.collapse()
        
        # Ensure coordinates are accessible
        coords = self.atom_coords
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()
            
        with open(file_path, 'w') as f:
            f.write(f"{len(self.atom_mask)}\n")
            f.write("\n") # Comment line
            for i, (atom_idx, pos) in enumerate(zip(self.atom_types, coords)):
                atom_symbol = self.atom_decoder[atom_idx]
                f.write("%s %.9f %.9f %.9f\n" % (atom_symbol, pos[0], pos[1], pos[2]))