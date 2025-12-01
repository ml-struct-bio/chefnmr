import numpy as np

def pad_atom_features(atom_features_dict, max_n_atoms, max_n_atom_types, max_n_conformers):
    """
    Pads atom feature arrays to fixed dimensions for batch processing.

    This function handles padding for:
    1. Atom coordinates (n_atoms, 3)
    2. Atom one-hot encodings (n_atoms, n_atom_types)
    3. Atom masks (n_atoms,)
    4. All conformer coordinates (n_conformers, n_atoms, 3) - Optional

    Args:
        atom_features_dict (dict): Dictionary containing raw atom features.
            Expected keys: 'atom_coords', 'atom_one_hot', 'atom_mask'.
            Optional keys: 'all_atom_coords', 'conformer_mask'.
        max_n_atoms (int): Target size for the atom dimension.
        max_n_atom_types (int): Target size for the atom type dimension.
        max_n_conformers (int): Target size for the conformer dimension.

    Returns:
        dict: A dictionary containing the padded arrays ready for model consumption.
    """
    f = atom_features_dict
    atom_coords  = f['atom_coords'][:]    # shape: (n_atoms, 3)
    atom_one_hot = f['atom_one_hot'][:]   # shape: (n_atoms, n_atom_types)
    atom_mask    = f['atom_mask'][:]      # shape: (n_atoms,)
    all_atom_coords = f.get('all_atom_coords', None)  # shape: (n_conformers, n_atoms, 3) or None
    conformer_mask = f.get('conformer_mask', None)    # shape: (n_conformers,) or None
    
    n_atoms, n_types = atom_one_hot.shape

    # Determine valid ranges to copy (truncate if input exceeds max dimensions)
    copy_n_atoms  = min(n_atoms, max_n_atoms)
    copy_n_types  = min(n_types, max_n_atom_types)

    # Initialize zero-filled arrays
    padded_atom_coords  = np.zeros((max_n_atoms, 3), dtype=atom_coords.dtype)
    padded_atom_one_hot = np.zeros((max_n_atoms, max_n_atom_types), dtype=atom_one_hot.dtype)
    padded_atom_mask    = np.zeros((max_n_atoms), dtype=atom_mask.dtype)

    # Copy data into padded arrays
    padded_atom_coords[:copy_n_atoms, :]  = atom_coords[:copy_n_atoms, :]
    padded_atom_one_hot[:copy_n_atoms, :copy_n_types] = atom_one_hot[:copy_n_atoms, :copy_n_types]
    padded_atom_mask[:copy_n_atoms]         = atom_mask[:copy_n_atoms]
    
    # Handle multi-conformer data if present
    padded_all_atom_coords = None
    padded_conformer_mask = None
    
    if all_atom_coords is not None and conformer_mask is not None:
        n_conformers = all_atom_coords.shape[0]
        copy_n_conformers = min(n_conformers, max_n_conformers)
        
        padded_all_atom_coords = np.zeros((max_n_conformers, max_n_atoms, 3), dtype=atom_coords.dtype)
        padded_all_atom_coords[:copy_n_conformers, :copy_n_atoms, :] = all_atom_coords[:copy_n_conformers, :copy_n_atoms, :]
        
        padded_conformer_mask = np.zeros((max_n_conformers,), dtype=conformer_mask.dtype)
        padded_conformer_mask[:copy_n_conformers] = conformer_mask[:copy_n_conformers]

    return {
        'atom_coords': padded_atom_coords,
        'atom_one_hot': padded_atom_one_hot,
        'atom_mask': padded_atom_mask,
        'atom_decoder': f.get('atom_decoder', None),
        'all_atom_coords': padded_all_atom_coords,
        'conformer_mask': padded_conformer_mask,
    }


def get_transform_function(data_dim, input_dim, transform_type):
    """
    Factory function to create data transformation callables.
    
    Args:
        data_dim (int): Dimension of the source data.
        input_dim (int): Expected dimension for the model input.
        transform_type (str): Strategy to use ('identity' or 'zero').
            - 'identity': Passes data through unchanged (requires dims to match).
            - 'zero': Returns a zero array of the same shape (used for masking/dropout).
        
    Returns:
        callable: A function taking an array and returning the transformed array.
    """
    if data_dim == input_dim and transform_type == "identity":
        return lambda x: x
    elif transform_type == "zero":
        return lambda x: np.zeros_like(x)
    else:
        raise NotImplementedError(f"Transform type '{transform_type}' with dims {data_dim}->{input_dim} not supported.")

def select_random_item(items, max_items=None):
    """
    Randomly selects a single item from a collection (list or numpy array).
    
    Used for data augmentation, e.g., selecting a random conformer or random 
    spectrum measurement from available data.
    
    Args:
        items: Collection to sample from.
        max_items (int, optional): If set, limits the selection range to [0, max_items).
    
    Returns:
        The selected item.
    """
    if max_items is None:
        max_n = len(items)
    else:
        # Handle numpy arrays vs lists
        collection_size = items.shape[0] if hasattr(items, 'shape') else len(items)
        max_n = min(max_items, collection_size)
    
    if max_n > 1:
        random_index = np.random.randint(0, max_n)
        return items[random_index]
    return items[0]

class NMRInputGenerator:
    """
    Prepares data samples for the NMR-to-Structure model.
    
    This class encapsulates the logic for:
    1. Processing molecule dictionaries into model-ready tensors.
    2. Handling different conditioning modes (H1, C13, or both).
    3. Applying data augmentation via condition dropout (masking inputs during training).
    """
    def __init__(self, condition_type, input_generator_addn_args: dict = None, use_all_conformers: bool = False):
        """
        Args:
            condition_type (str): The input modality. Options:
                - 'H1NMRSpectrum': 1H NMR only.
                - 'C13NMRSpectrum': 13C NMR only.
                - 'H1C13NMRSpectrum': Both 1H and 13C NMR.
            input_generator_addn_args (dict): Configuration dictionary containing:
                - 'atom_features': Max dimensions for atoms/types/conformers.
                - 'h1nmr'/'c13nmr': Dimensions and transform settings for spectra.
                - 'multitask_args': Dropout probabilities for robustness.
            use_all_conformers (bool): If True, returns all conformers (eval mode). 
                                       If False, samples one random conformer (train mode).
        """
        if input_generator_addn_args:
            self.max_n_atoms = input_generator_addn_args['atom_features']['max_n_atoms']
            self.max_n_atom_types = input_generator_addn_args['atom_features']['max_n_atom_types']
            self.max_n_conformers = input_generator_addn_args['atom_features']['max_n_conformers']
            self.known_atoms = input_generator_addn_args['atom_features']['known_atoms']
        
        self.condition_type = condition_type
        self.use_all_conformers = use_all_conformers
        
        # Initialize condition-specific handlers
        if condition_type == 'H1NMRSpectrum':
            self._setup_h1nmr(input_generator_addn_args)
        elif condition_type == 'C13NMRSpectrum':
            self._setup_c13nmr(input_generator_addn_args)
        elif condition_type == 'H1C13NMRSpectrum':
            self._setup_h1nmr(input_generator_addn_args)
            self._setup_c13nmr(input_generator_addn_args)
        else:
            raise ValueError(f"Unknown condition type: {condition_type}")

        # Configure dropout probabilities for multi-task/robustness training
        # These control how often a specific modality is zeroed out during training.
        multitask_args = input_generator_addn_args.get('multitask_args', {})
        self.p_drop_h1nmr = multitask_args.get('p_drop_h1nmr', 0.0)
        self.p_drop_c13nmr = multitask_args.get('p_drop_c13nmr', 0.0)
        self.p_drop_both = multitask_args.get('p_drop_both', 0.0)
        
        self.p_drop_sum = self.p_drop_h1nmr + self.p_drop_c13nmr + self.p_drop_both
        assert self.p_drop_sum <= 1.0, "Total dropout probability must not exceed 1.0"
        
        # Transform used when dropping a condition (i.e., zeroes it out)
        self.drop_transform = get_transform_function(None, None, multitask_args.get('drop_transform', 'zero'))

    def _setup_h1nmr(self, args):
        """Configures H1 NMR specific parameters and dummy placeholders."""
        self.h1_data_dim = args['h1nmr']['data_dim']
        self.h1_input_dim = args['h1nmr']['input_dim']
        
        if self.h1_data_dim == 10000:
            self.h1_spectra_key = 'h_10k'
            self.dummy_h1_spectra = np.zeros((10000), dtype=np.float32)
        else:
            raise NotImplementedError(f"Unsupported H1 data dimension: {self.h1_data_dim}. Only 10k supported.")
            
        self.max_n_h1spectra = args['h1nmr']['max_n_spectra']
        self.transform_h1spectra = get_transform_function(
            self.h1_data_dim, 
            self.h1_input_dim, 
            args['h1nmr']['transform']
        )

    def _setup_c13nmr(self, args):
        """Configures C13 NMR specific parameters and dummy placeholders."""
        self.c13_data_dim = args['c13nmr']['data_dim']
        self.c13_input_dim = args['c13nmr']['input_dim']
        
        if self.c13_data_dim == 10000:
            self.c13_spectra_key = 'c_10k'
            self.dummy_c13_spectra = np.zeros((10000), dtype=np.float32)
        elif self.c13_data_dim == 80:
            self.c13_spectra_key = 'c_80'
            self.dummy_c13_spectra = np.zeros((80), dtype=np.float32)
        else:
            raise NotImplementedError(f"Unsupported C13 data dimension: {self.c13_data_dim}. Supported: 10k, 80.")
            
        self.max_n_c13spectra = args['c13nmr']['max_n_spectra']
        self.transform_c13spectra = get_transform_function(
            self.c13_data_dim, 
            self.c13_input_dim, 
            args['c13nmr']['transform']
        )

    def _get_h1nmr_condition(self, mol_dict):
        """Extracts H1 spectrum, handling missing data with dummy zero-vectors."""
        if self.h1_spectra_key in mol_dict['spectra']:
            h1_spectra = mol_dict['spectra'][self.h1_spectra_key]
            # Validate spectrum data integrity
            if (isinstance(h1_spectra, np.ndarray) and h1_spectra.ndim == 2 
                    and h1_spectra.shape[0] > 0 and not np.isnan(h1_spectra).any()):
                # Select one spectrum if multiple exist (augmentation)
                h1_condition = select_random_item(h1_spectra, self.max_n_h1spectra)
                return self.transform_h1spectra(h1_condition)
        
        # Fallback for missing/invalid data
        return self.drop_transform(self.dummy_h1_spectra)
    
    def _get_c13nmr_condition(self, mol_dict):
        """Extracts C13 spectrum, handling missing data with dummy zero-vectors."""
        if self.c13_spectra_key in mol_dict['spectra']:
            c13_spectra = mol_dict['spectra'][self.c13_spectra_key]
            # Validate spectrum data integrity
            if (isinstance(c13_spectra, np.ndarray) and c13_spectra.ndim == 2 
                    and c13_spectra.shape[0] > 0 and not np.isnan(c13_spectra).any()):
                # Select one spectrum if multiple exist (augmentation)
                c13_condition = select_random_item(c13_spectra, self.max_n_c13spectra)
                return self.transform_c13spectra(c13_condition)
        
        # Fallback for missing/invalid data
        return self.drop_transform(self.dummy_c13_spectra)

    def _prepare_common_inputs(self, mol_dict, condition=None):
        """
        Prepares the standard inputs required by the model regardless of condition.
        
        Args:
            mol_dict: Raw molecule data.
            condition: The processed condition vector (e.g., NMR spectrum).
            
        Returns:
            tuple: (model_inputs, smiles, model_outputs)
        """
        model_inputs = {}
        model_outputs = {}
        
        atom_features = mol_dict['atom_features']
        all_atom_coords = atom_features['atom_coords']
        
        # Data Augmentation: Randomly select one conformer as the "target" for this step
        # Note: We keep all conformers in 'all_atom_coords' for evaluation AMR metrics during validation/testing.
        atom_features['atom_coords'] = select_random_item(all_atom_coords, self.max_n_conformers)
        atom_features['all_atom_coords'] = all_atom_coords
        atom_features['conformer_mask'] = np.ones(len(all_atom_coords), dtype=atom_features['atom_mask'].dtype)
        
        # Pad features to fixed size for batching
        padded_atom_features = pad_atom_features(
            atom_features, 
            self.max_n_atoms, 
            self.max_n_atom_types, 
            self.max_n_conformers
        )
        
        # Inputs: What the model sees
        model_inputs['atom_mask'] = padded_atom_features['atom_mask']
        if self.known_atoms:
            model_inputs['atom_one_hot'] = padded_atom_features['atom_one_hot']
        
        if condition is not None:
            model_inputs['condition'] = condition
            
        smiles = mol_dict['smiles']
        
        # Outputs: Ground truth for loss calculation
        model_outputs['atom_coords'] = padded_atom_features['atom_coords']
        
        if self.use_all_conformers:
            model_outputs['all_atom_coords'] = padded_atom_features['all_atom_coords']
            model_outputs['conformer_mask'] = padded_atom_features['conformer_mask']
        
        return model_inputs, smiles, model_outputs

    def forward(self, mol_dict: dict) -> tuple:
        """
        Main processing pipeline for a single molecule.
        
        1. Extracts NMR conditions.
        2. Applies dropout logic (randomly zeroing out conditions).
        3. Prepares atom features and coordinates.
        
        Args:
            mol_dict: Dictionary containing molecule data.
            
        Returns:
            tuple: ((model_inputs, smiles), model_outputs)
        """
        condition = None
        
        # --- Condition Processing & Dropout Logic ---
        if self.condition_type == 'H1NMRSpectrum':
            condition = self._get_h1nmr_condition(mol_dict)
            # Randomly drop H1 condition based on probability
            if self.p_drop_h1nmr > 0.0 and np.random.rand() < self.p_drop_h1nmr:
                condition = self.drop_transform(condition)
                
        elif self.condition_type == 'C13NMRSpectrum':
            condition = self._get_c13nmr_condition(mol_dict)
            # Randomly drop C13 condition based on probability
            if self.p_drop_c13nmr > 0.0 and np.random.rand() < self.p_drop_c13nmr:
                condition = self.drop_transform(condition)
                
        elif self.condition_type == 'H1C13NMRSpectrum':
            transformed_h1 = self._get_h1nmr_condition(mol_dict)
            transformed_c13 = self._get_c13nmr_condition(mol_dict)
            
            # Complex dropout logic for dual-condition:
            # Can drop H1, C13, Both, or None based on defined probabilities.
            if self.p_drop_sum > 0.0:
                drop_actions = ['h1', 'c13', 'both', 'none']
                drop_probs = [
                    self.p_drop_h1nmr, 
                    self.p_drop_c13nmr, 
                    self.p_drop_both, 
                    1.0 - (self.p_drop_sum)
                ]
                drop_action = np.random.choice(drop_actions, p=drop_probs)
                
                if drop_action in ['h1', 'both']:
                    transformed_h1 = self.drop_transform(transformed_h1)
                if drop_action in ['c13', 'both']:
                    transformed_c13 = self.drop_transform(transformed_c13)
            
            # Concatenate conditions (H1 + C13)
            condition = np.concatenate((transformed_h1, transformed_c13), axis=0)
        
        # --- Input Preparation ---
        model_inputs, smiles, model_outputs = self._prepare_common_inputs(mol_dict, condition)
        
        return (model_inputs, smiles), model_outputs