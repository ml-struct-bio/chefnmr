import pathlib
import numpy as np
from typing import Any, List
from omegaconf import OmegaConf

from torch.utils.data import DataLoader, Dataset
import lightning as pl
from lightning.pytorch.utilities import rank_zero_only
from src.data.factory.factory import MoleculeFactory
from src.data.input_generator import NMRInputGenerator

class NMRDataset(Dataset):
    """
    Dataset class for processing molecular data with NMR spectra.
    
    This class handles loading and preprocessing of NMR spectra data along with their 
    corresponding molecular structures (atom coordinates, one-hot atom type encodings, etc.).
    It supports different types of NMR spectra (1H, 13C) at various resolutions and 
    can combine them based on the requested input generator configuration.
    """
    def __init__(self, 
                datadir: str | pathlib.Path,
                data_indices: np.ndarray | List[int],
                input_generator: str,
                input_generator_addn_args: dict[str, Any],
                use_all_conformers: bool = False):
        """
        Initialize the NMRDataset.
        
        Args:
            datadir: Path to the directory containing the dataset.
            data_indices: Array or list of indices defining the subset of data to use.
            input_generator: Identifier string for the input generation strategy.
            input_generator_addn_args: Dictionary of additional arguments for the input generator.
            use_all_conformers: If True, returns all conformers (useful for AMR calculation during validation/testing).
        """
        self.data_manager = MoleculeFactory(datadir=datadir, mode='r')
        self.data_indices = data_indices
        self.use_all_conformers = use_all_conformers
        self.input_generator = NMRInputGenerator(input_generator, input_generator_addn_args=input_generator_addn_args, use_all_conformers=use_all_conformers)
        
    def __len__(self):
        """Return the number of molecules in the dataset."""
        return len(self.data_indices)

    def __getitem__(self, idx):
        """
        Retrieve a single data item.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            tuple: ((model_inputs, smiles), model_outputs)
                - model_inputs: Dictionary containing atom features, spectra, and conditioning info.
                - smiles: SMILES string of the molecule.
                - model_outputs: Dictionary containing target coordinates (single conformer and/or all conformers).
        """
        
        mol_dict = self.data_manager.get_molecule_data_by_idx(
            mol_idx=self.data_indices[idx],
            include_skeleton_smiles=False,
            include_spectra=True,
            include_atom_features=True,
        )
        
        return self.input_generator.forward(mol_dict)
        
class NMRDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the NMR to 3D structure elucidation task.
    
    This module encapsulates all data loading logic, including:
    - Data preparation (splitting, statistics computation).
    - Dataset instantiation for training, validation, and testing.
    - DataLoader creation.
    """
    def __init__(self, datacfg):
        """
        Initialize the NMRDataModule.
        
        Args:
            datacfg: Configuration object (e.g., Hydra config) containing dataset parameters.
        """
        super().__init__()
        self.datacfg = datacfg
        
        # Verify current supported configurations
        assert not self.datacfg.remove_h, "Currently only supports remove_h = False"
        
        # Set up data directory
        self.datadir = pathlib.Path(self.datacfg.datadir)
        
        # Initialize MoleculeFactory for data access
        self.mol_factory = MoleculeFactory(datadir=self.datadir, mode='r')
        
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.sigma_data = self.datacfg.sigma_data
        self.max_n_atoms = self.datacfg.max_n_atoms
        self.input_generator_addn_args = OmegaConf.to_container(self.datacfg.input_generator_addn_args, resolve=True)
    
    def prepare_data(self):
        """
        Prepare data for use.
        
        This method is called only once by the rank 0 process in distributed settings.
        It handles:
        - Logging data configuration.
        - Loading dataset splits.
        - Computing or loading data statistics (sigma_data, max_n_atoms).
        - Configuring input generator arguments based on dataset statistics.
        """
        rank_zero_only(print)(f"## Preparing data for {self.datacfg.name}")
        rank_zero_only(print)(f"## Data configs:")
        rank_zero_only(print)(f"### remove_stereo: {self.datacfg.remove_stereo}")
        rank_zero_only(print)(f"### condition: {self.datacfg.input_generator}")
        rank_zero_only(print)(f"### known chemical formula: {self.datacfg.known_atoms}")
        rank_zero_only(print)(f"### max_n_h1spectra: {self.datacfg.augmentation_args.max_n_h1spectra}")
        rank_zero_only(print)(f"### max_n_c13spectra: {self.datacfg.augmentation_args.max_n_c13spectra}")
        rank_zero_only(print)(f"### max_n_conformers: {self.datacfg.augmentation_args.max_n_conformers}")
        
        # Load split indices, sigma_data, and max_n_atoms from MoleculeFactory
        recompute_sigma_data = self.sigma_data is None
        recompute_max_n_atoms = self.max_n_atoms is None
        split_dict = self.mol_factory.get_split(suffix=self.datacfg.split_indices_suffix,
                                                recompute_sigma_data=recompute_sigma_data, 
                                                 recompute_max_n_atoms=recompute_max_n_atoms)
        assert split_dict is not None, "Failed to get split indices from MoleculeFactory"
        self.train_indices = split_dict['train']
        self.val_indices = split_dict['val']
        self.test_indices = split_dict['test']
        if recompute_sigma_data:
            self.sigma_data = split_dict['sigma_data']
            rank_zero_only(print)(f"### Computed sigma_data: {self.sigma_data}")
        if recompute_max_n_atoms:
            self.max_n_atoms = split_dict['max_n_atoms']
            rank_zero_only(print)(f"### Computed max_n_atoms: {self.max_n_atoms}")
        
        if 'h1nmr' not in self.input_generator_addn_args:
            self.input_generator_addn_args['h1nmr'] = {}
        self.input_generator_addn_args['h1nmr']['max_n_spectra'] = self.datacfg.augmentation_args.max_n_h1spectra

        if 'c13nmr' not in self.input_generator_addn_args:
            self.input_generator_addn_args['c13nmr'] = {}
        self.input_generator_addn_args['c13nmr']['max_n_spectra'] = self.datacfg.augmentation_args.max_n_c13spectra

        if 'atom_features' not in self.input_generator_addn_args:
            self.input_generator_addn_args['atom_features'] = {}
        self.input_generator_addn_args['atom_features']['max_n_conformers'] = self.datacfg.augmentation_args.max_n_conformers
        self.input_generator_addn_args['atom_features']['max_n_atoms'] = self.max_n_atoms
        self.input_generator_addn_args['atom_features']['max_n_atom_types'] = len(self.datacfg.atom_decoder)
        self.input_generator_addn_args['atom_features']['known_atoms'] = self.datacfg.known_atoms
    
        if 'multitask_args' not in self.input_generator_addn_args:
            self.input_generator_addn_args['multitask_args'] = {}
        self.input_generator_addn_args['multitask_args']['p_drop_h1nmr'] = self.datacfg.multitask_args.p_drop_h1nmr
        self.input_generator_addn_args['multitask_args']['p_drop_c13nmr'] = self.datacfg.multitask_args.p_drop_c13nmr
        self.input_generator_addn_args['multitask_args']['p_drop_both'] = self.datacfg.multitask_args.p_drop_both
        self.input_generator_addn_args['multitask_args']['drop_transform'] = self.datacfg.multitask_args.drop_transform

    def setup(self, stage):
        """
        Set up datasets for training, validation, and testing.
        
        This method runs on every GPU in distributed training. It instantiates
        the NMRDataset objects for the requested stage using the indices and 
        statistics prepared in prepare_data().
        
        Args:
            stage: The stage of processing ('fit', 'test')
        """
        if stage == 'fit':
            if self.datacfg.train_args.use_subset:
                train_rng = np.random.default_rng(self.datacfg.train_args.train_seed)  # Fixed seed for reproducibility
                num_samples = min(self.datacfg.train_args.train_samples, len(self.train_indices))
                self.train_indices = train_rng.choice(self.train_indices, num_samples, replace=False)
            self._train_set = NMRDataset(
                datadir=self.datadir,
                data_indices=self.train_indices,
                input_generator=self.datacfg.input_generator,
                input_generator_addn_args=self.input_generator_addn_args,
                use_all_conformers=False,  # Training uses single random conformer
            )
            
            val_input_generator_addn_args = self.input_generator_addn_args.copy()
            val_input_generator_addn_args['multitask_args']['p_drop_h1nmr'] = 0.0 # No dropout for validation
            val_input_generator_addn_args['multitask_args']['p_drop_c13nmr'] = 0.0
            val_input_generator_addn_args['multitask_args']['p_drop_both'] = 0.0
            self._val_set = NMRDataset(
                datadir=self.datadir,
                data_indices=self.val_indices,
                input_generator=self.datacfg.input_generator,
                input_generator_addn_args=val_input_generator_addn_args,
                use_all_conformers=True,  # Validation uses all conformers for Average Minimum RMSD
            )
            
            rank_zero_only(print)("### len(train_set):", len(self._train_set))
            rank_zero_only(print)("### len(val_set):", len(self._val_set))
        
        elif stage == "test":
            if self.datacfg.test_args.test_index:
                test_index = int(self.datacfg.test_args.test_index.lstrip('0') or '0')
                # test_index is an integer. Check if it is in self.test_indices (an np.ndarray)
                if isinstance(test_index, int) and test_index in self.test_indices:
                    test_indices = np.array([test_index])
                else:
                    raise ValueError(f"test_index {test_index} is not in the test indices.")
            else:
                rng = np.random.default_rng(self.datacfg.test_args.test_seed)  # Fixed seed for reproducibility
                num_samples = min(self.datacfg.test_args.test_samples, len(self.test_indices))
                test_indices = rng.choice(self.test_indices, num_samples, replace=False)
            test_input_generator_addn_args = self.input_generator_addn_args.copy()
            test_input_generator_addn_args['multitask_args']['p_drop_h1nmr'] = self.datacfg.test_args.test_p_drop_h1nmr
            test_input_generator_addn_args['multitask_args']['p_drop_c13nmr'] = self.datacfg.test_args.test_p_drop_c13nmr
            test_input_generator_addn_args['multitask_args']['p_drop_both'] = 0.0 # No dropout for test
            self._test_subset = NMRDataset(
                datadir=self.datadir,
                data_indices=test_indices,
                input_generator=self.datacfg.input_generator,
                input_generator_addn_args=test_input_generator_addn_args,
                use_all_conformers=True,  # Test uses all conformers for Average Minimum RMSD
            )
            rank_zero_only(print)("### len(test_subset):", len(self._test_subset))

    def train_dataloader(self):
        return DataLoader(
            self._train_set,
            batch_size=self.datacfg.batch_size,
            num_workers=self.datacfg.num_workers,
            pin_memory=self.datacfg.pin_memory,
            shuffle=self.datacfg.shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_set,
            batch_size=self.datacfg.batch_size,
            num_workers=self.datacfg.num_workers,
            pin_memory=self.datacfg.pin_memory,
            shuffle=False
        )
        
    def test_dataloader(self):
        """Return the DataLoader for test data (using the sampled subset)."""
        return DataLoader(
            self._test_subset,
            batch_size=self.datacfg.test_args.test_batch_size,
            num_workers=self.datacfg.num_workers,
            pin_memory=self.datacfg.pin_memory,
            shuffle=False
        )