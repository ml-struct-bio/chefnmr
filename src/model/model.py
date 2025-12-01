import os
import time
from typing import Dict, Any, List
import wandb
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import rank_zero_only

from src.data.utils import Molecule
from src.model.modules.diffusion import AtomDiffusion
from src.model.modules.utils import ExponentialMovingAverage
from src.evaluation import bond_analyzer, metrics

class NMRTo3DStructureElucidation(LightningModule):
    """
    LightningModule for NMR to 3D Structure Elucidation using Diffusion Models.
    
    This class handles:
    1. Model initialization and configuration.
    2. Training, validation, and testing loops.
    3. Diffusion loss computation.
    4. Sampling (inference) and molecule reconstruction.
    5. Metric evaluation (Validity, Matching Accuracy, Max Similarity, AMR) and visualization.
    """
    def __init__(
        self,
        cfg: DictConfig,
    ):
        """
        Args:
            cfg (DictConfig): Hydra configuration object containing all hyperparameters.
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_args = self.cfg.dataset_args
        self.training_args = self.cfg.training_args
        self.validation_args = self.cfg.validation_args
        self.test_args = self.cfg.test_args
        self.neural_network_args = self.cfg.neural_network_args
        self.diffusion_process_args = self.cfg.diffusion_process_args
        self.diffusion_loss_args = self.cfg.diffusion_loss_args
        self.score_model_args = self.cfg.score_model_args
        self.visualization_args = self.cfg.visualization_args
        self.trainer_args = self.cfg.trainer
        
        # Configure the backbone score model
        self._configure_score_model()
        
        self.model = AtomDiffusion(
            score_model_args=self.score_model_args,
            **self.diffusion_process_args
        )
        
        # Exponential Moving Average (EMA) setup
        self.ema = None
        self.use_ema = self.neural_network_args.use_ema
        self.ema_decay = self.neural_network_args.ema_decay
        
        # Initialize storage for epoch-level predictions when inference
        self.epoch_predictions = self._init_epoch_predictions()
        
        self.save_hyperparameters()
        
        # Auxiliary variables
        self.start_epoch_time = None # To track epoch period time
        self.loaded_ckpt_epoch = None
        
        rank_zero_only(print)("NMRTo3DStructureElucidation.dtype:", self.dtype)

    def _configure_score_model(self):
        """Helper to configure the score model based on dataset arguments."""
        score_model_name = self.score_model_args.model_name
        if score_model_name == "DiffusionModuleTransformer":
            # One-hot atom type encoding size
            self.score_model_args[score_model_name].in_atom_feature_size = len(self.dataset_args.atom_decoder)
            self.score_model_args[score_model_name].max_n_atoms = self.dataset_args.max_n_atoms
            
            rank_zero_only(print)(f"score_model_args[{score_model_name}].in_atom_feature_size:", {self.score_model_args[score_model_name].in_atom_feature_size})
            rank_zero_only(print)(f"score_model_args[{score_model_name}].max_n_atoms:", {self.score_model_args[score_model_name].max_n_atoms})
            
            # Set up conditioning type and input size
            self.score_model_args[score_model_name].condition = self.dataset_args.input_generator
            self.score_model_args[score_model_name].drop_transform = self.dataset_args.multitask_args.drop_transform
            
            if self.score_model_args[score_model_name].condition == "H1NMRSpectrum":
                self.score_model_args[score_model_name].in_condition_size = self.dataset_args.input_generator_addn_args['h1nmr']['input_dim']
            elif self.score_model_args[score_model_name].condition == "C13NMRSpectrum":
                self.score_model_args[score_model_name].in_condition_size = self.dataset_args.input_generator_addn_args['c13nmr']['input_dim']
            elif self.score_model_args[score_model_name].condition == "H1C13NMRSpectrum":
                self.score_model_args[score_model_name].in_condition_size = [
                    self.dataset_args.input_generator_addn_args['h1nmr']['input_dim'], 
                    self.dataset_args.input_generator_addn_args['c13nmr']['input_dim']
                ]
            else:
                raise NotImplementedError(f"Condition {self.dataset_args.input_generator} is not supported.")
                
            rank_zero_only(print)(f"score_model_args[{score_model_name}].condition:", {self.score_model_args[score_model_name].condition})
            rank_zero_only(print)(f"score_model_args[{score_model_name}].in_condition_size:", {self.score_model_args[score_model_name].in_condition_size})
        else:
            raise NotImplementedError(f"Score model {score_model_name} is not supported. Only DiffusionModuleTransformer is supported.")

    def _init_epoch_predictions(self) -> Dict[str, List]:
        """Initialize the dictionary to store predictions for the current epoch."""
        return {
            "target_smiles": [],
            "target_atom_coords": [],
            "predicted_smiles": [],
            "predicted_largest_fragment_smiles": [],
            "atom_coords": [],
            "atom_coords_chains": [],
            "atom_mask": [],
            "atom_one_hot": [],
            "similarity_list": [],
            "rmsd_list": [],
            "rmsd_wo_h_list": [],
        }
    
    def forward(
        self, 
        model_inputs: Dict[str, torch.Tensor],
        target_atom_coords: torch.Tensor,
        multiplicity_diffusion_train: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass wrapping the diffusion model.
        
        Args:
            model_inputs: Dictionary containing 'atom_one_hot', 'atom_mask', 'condition'.
            target_atom_coords: Ground truth coordinates for training loss.
            multiplicity_diffusion_train: Number of variations/augmentations per sample during training.
            
        Returns:
            Dict containing model outputs.
        """
        dict_out = self.model(
                model_inputs=model_inputs,
                atom_coords=target_atom_coords,
                multiplicity=multiplicity_diffusion_train,
            )
        
        return dict_out
    
    def on_train_epoch_start(self):
        if self.trainer.is_global_zero:
            self.start_epoch_time = time.time()

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: Tuple of ((inputs, smiles), targets).
            batch_idx: Index of the batch.
        """
        (model_inputs, smiles), model_outputs = batch
        batch_size = len(smiles)
        
        # Forward pass
        dict_out = self(
            model_inputs=model_inputs,
            target_atom_coords=model_outputs['atom_coords'],
            multiplicity_diffusion_train=self.training_args.diffusion_multiplicity)
        
        # Compute loss
        diffusion_loss_dict = self.model.compute_loss(
            model_inputs=model_inputs,
            dict_out=dict_out,
            multiplicity=self.training_args.diffusion_multiplicity,
            add_smooth_lddt_loss=self.diffusion_loss_args.add_smooth_lddt_loss,
            lddt_loss_threshold=self.diffusion_loss_args.lddt_loss_threshold,
        )
        
        loss = diffusion_loss_dict["loss"]
        self.log("train/diffusion_loss", loss, batch_size=batch_size)
        self.log("train/mse_loss", diffusion_loss_dict["loss_breakdown"]["mse_loss"], batch_size=batch_size)
        self.log("train/lddt_loss", diffusion_loss_dict["loss_breakdown"]["smooth_lddt_loss"], batch_size=batch_size)
        self.training_log()
        
        return loss

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:
            print(f"Training Epoch {self.current_epoch} - {time.time() - self.start_epoch_time:.1f}s")
        
    def validation_step(self, batch, batch_idx):
        """Validation step with optional sampling."""
        (model_inputs, smiles), model_outputs = batch
        
        batch_size = len(smiles)
        
        # Forward pass
        dict_out = self(
            model_inputs=model_inputs,
            target_atom_coords=model_outputs['atom_coords'],
            multiplicity_diffusion_train=self.validation_args.diffusion_multiplicity)
        
        # Compute loss
        diffusion_loss_dict = self.model.compute_loss(
            model_inputs=model_inputs,
            dict_out=dict_out,
            multiplicity=self.validation_args.diffusion_multiplicity,
            add_smooth_lddt_loss=self.diffusion_loss_args.add_smooth_lddt_loss,
            lddt_loss_threshold=self.diffusion_loss_args.lddt_loss_threshold,
        )
        
        loss = diffusion_loss_dict["loss"]
        self.log("val/diffusion_loss", loss * 100, sync_dist=True, batch_size=batch_size)
        self.log("val/mse_loss", diffusion_loss_dict["loss_breakdown"]["mse_loss"], sync_dist=True, batch_size=batch_size)
        self.log("val/lddt_loss", diffusion_loss_dict["loss_breakdown"]["smooth_lddt_loss"], sync_dist=True, batch_size=batch_size)
        
        # Sample for validation (periodically)
        should_sample = (self.current_epoch + 1) % (self.validation_args.sample_every_n_val_epoch * self.trainer_args.check_val_every_n_epoch) == 0
        if should_sample:
            if batch_idx == 0:
                rank_zero_only(print)("Sample for validation")
            self.sample_batch(batch, 
                              self.validation_args.diffusion_samples, 
                              self.validation_args.num_sampling_steps)
        return loss

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % (self.validation_args.sample_every_n_val_epoch * self.trainer_args.check_val_every_n_epoch) == 0:
            self.analyze_epoch_predictions(stage="val")

    def on_test_epoch_start(self):
        if self.trainer.is_global_zero:
            self.start_epoch_time = time.time()
        
    def test_step(self, batch, batch_idx):
        self.sample_batch(batch, 
                          self.test_args.diffusion_samples, 
                          self.test_args.num_sampling_steps)

    def on_test_epoch_end(self):
        self.analyze_epoch_predictions(stage="test")
        if self.trainer.is_global_zero:
            print(f"Test Epoch {self.loaded_ckpt_epoch} - {time.time() - self.start_epoch_time:.1f}s")

    def sample_batch(
            self,
            batch,
            diffusion_samples: int,
            num_sampling_steps: int,
    ):
        """
        Perform sampling (inference) on a batch and compute metrics.
        
        Args:
            batch: Input batch.
            diffusion_samples: Number of samples to generate per input (multiplicity).
            num_sampling_steps: Number of diffusion steps for reverse process.
        """
        (model_inputs, target_smiles), model_outputs = batch
        
        multiplicity = diffusion_samples
        predicted_atom_coords, predicted_atom_coords_chains = self.model.sample(
            model_inputs=model_inputs,
            num_sampling_steps=num_sampling_steps,
            multiplicity=multiplicity,
            n_chain_frames=self.visualization_args.n_chain_frames,
        )
        
        pred = {
            "atom_coords": predicted_atom_coords.cpu().detach().numpy(), # shape [batch_size * multiplicity, n_atoms, 3]
            "atom_coords_chains": predicted_atom_coords_chains.cpu().detach().numpy(), # shape [batch_size * multiplicity, n_chain_frames, n_atoms, 3]
            "atom_mask": model_inputs["atom_mask"].cpu().detach().numpy(),
            "atom_one_hot": model_inputs["atom_one_hot"].cpu().detach().numpy(),
            "target_smiles": target_smiles,
            "target_atom_coords": model_outputs['atom_coords'].cpu().detach().numpy(), # shape [batch_size, n_atoms, 3]
        }
        
        # Convert atom features to SMILES (reconstruction)
        # return List[str], len = batch_size * multiplicity
        # None if the SMILES is invalid
        predicted_smiles, predicted_largest_fragment_smiles = bond_analyzer.atom_features_to_smiles(
            self.dataset_args.name,
            pred, 
            multiplicity, 
            remove_h=self.dataset_args.remove_h, 
            remove_stereo=self.dataset_args.remove_stereo,
            atom_decoder=self.dataset_args.atom_decoder,
            timeout_seconds=self.dataset_args.timeout_seconds,
            num_workers=os.cpu_count() // self.trainer.world_size
        )
        
        # Tanimoto similarity
        similarity_list = metrics.smiles_to_similarity(
            target_smiles,
            predicted_smiles,
            multiplicity,
        )
        
        # RMSD
        # Support multiple ground truth conformations per target
        # Use all conformers for Average Minimum RMSD
        rmsd_list, rmsd_wo_h_list = metrics.compute_rmsd(
            predicted_atom_coords.cpu().detach().float(),
            model_outputs['all_atom_coords'].cpu().detach().float(),
            model_outputs['conformer_mask'].cpu().detach(),
            model_inputs["atom_mask"].cpu().detach().float(),
            model_inputs["atom_one_hot"].cpu().detach().float(),
            self.dataset_args.atom_decoder,
            multiplicity,
            num_workers=os.cpu_count() // self.trainer.world_size
        )
        
        pred.update({
            "predicted_smiles": predicted_smiles,
            "predicted_largest_fragment_smiles": predicted_largest_fragment_smiles,
            "similarity_list": similarity_list,
            "rmsd_list": rmsd_list,
            "rmsd_wo_h_list": rmsd_wo_h_list,
        })
        
        for key, value in pred.items():
            self.epoch_predictions[key].extend(value)

    def analyze_epoch_predictions(self, stage: str):
        """
        Aggregate predictions from the epoch, compute metrics, save samples, and visualize.
        
        Args:
            stage: 'val' or 'test'.
        """
        # stack the predictions
        processed_predictions = {}
        for key, value in self.epoch_predictions.items():
            if not value: # Skip if the list is empty
                processed_predictions[key] = []
                continue

            if isinstance(value[0], torch.Tensor):
                # Move tensors to CPU, detach, convert to numpy, then stack
                processed_predictions[key] = np.stack([item.cpu().detach().numpy() for item in value])
            elif isinstance(value[0], np.ndarray):
                # Already numpy arrays, just stack
                processed_predictions[key] = np.stack(value)
            else:
                # Keep lists of other types (like SMILES strings) as is
                processed_predictions[key] = value

        # Replace the original dictionary with the processed one
        self.epoch_predictions = processed_predictions
        
        # Compute metrics
        multiplicity = 1
        if len(self.epoch_predictions["target_smiles"]) > 0:
             multiplicity = len(self.epoch_predictions["predicted_smiles"]) // len(self.epoch_predictions["target_smiles"])
        
        # 1. Compute validity = number of valid SMILES / total number of SMILES (= sample_size * multiplicity)
        # number of valid SMILES = number of SMILES that are not None in predicted_smiles
        validity = metrics.compute_validity(self.epoch_predictions["predicted_smiles"])
        largest_fragment_validity = metrics.compute_validity(self.epoch_predictions["predicted_largest_fragment_smiles"])

        # Compute first-k metrics
        first_k_metrics = {}
        if len(self.epoch_predictions["target_smiles"]) > 0 and multiplicity > 0:
            if stage == "val":
                k_values = [multiplicity] # k = multiplicity
            elif stage == "test":
                k_values = np.arange(1, multiplicity + 1) # k = 1, 2, ..., multiplicity
            num_targets = len(self.epoch_predictions["target_smiles"])
            
            for k in k_values:
                matched_count_k = 0
                max_sim_k_list = []
                amr_k_list = []
                amr_wo_h_k_list = []
                for i, target_smiles in enumerate(self.epoch_predictions["target_smiles"]):
                    start_idx = i * multiplicity
                    end_idx = start_idx + k # Only consider the first k predictions
                    
                    # First-k matching
                    preds_k = self.epoch_predictions["predicted_smiles"][start_idx:end_idx]
                    for predicted_smile in preds_k:
                        if metrics.match_smiles(target_smiles, predicted_smile, remove_stereo=True): # ignore stereochemistry for matching
                            matched_count_k += 1
                            break

                    # First-k max similarity
                    sims_k = self.epoch_predictions["similarity_list"][start_idx:end_idx]
                    if sims_k: # Check if the list is not empty
                        max_sim_k_list.append(max(sims_k))
                    else:
                        max_sim_k_list.append(0.0) # Handle case with no valid similarities in top k
                        
                    # First-k min RMSD
                    rmsd_k = self.epoch_predictions["rmsd_list"][start_idx:end_idx]
                    amr_k_list.append(min(rmsd_k))
                    
                    rmsd_wo_h_k = self.epoch_predictions["rmsd_wo_h_list"][start_idx:end_idx]
                    amr_wo_h_k_list.append(min(rmsd_wo_h_k))

                first_k_metrics[f"match_acc_top{k}"] = matched_count_k / num_targets * 100.0 if num_targets > 0 else 0.0
                first_k_metrics[f"max_sim_top{k}"] = np.mean(max_sim_k_list) if max_sim_k_list else 0.0
                first_k_metrics[f"amr_top{k}"] = np.mean(amr_k_list) if amr_k_list else 0.0
                first_k_metrics[f"amr_wo_h_top{k}"] = np.mean(amr_wo_h_k_list) if amr_wo_h_k_list else 0.0
        
        if stage == "val":
            n_epoch = self.current_epoch
        elif stage == "test":
            n_epoch = self.loaded_ckpt_epoch if self.loaded_ckpt_epoch is not None else self.current_epoch
        # Save the samples
        if self.trainer.is_global_zero:
            if stage == "val":
                samples_dir = f"samples/{stage}_epoch{n_epoch:04d}"
            elif stage == "test":
                samples_dir = f"samples/{self.cfg.general.experiment_name}"
            os.makedirs(os.path.join(os.getcwd(), samples_dir), exist_ok=True)
            samples_file = {
                "target_smiles": os.path.join(os.getcwd(), samples_dir, "target_smiles.txt"),
                "predicted_smiles": os.path.join(os.getcwd(), samples_dir, "predicted_smiles.txt"),
                "predicted_largest_fragment_smiles": os.path.join(os.getcwd(), samples_dir, "predicted_largest_fragment_smiles.txt"),
                "similarity_list": os.path.join(os.getcwd(), samples_dir, "similarity_list.txt"),
                "target_predicted_similarity": os.path.join(os.getcwd(), samples_dir, "target_predicted_similarity.csv"),
            }
            for key, file_path in samples_file.items():
                if key == "target_predicted_similarity":
                    # Save target_smiles, predicted_smiles, similarity_list as a CSV file
                    rows = []
                    if len(self.epoch_predictions["target_smiles"]) > 0 and multiplicity > 0:
                        for i, target in enumerate(self.epoch_predictions["target_smiles"]):
                            # Get all predictions and similarities for this target
                            start_idx = i * multiplicity
                            end_idx = (i + 1) * multiplicity
                            preds = self.epoch_predictions["predicted_smiles"][start_idx:end_idx]
                            sims = self.epoch_predictions["similarity_list"][start_idx:end_idx]
                            # Add rows for this target
                            for pred, sim in zip(preds, sims):
                                rows.append({"target_smiles": target, "predicted_smiles": pred, "similarity": sim})
                    if rows: # Only save if there are rows
                        df = pd.DataFrame(rows)
                        df.to_csv(file_path, index=False)
                elif key in self.epoch_predictions and self.epoch_predictions[key]: # Check if key exists and list is not empty
                    with open(file_path, "w") as f:
                        for item in self.epoch_predictions[key]:
                            if item is not None:
                                f.write("%s\n" % item)
                            else:
                                f.write("None\n")
        
        self.log(f"{stage}/validity", validity, sync_dist=True)
        self.log(f"{stage}/largest_fragment_validity", largest_fragment_validity, sync_dist=True)
        self.log(f"{stage}/matching_accuracy", first_k_metrics.get(f"match_acc_top{multiplicity}", 0.0), sync_dist=True)
        self.log(f"{stage}/max_similarity", first_k_metrics.get(f"max_sim_top{multiplicity}", 0.0), sync_dist=True)
        self.log(f"{stage}/amr", first_k_metrics.get(f"amr_top{multiplicity}", 0.0), sync_dist=True)
        self.log(f"{stage}/amr_wo_h", first_k_metrics.get(f"amr_wo_h_top{multiplicity}", 0.0), sync_dist=True)
        
        if stage == "test":
            # Log first-k metrics
            for k_metric_name, k_metric_value in first_k_metrics.items():
                self.log(f"{stage}/{k_metric_name}", k_metric_value, sync_dist=True)
            
        # Visualize first k samples
        if self.trainer.is_global_zero:
            if stage == "val":
                visualize_samples = self.validation_args.visualize_samples
                visualize_chains = self.validation_args.visualize_chains # Use this to limit chain visualizations too
            elif stage == "test":
                visualize_samples = self.test_args.visualize_samples
                visualize_chains = self.test_args.visualize_chains # Use this to limit chain visualizations too
                
            # --- Visualize first prediction per target AND its chain ---
            print(f"Visualizing first prediction and its chain for up to {visualize_samples} unique targets.")
            visualized_targets = set()
            visualized_count = 0
            
            # Ensure multiplicity is valid and predictions exist
            if multiplicity > 0 and len(self.epoch_predictions["target_smiles"]) > 0 and len(self.epoch_predictions["atom_coords"]) > 0:
                
                first_pred_dir = os.path.join(os.getcwd(), samples_dir, "first_prediction_per_target")
                os.makedirs(first_pred_dir, exist_ok=True)
                os.makedirs(os.path.join(first_pred_dir, "xyz_files"), exist_ok=True)

                first_chain_dir = os.path.join(os.getcwd(), samples_dir, "first_prediction_chains")
                os.makedirs(first_chain_dir, exist_ok=True) # Directory for chains of first predictions

                for target_idx, target_smiles in enumerate(self.epoch_predictions["target_smiles"]):
                    if visualized_count >= visualize_samples:
                        break # Stop if we reached the desired number of visualizations

                    if target_smiles is not None and target_smiles not in visualized_targets:
                        pred_idx = target_idx * multiplicity # Index of the first prediction for this target

                        # Check if pred_idx is valid for both coords and chains
                        if pred_idx < len(self.epoch_predictions["atom_coords"]) and pred_idx < len(self.epoch_predictions["atom_coords_chains"]):
                            print(f"Visualizing first prediction and chain for target {target_idx} (SMILES: {target_smiles})")
                            
                            # --- Visualize the final state of the first prediction ---
                            atom_coords = self.epoch_predictions["atom_coords"][pred_idx]
                            atom_mask = self.epoch_predictions["atom_mask"][target_idx] # Mask/OneHot are per target
                            atom_one_hot = self.epoch_predictions["atom_one_hot"][target_idx]
                            
                            mol = Molecule(
                                atom_coords=atom_coords,
                                atom_one_hot=atom_one_hot,
                                atom_mask=atom_mask,
                                atom_decoder=self.dataset_args.atom_decoder,
                                remove_h=self.dataset_args.remove_h,
                                collapse=True
                            )
                            
                            safe_filename_base = f"target_{target_idx}"

                            # Save the final molecule as an .xyz file
                            xyz_file_path = os.path.join(first_pred_dir, f"xyz_files/{safe_filename_base}.xyz")
                            mol.save_as_xyz_file(xyz_file_path)

                            # Save the target SMILES for the final prediction
                            smiles_file_path_pred = os.path.join(first_pred_dir, f"{safe_filename_base}_smiles.txt")
                            with open(smiles_file_path_pred, 'w') as f:
                                f.write(target_smiles)

                            # --- Visualize the chain for the first prediction ---
                            if visualized_count < visualize_chains: # Check against chain visualization limit
                                atom_coords_chains = self.epoch_predictions["atom_coords_chains"][pred_idx]
                                n_chain_frames = atom_coords_chains.shape[0]
                                img_path_list = []

                                # Create directories for this specific chain
                                chain_mol_dir = os.path.join(first_chain_dir, f"target_{target_idx}")
                                chain_xyz_dir = os.path.join(chain_mol_dir, "xyz_files")
                                os.makedirs(chain_xyz_dir, exist_ok=True)

                                for j in range(n_chain_frames):
                                    mol_frame = Molecule(
                                        atom_coords=atom_coords_chains[j],
                                        atom_one_hot=atom_one_hot, # Use the same one_hot/mask
                                        atom_mask=atom_mask,
                                        atom_decoder=self.dataset_args.atom_decoder,
                                        remove_h=self.dataset_args.remove_h,
                                        collapse=True
                                    )

                                    # Save frame xyz
                                    frame_xyz_path = os.path.join(chain_xyz_dir, f"frame_{j}.xyz")
                                    mol_frame.save_as_xyz_file(frame_xyz_path)

                                # Save the target SMILES for the chain visualization
                                smiles_file_path_chain = os.path.join(chain_mol_dir, f"target_{target_idx}_smiles.txt")
                                with open(smiles_file_path_chain, 'w') as f:
                                    f.write(target_smiles)

                            visualized_targets.add(target_smiles)
                            visualized_count += 1
                        else:
                             print(f"Warning: Prediction index {pred_idx} out of bounds for target {target_idx} (coords len: {len(self.epoch_predictions['atom_coords'])}, chains len: {len(self.epoch_predictions['atom_coords_chains'])}).")

        # Clean up the predictions
        self.epoch_predictions = self._init_epoch_predictions()
    
    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        if self.training_args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                parameters,
                betas=(self.training_args.adam.adam_beta_1, self.training_args.adam.adam_beta_2),
                eps=self.training_args.adam.adam_eps,
                lr=self.training_args.base_lr,
            )
        else:
            raise NotImplementedError(f"Optimizer {self.training_args.optimizer} is not supported.")
        return [optimizer]

    def gradient_norm(self, module) -> float:
        # Only compute over parameters that are being trained
        parameters = filter(lambda p: p.requires_grad, module.parameters())
        parameters = filter(lambda p: p.grad is not None, parameters)
        norm = torch.tensor([p.grad.norm(p=2) ** 2 for p in parameters]).sum().sqrt()
        return norm
    
    def parameter_norm(self, module) -> float:
        # Only compute over parameters that are being trained
        parameters = filter(lambda p: p.requires_grad, module.parameters())
        norm = torch.tensor([p.norm(p=2) ** 2 for p in parameters]).sum().sqrt()
        return norm

    def training_log(self):
        self.log("train/gradient_norm", self.gradient_norm(self), prog_bar=False)
        self.log("train/parameter_norm", self.parameter_norm(self), prog_bar=False)
        
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False)

    # EMA functions
    # --> EMA functions begin
    def prepare_eval(self) -> None:
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )

        if self.use_ema:
            self.ema.store(self.parameters())
            self.ema.copy_to(self.parameters())
            
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.use_ema:
            checkpoint["ema"] = self.ema.state_dict()
            
    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.use_ema and "ema" in checkpoint:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )
            if self.ema.compatible(checkpoint["ema"]["shadow_params"]):
                self.ema.load_state_dict(checkpoint["ema"], device=torch.device("cpu"))
            else:
                self.ema = None
                print("Warning: EMA state not loaded due to incompatible model parameters.")
        self.loaded_ckpt_epoch = checkpoint["epoch"]
                
    def on_train_start(self):
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )
        elif self.use_ema:
            self.ema.to(self.device)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        # Updates EMA parameters after optimizer.step()
        if self.use_ema:
            self.ema.update(self.parameters())
            
    def on_validation_start(self):
        self.prepare_eval()
    
    def on_validation_end(self):
        if self.use_ema:
            self.ema.restore(self.parameters())
            
    def on_test_start(self):
        self.prepare_eval()

    def on_test_end(self):
        if self.use_ema:
            self.ema.restore(self.parameters())
    # <-- EMA functions end
            
    def setup(self, stage: str):
        """
        Setup hook for LightningModule.
        Handles distributed W&B logging setup for offline-sync mode.
        """
        if self.cfg.general.wandb == "offline-sync":
            # Ensure distributed mode
            if self.trainer.world_size > 1:
                if self.trainer.is_global_zero:
                    # Global rank 0 initializes the environment variable
                    wandb_dir = wandb.run.dir
                else:
                    wandb_dir = None  # Placeholder for other ranks
                
                # Broadcast the environment variable from rank 0 to all ranks
                wandb_dir_list = [wandb_dir]
                torch.distributed.broadcast_object_list(wandb_dir_list, src=0)
                os.environ["WANDB_DIR"] = wandb_dir_list[0]  # Set the variable

                print(f"Rank {self.global_rank}: Set WANDB_DIR to {os.environ['WANDB_DIR']}")

            else:
                # If single-GPU or CPU, just set the environment variable normally
                os.environ["WANDB_DIR"] = wandb.run.dir
                print(f"Single GPU or CPU: Set WANDB_DIR to {os.environ['WANDB_DIR']}")