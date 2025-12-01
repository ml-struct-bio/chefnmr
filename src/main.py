import hydra
from omegaconf import DictConfig

# PyTorch Lightning imports for training management
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from src.log.wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback
from lightning.pytorch.utilities import rank_zero_only

import sys
import os
from pathlib import Path
# Add the current directory to system path to ensure local imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Local module imports
from src.data.datamodule import NMRDataModule
from src.model.model import NMRTo3DStructureElucidation
from src.utils import (
    rank0_print,  # Utility for printing only from rank 0 process in distributed training
    update_ckpt_config  # Utility to update config with checkpoint values
)

# Get the project root directory for wandb offline sync in SLURM jobs
project_dir = Path(__file__).resolve().parents[1]

@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    """
    Main function for training and evaluating NMR to 3D structure models.
    
    The pipeline:
    1. Load and possibly update configuration
    2. Set up data module
    3. Initialize model
    4. Configure callbacks, loggers, and training strategy
    5. Create trainer and run training/testing
    
    Args:
        cfg (DictConfig): Hydra configuration object containing all parameters
    """
    
    # If checkpoint path is provided, update config with values from checkpoint
    if cfg.general.ckpt_abs_path is not None:
        rank0_print(f"# Updated config with new keys from checkpoint: {cfg.general.ckpt_abs_path}")
        new_cfg = cfg.copy() # Create a copy to avoid issues during config update
        cfg = update_ckpt_config(new_cfg=new_cfg)
        # Change working directory to the checkpoint's project directory

    rank0_print(f"# Config:\n{cfg}")

    # Set up working directory and create samples folder. 
    cwd = os.getcwd()
    rank0_print("# Current Working Directory:", cwd)
    rank_zero_only(os.makedirs)(os.path.join(cwd, "samples"), exist_ok=True)
    
    # Extract configuration sections for easier access
    dataset_args = cfg.dataset_args
    general_cfg = cfg.general
    trainer_cfg = cfg.trainer
    
    # Set random seed for reproducibility
    if general_cfg.seed is not None:
        rank0_print(f"# Setting global random seed: {general_cfg.seed}")
        seed_everything(general_cfg.seed, workers=True)
    
    # Initialize and prepare the data module
    rank0_print("# Creating datamodule...")
    datamodule = NMRDataModule(dataset_args)
    datamodule.prepare_data()
    
    # Set sigma_data from datamodule if not provided in config
    if cfg.diffusion_process_args.edm_args.sigma_data is None:
        cfg.diffusion_process_args.edm_args.sigma_data = datamodule.sigma_data
    
    # Update max number of atoms in config from datamodule
    cfg.dataset_args.max_n_atoms = datamodule.max_n_atoms
    rank0_print(f"cfg.diffusion_process_args.edm_args.sigma_data: {cfg.diffusion_process_args.edm_args.sigma_data}")
    rank0_print(f"cfg.dataset_args.max_n_atoms: {cfg.dataset_args.max_n_atoms}")

    # Initialize the model with updated configuration
    model = NMRTo3DStructureElucidation(cfg=cfg)
    
    # Set up callbacks for model checkpointing and logging
    callbacks = []
    if general_cfg.save_checkpoint:
        # Callback to save checkpoints based on matching accuracy
        matching_accuracy_ckpt_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{general_cfg.experiment_name}/accuracy",
            filename="epoch{epoch:04d}-accuracy{val/matching_accuracy:.2f}",
            auto_insert_metric_name=False,
            monitor="val/matching_accuracy",
            save_top_k=general_cfg.save_top_n_ckpts,
            mode="max",
            every_n_epochs=trainer_cfg.check_val_every_n_epoch * cfg.validation_args.sample_every_n_val_epoch)
        callbacks.append(matching_accuracy_ckpt_callback)
        
        # Callback to save checkpoints based on diffusion loss
        diffusion_loss_ckpt_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{general_cfg.experiment_name}/diffusion_loss",
            filename="epoch{epoch:04d}-loss{val/diffusion_loss:.2f}",
            auto_insert_metric_name=False,
            monitor="val/diffusion_loss",
            save_top_k=general_cfg.save_top_n_ckpts,
            mode="min",
            every_n_epochs=trainer_cfg.check_val_every_n_epoch,
            save_last=True)  # Also save the last model checkpoint
        callbacks.append(diffusion_loss_ckpt_callback)
    
    # Add callback for offline wandb synchronization if enabled for SLURM jobs
    if general_cfg.wandb == "offline-sync":
        callbacks.append(TriggerWandbSyncLightningCallback(
            communication_dir=project_dir / ".wandb_osh_command_dir"
        ))  # Wandb offline sync

    # Configure logging with Weights & Biases (Wandb)
    loggers = []
    if general_cfg.wandb != "disabled":
        wandb_logger = WandbLogger(
            name=os.getcwd().split("/")[-1],  # Use directory name as run name
            project=dataset_args.name,        # Use dataset name as project name
            log_model=False,                  # Don't log model checkpoints to W&B
            offline=general_cfg.wandb == "offline" or general_cfg.wandb == "offline-sync",
        )
        loggers.append(wandb_logger)

    # Configure distributed training strategy if using multiple devices
    strategy = "auto"
    devices = trainer_cfg.get("devices", 1)
    if isinstance(devices, int) and devices > 1:
        strategy = DDPStrategy(find_unused_parameters=dataset_args.find_unused_parameters)

    # Determine if running in an interactive SLURM job to enable progress bar when training and testing
    is_salloc = os.environ.get("SLURM_JOB_NAME", "") == "interactive"  # safer check
    if is_salloc:
        rank0_print(f"# Running inside SALLOC, enable progress bar")
    
    # Initialize the PyTorch Lightning trainer
    trainer = Trainer(
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=is_salloc,  # Only show progress bar in interactive sessions
        **trainer_cfg  # Pass all other trainer configuration parameters
    )
    
    # Execute training or testing based on specified mode
    if general_cfg.mode == "train":
        trainer.fit(model, datamodule=datamodule, ckpt_path=general_cfg.ckpt_abs_path)
        trainer.test(model, datamodule=datamodule)  # Run test after training completes
    elif general_cfg.mode == "test":
        trainer.test(model, datamodule=datamodule, ckpt_path=general_cfg.ckpt_abs_path)
        
if __name__ == '__main__':
    main()