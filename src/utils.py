from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf, open_dict
from src.model.model import NMRTo3DStructureElucidation

def format_time(seconds):
    """Format time in seconds to a readable string with appropriate units."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def rank0_print(*args, **kwargs):
    """
    Utility function to print messages only from the rank 0 process during distributed training.
    """
    rank_zero_only(print)(*args, **kwargs)

def add_config_with_new_keys(old_cfg, new_cfg):
    """
    Add new configuration keys from new_cfg to old_cfg without overwriting existing values.
    
    This function ensures backward compatibility when loading models with updated configurations.
    It only adds keys that don't exist in the old configuration.
    
    Args:
        old_cfg (DictConfig): Original configuration from a checkpoint
        new_cfg (DictConfig): New configuration with potentially new parameters
        
    Returns:
        DictConfig: Updated configuration with new keys added
    """
    categories = [
        'dataset_args', 
        'general',
        'neural_network_args',
        'score_model_args',
        'trainer',
        'training_args',
        'validation_args',
        'test_args',
        'visualization_args',
        'diffusion_process_args',
        'diffusion_loss_args',
    ]
    for category in categories:
        for key, val in new_cfg[category].items():
            # Enable modification of the OmegaConf structure
            OmegaConf.set_struct(old_cfg[category], True)
            with open_dict(old_cfg[category]):
                # Only add keys that don't exist in the old config
                if key not in old_cfg[category].keys():
                    setattr(old_cfg[category], key, val)
                    rank0_print(f"Added new key: {category}.{key}")
    
    return old_cfg

def update_ckpt_config(new_cfg):
    """
    Update configuration when loading from a checkpoint.
    
    This function handles the process of:
    1. Loading a model from checkpoint
    2. Extracting its configuration
    3. Adding any new configuration keys from the current run
    4. Updating values from the current configuration
    5. Setting checkpoint path and experiment name
    
    Args:
        new_cfg (DictConfig): Current run configuration
        
    Returns:
        DictConfig: Merged configuration combining checkpoint config and current config
    """
    # Load the model from checkpoint to get its configuration
    model = NMRTo3DStructureElucidation.load_from_checkpoint(new_cfg.general.ckpt_abs_path)
    old_cfg = model.cfg
    
    # Add new keys to the config that might not exist in older versions
    updated_cfg = add_config_with_new_keys(old_cfg, new_cfg)
    
    # Override all configuration values with those from the current run
    for category in old_cfg:
        for arg in new_cfg[category]:
            updated_cfg[category][arg] = new_cfg[category][arg]
    
    # Update the checkpoint path in the configuration
    updated_cfg.general.ckpt_abs_path = new_cfg.general.ckpt_abs_path
    
    # Append the mode (train/test) to the experiment name for clarity
    updated_cfg.general.experiment_name = new_cfg.general.experiment_name + "_" + new_cfg.general.mode
    
    return updated_cfg