import torch
from torch import nn
from typing import Dict, Optional, Union, List
from omegaconf import ListConfig
from lightning.pytorch.utilities import rank_zero_only

from src.model.modules.layers import TimestepEmbedder, DiTBlock, FinalLayer
from src.model.modules.embedders import NMRSpectraEmbedder

# Atom Diffusion Transformer 
# Based on DiT: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L145
class DiffusionModuleTransformer(nn.Module):
    """
    Diffusion Transformer (DiT) for molecular structure generation conditioned on NMR spectra and chemical formula (atom type one-hot encoding).
    
    This model takes noisy atom coordinates, atom features (one-hot), and conditioning information (NMR spectra)
    to predict the denoising update.
    """
    def __init__(
        self,
        in_atom_feature_size: int = 10,
        out_atom_coords_size: int = 3,
        condition: str = "H1C13NMRSpectrum",
        in_condition_size: Union[int, List[int], ListConfig] = [10000, 80],
        max_n_atoms: int = 300,
        drop_transform: str = "zero",
        n_blocks: int = 10,
        n_heads: int = 8,
        hidden_size: int = 512,
        mlp_ratio: float = 4.0,
        embedder_args: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize the Diffusion Transformer.

        Args:
            in_atom_feature_size: Dimension of input atom features (i.e., one-hot encoding for atom types).
            out_atom_coords_size: Dimension of output coordinates (usually 3).
            condition: Type of conditioning ("H1NMRSpectrum", "C13NMRSpectrum", "H1C13NMRSpectrum").
            in_condition_size: Input dimension(s) for the conditioning spectra. [int] for single spectrum, [int, int] for H1+C13.
            max_n_atoms: Maximum number of atoms supported (for positional embedding).
            drop_transform: Strategy for dropping conditioning during classifier-free guidance training ("zero").
            n_blocks: Number of DiT blocks.
            n_heads: Number of attention heads.
            hidden_size: Hidden dimension size.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim.
            embedder_args: Arguments for the NMR spectra embedder.
        """
        super().__init__()

        # Atom embedding
        # Projects concatenated [noisy_coords, atom_features] to hidden_size
        # Copied from https://github.com/acharkq/NExT-Mol/blob/6ab80ae3adb487a0c689a9a24998189784c935da/model/diffusion_model_dgt.py#L604
        self.x_embedder = nn.Sequential(
            nn.Linear(in_atom_feature_size + 3, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        
        # Time (noise level) embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.condition = condition
        if condition in ["H1NMRSpectrum", "C13NMRSpectrum", "H1C13NMRSpectrum"]:
            use_hnmr = "H1" in condition
            use_cnmr = "C13" in condition
            hnmr_dim = 0
            cnmr_dim = 0
            
            # Determine input dimensions based on the condition type.
            if use_hnmr and use_cnmr:
                # Expecting in_condition_size = [hnmr_dim, cnmr_dim]
                assert isinstance(in_condition_size, (list, tuple, ListConfig)), \
                    f"Expected in_condition_size to be a list/tuple/ListConfig for {condition}, but got type {type(in_condition_size)}"
                hnmr_dim = in_condition_size[0]
                cnmr_dim = in_condition_size[1]
            elif use_hnmr:
                assert isinstance(in_condition_size, int), \
                    f"Expected in_condition_size to be an integer for {condition}, but got {in_condition_size}"
                hnmr_dim = in_condition_size
            elif use_cnmr:
                assert isinstance(in_condition_size, int), \
                    f"Expected in_condition_size to be an integer for {condition}, but got {in_condition_size}"
                cnmr_dim = in_condition_size
            else:
                raise ValueError("At least use one of 1H and 13C spectra.")

            self.y_embedder = NMRSpectraEmbedder(
                use_hnmr=use_hnmr,
                use_cnmr=use_cnmr,
                hnmr_dim=hnmr_dim,
                cnmr_dim=cnmr_dim,
                hidden_dim=embedder_args['hidden_dim'],
                output_dim=hidden_size,
                dropout=embedder_args['dropout'],
                pooling=embedder_args['pooling'],
                tokenizer_args=embedder_args['tokenizer_args'],
                transformer_args=embedder_args['transformer_args'],
            ) 
        else:
            raise NotImplementedError(f"Condition embedding {condition} not implemented.")
        
        # Positional encoding (currently unused in calculation but kept for compatibility/future use)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_n_atoms, hidden_size), requires_grad=False)
    
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, n_heads, mlp_ratio=mlp_ratio) for _ in range(n_blocks)
        ])
        
        self.final_layer = FinalLayer(hidden_size, out_atom_coords_size)
        self.initialize_weights()
        
        self.drop_transform = drop_transform
        
    def initialize_weights(self):
        """Initialize weights for DiT blocks and embeddings."""
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        rank_zero_only(print)('DiffusionModuleTransformer initialized weights.')
        
    def forward(
        self,
        r_noisy: torch.Tensor,
        times: torch.Tensor,
        model_inputs: Dict[str, torch.Tensor],
        multiplicity: int = 1,
        guidance_scale: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the score model.

        Args:
            r_noisy: Noisy atom coordinates (B, N, 3).
            times: Time embeddings/noise levels (B,).
            model_inputs: Dictionary containing 'atom_mask', 'atom_one_hot', 'condition'.
            multiplicity: Number of samples per input (used for expansion).
            guidance_scale: Scale for classifier-free guidance.

        Returns:
            Dict containing 'r_update' (predicted update/score).
        """
        # Expand inputs based on multiplicity (number of samples per condition)
        atom_mask = model_inputs["atom_mask"].repeat_interleave(multiplicity, 0).bool() # (B, max_n_atoms), B = batch_size * multiplicity
        padded_atom_mask = atom_mask[..., None] # (B, max_n_atoms, 1)
        atom_one_hot = model_inputs["atom_one_hot"].repeat_interleave(multiplicity, 0)
        condition = model_inputs["condition"].repeat_interleave(multiplicity, 0)
        
        # Handle Classifier-Free Guidance (CFG)
        if guidance_scale != 0.0: 
            # Duplicate inputs for conditional and unconditional passes
            r_noisy = torch.cat([r_noisy, r_noisy], dim=0)
            times = torch.cat([times, times], dim=0)
            atom_mask = torch.cat([atom_mask, atom_mask], dim=0)
            padded_atom_mask = torch.cat([padded_atom_mask, padded_atom_mask], dim=0)
            atom_one_hot = torch.cat([atom_one_hot, atom_one_hot], dim=0)
            
            # Create unconditional input by zeroing out the condition
            if self.drop_transform == "zero":
                uncondition = torch.zeros_like(condition)
            else:
                raise ValueError(f"Unsupported drop_transform: {self.drop_transform}")
            
            # Concatenate conditional and unconditional inputs
            condition = torch.cat([condition, uncondition], dim=0)
        
        # Input embedding: Concatenate noisy coords and atom features
        x = torch.cat([r_noisy, atom_one_hot], dim=-1) # (B, max_n_atoms, in_atom_feature_size + 3)
        
        # Apply mask
        x = self.x_embedder(x) * padded_atom_mask
        
        # Time embedding
        t = self.t_embedder(times) # (B, hidden_size)
        
        # Condition embedding
        y = self.y_embedder(condition)
        
        # Combine time and condition embeddings
        c = t + y
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, atom_mask)
            
        # Final projection
        x = self.final_layer(x, c) # (B, max_n_atoms, out_atom_coords_size=3)
        x = x * padded_atom_mask
        
        # Apply Classifier-Free Guidance
        if guidance_scale != 0.0: 
            # Split output into guided and unconditioned parts
            x_guided = x[:x.shape[0] // 2]
            x_unconditioned = x[x.shape[0] // 2:]
            # Apply guidance formula: guided + scale * (guided - unconditioned)
            x = (1 + guidance_scale) * x_guided - guidance_scale * x_unconditioned
        
        return dict(
            r_update=x,
        )