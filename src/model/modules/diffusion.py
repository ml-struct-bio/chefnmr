# Started from code from https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/modules/diffusion.py
# MIT License

# Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch.nn import Module
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, Any
from omegaconf import DictConfig

from einops import rearrange
from math import sqrt

from src.model.modules import score_models
from src.model.modules.utils import log, default, center_random_augmentation, smooth_lddt_loss

class AtomDiffusion(Module):
    """
    Atom diffusion module implementing EDM and AlphaFold3-style diffusion.
    
    This module handles the forward diffusion process (adding noise) during training
    and the reverse diffusion process (denoising) during sampling.
    
    References:
    - Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models.
    - Abramson et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3.
    """
    def __init__(
        self,
        score_model_args: DictConfig,
        train_sigma_distribution_type: str = "af3", 
        sample_sigma_schedule_type: str = "edm",
        sample_gamma_schedule_type: str = "edm",
        num_sampling_steps: int = 50, 
        sigma_min: float = 0.0004, 
        sigma_max: float = 80.0,
        gamma_min: float = 1.0, 
        noise_scale: float = 1.0,
        step_scale: float = 1.0,
        guidance_scale: float = 0.0, # Guidance scale for classifier-free guidance
        synchronize_sigmas: bool = False,
        coordinate_transformation_when_training: str = "centering_rotation_translation",
        edm_args: Optional[Dict] = None,
        train_sigma_args: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialize the AtomDiffusion module.

        Args:
            score_model_args: Configuration for the underlying score model (denoiser).
            train_sigma_distribution_type: Distribution of noise levels during training ('af3' or 'edm').
            sample_sigma_schedule_type: Schedule of noise levels during sampling.
            sample_gamma_schedule_type: Schedule of injected noise (stochasticity) during sampling.
            num_sampling_steps: Number of steps for the reverse diffusion process.
            sigma_min: Minimum noise level.
            sigma_max: Maximum noise level.
            gamma_min: Minimum gamma for stochastic sampling.
            noise_scale: Scaling factor for noise.
            step_scale: Scaling factor for update steps.
            guidance_scale: Scale for classifier-free guidance (0.0 means no guidance).
            synchronize_sigmas: Whether to use the same sigma for all samples in a batch (if multiplicity > 1).
            coordinate_transformation_when_training: Augmentation strategy ('centering_rotation_translation').
            edm_args: Additional arguments for EDM (rho, sigma_data, etc.).
            train_sigma_args: Arguments for training sigma distribution (mean, std).
        """
        super().__init__()
        score_model_name = score_model_args.model_name
        self.score_model = getattr(score_models, score_model_args.model_name)(
            **score_model_args[score_model_name],
            )

        # Hyperparameters
        self.train_sigma_distribution_type = train_sigma_distribution_type
        self.sample_sigma_schedule_type = sample_sigma_schedule_type
        self.sample_gamma_schedule_type = sample_gamma_schedule_type
        self.num_sampling_steps = num_sampling_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.guidance_scale = guidance_scale
        self.synchronize_sigmas = synchronize_sigmas
        self.coordinate_transformation_when_training = coordinate_transformation_when_training
        
        default_edm_args = {
            "sigma_data": 3.0,
            "rho": 7,
            "use_heun_solver": True,
            "gamma_0": 0.8,
        }
        self.edm_args = default(edm_args, default_edm_args)
        default_train_sigma_args = {
            "edm_P_mean": -1.2,
            "edm_P_std": 1.3,
        }
        self.train_sigma_args = default(train_sigma_args, default_train_sigma_args)
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return next(self.score_model.parameters()).device
    
    def float_to_tensor(self, value: Union[float, torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        """Convert a float value to a tensor of shape (batch_size,)."""
        if isinstance(value, float):
            value = torch.full((batch_size,), value, device=device)
        return value
    
    def pad_sigma(self, sigma: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """Pad sigma to match the shape of the model inputs (b, 1, 1)."""
        sigma = self.float_to_tensor(sigma, batch_size, device)
        padded_sigma = rearrange(sigma, "b -> b 1 1")
        return padded_sigma
    
    # --- Preconditioning functions (Karras et al. 2022) ---
    # noisy_atom_coords = a_sigma * atom_coords + b_sigma * gaussian_noise
    def a_sigma(self, sigma):
        return torch.ones_like(sigma) 
    
    def b_sigma(self, sigma):
        return sigma
    
    def interpolate(self, atom_coords, noise, sigma):
        """Interpolate atom coordinates with noise based on sigma."""
        padded_sigma = self.pad_sigma(sigma, atom_coords.shape[0], atom_coords.device)
        return self.a_sigma(padded_sigma) * atom_coords + self.b_sigma(padded_sigma) * noise
    
    # neural_network_input = c_in * atom_coords
    def c_in(self, sigma):
        return 1 / (torch.sqrt(sigma**2 + self.edm_args.sigma_data**2))
    
    def noised_coords_in_network(self, atom_coords, sigma):
        padded_sigma = self.pad_sigma(sigma, atom_coords.shape[0], atom_coords.device)
        return self.c_in(padded_sigma) * atom_coords
    
    def sigma_in_network(self, sigma):
        return log(sigma) * 0.25
    
    # neural_network_target = c_sigma * atom_coords + d_sigma * gaussian_noise
    def c_sigma(self, sigma):
        return sigma / (self.edm_args.sigma_data * torch.sqrt(sigma**2 + self.edm_args.sigma_data**2))
    
    def d_sigma(self, sigma):
        return -self.edm_args.sigma_data / torch.sqrt(sigma**2 + self.edm_args.sigma_data**2)
    
    def net_target(self, atom_coords, noise, sigma):
        padded_sigma = self.pad_sigma(sigma, atom_coords.shape[0], atom_coords.device)
        return self.c_sigma(padded_sigma) * atom_coords + self.d_sigma(padded_sigma) * noise
    
    def sample_sigma_schedule(self, num_sampling_steps=None):
        """Compute the schedule of noise levels (sigmas) for sampling."""
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        if self.sample_sigma_schedule_type in ["edm", "af3"]:
            inv_rho = 1 / self.edm_args.rho
            steps = torch.arange(
                num_sampling_steps, device=self.device, dtype=torch.float32
            )
            sigmas = (
                self.sigma_max**inv_rho
                + steps
                / (num_sampling_steps - 1)
                * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
            ) ** self.edm_args.rho
            if self.sample_sigma_schedule_type == "af3":
                sigmas = sigmas * self.edm_args.sigma_data # NOTE: This is different from EDM paper
            sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas
    
    def sample_gamma_schedule(self, sigmas):
        """Compute the schedule of gamma values for stochastic sampling."""
        return torch.where(sigmas > self.gamma_min, self.edm_args.gamma_0, 0.0)
    
    def sample(
        self,
        model_inputs: Dict[str, torch.Tensor],
        num_sampling_steps: Optional[int] = None,
        multiplicity: int = 1, 
        n_chain_frames: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate samples using the reverse diffusion process.

        Args:
            model_inputs: Dictionary of model inputs (masks, conditions).
            num_sampling_steps: Number of diffusion steps.
            multiplicity: Number of samples to generate per input.
            n_chain_frames: Number of intermediate frames to save for visualization.

        Returns:
            Tuple of (final_atom_coords, atom_coords_chains).
        """
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        
        # Determine which steps to save for the chain visualization
        if num_sampling_steps < n_chain_frames:
            save_chain_indices = torch.arange(0, num_sampling_steps, device=self.device)
        else:
            save_chain_indices = torch.arange(0, num_sampling_steps * n_chain_frames, num_sampling_steps) // n_chain_frames
        save_chain_indices = save_chain_indices.tolist()
        atom_coords_chains = torch.tensor([], device=self.device)
        
        atom_mask = model_inputs["atom_mask"].repeat_interleave(multiplicity, 0)
        shape = (*atom_mask.shape, 3)

        sigmas = self.sample_sigma_schedule(num_sampling_steps)
        gammas = self.sample_gamma_schedule(sigmas)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # Initialize with random noise scaled by the initial sigma
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)

        # Gradually denoise
        for i, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            if i in save_chain_indices:
                atom_coords_chains = torch.cat([atom_coords_chains, atom_coords[:, None, ...]], dim=1)

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            eps = torch.randn(shape, device=self.device)
            
            atom_coords_next = self._sample_one_step_edm(
                model_inputs=model_inputs,
                multiplicity=multiplicity,
                atom_coords=atom_coords,
                sigma_tm=sigma_tm,
                sigma_t=sigma_t,
                gamma=gamma,
                eps=eps,
            )
                
            atom_coords = atom_coords_next
        
        # Append final state
        atom_coords_chains = torch.cat([atom_coords_chains, atom_coords[:, None, ...]], dim=1)
        
        return atom_coords, atom_coords_chains
    
    def _sample_one_step_edm(self, model_inputs, multiplicity, atom_coords, sigma_tm, sigma_t, gamma, eps):
        """Perform one step of EDM sampling (Euler or Heun solver)."""
        # 1. Stochastic sampling step (add noise)
        t_hat = sigma_tm * (1 + gamma)
        eps = eps * self.noise_scale * sqrt(t_hat**2 - sigma_tm**2)
        noisy_atom_coords = atom_coords + eps
        
        # 2. Predict denoised coordinates
        with torch.no_grad():
            net_out = self.neural_network_forward(
                noisy_atom_coords,
                t_hat,
                network_condition_kwargs=dict(
                    multiplicity=multiplicity,
                    model_inputs=model_inputs,
                    guidance_scale=self.guidance_scale,
                ),
            )
        
        denoised_atom_coords = self.predict_denoised_atom_coords(
            noisy_atom_coords,
            net_out,
            t_hat,
        )
        
        # 3. Calculate velocity (d_ode)
        velocity = self.predict_velocity(
            noisy_atom_coords=noisy_atom_coords,
            net_out=net_out,
            sigma=t_hat,
            denoised_atom_coords=denoised_atom_coords,
        )
        
        # 4. Euler step
        atom_coords_next = (
            noisy_atom_coords
            + self.step_scale * (sigma_t - t_hat) * velocity
        )

        # 5. Optional Heun step (2nd order correction)
        if self.edm_args.use_heun_solver and sigma_t > 0:
            with torch.no_grad():
                net_out = self.neural_network_forward(
                    atom_coords_next,
                    sigma_t,
                    network_condition_kwargs=dict(
                        multiplicity=multiplicity,
                        model_inputs=model_inputs,
                        guidance_scale=self.guidance_scale,
                    ),
                )
            
            denoised_atom_coords = self.predict_denoised_atom_coords(
                atom_coords_next,
                net_out,
                sigma_t,
            )
            
            velocity_next = self.predict_velocity(
                noisy_atom_coords=atom_coords_next,
                net_out=net_out,
                sigma=sigma_t,
                denoised_atom_coords=denoised_atom_coords,
            )
            
            atom_coords_next = (
                noisy_atom_coords
                + 0.5 * self.step_scale * (sigma_t - t_hat) * velocity
                + 0.5 * self.step_scale * (sigma_t - t_hat) * velocity_next
            )
            
        return atom_coords_next

    def neural_network_forward(
        self,
        noisy_atom_coords: torch.Tensor,
        sigma: Union[float, torch.Tensor],
        network_condition_kwargs: dict,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the score model with preconditioning."""
        batch_size, device = noisy_atom_coords.shape[0], noisy_atom_coords.device
        sigma = self.float_to_tensor(sigma, batch_size, device)
        net_out = self.score_model(
            r_noisy=self.noised_coords_in_network(noisy_atom_coords, sigma),
            times=self.sigma_in_network(sigma),
            **network_condition_kwargs,
        )
        return net_out
    
    def predict_denoised_atom_coords(
        self,
        noisy_atom_coords,
        net_out,
        sigma,
    ):
        """Recover x_0 (denoised coords) from x_t and network output."""
        # x_0 = (d*x_t - b*nn(x_t, sigma)) / (ad - bc)
        # Simplified based on EDM preconditioning choices
        batch_size, device = noisy_atom_coords.shape[0], noisy_atom_coords.device
        padded_sigma = self.pad_sigma(sigma, batch_size, device)
        
        denoised_atom_coords = self.d_sigma(padded_sigma) * noisy_atom_coords - self.b_sigma(padded_sigma) * net_out["r_update"]
        denoised_atom_coords = denoised_atom_coords * -self.edm_args.sigma_data * self.c_in(padded_sigma)
        
        return denoised_atom_coords
    
    def predict_velocity(
        self,
        noisy_atom_coords,
        net_out,
        sigma,
        denoised_atom_coords=None,
    ):
        """Compute the velocity (time derivative) for the ODE solver."""
        batch_size, device = noisy_atom_coords.shape[0], noisy_atom_coords.device
        padded_sigma = self.pad_sigma(sigma, batch_size, device)
        
        if denoised_atom_coords is None:
            denoised_atom_coords = self.predict_denoised_atom_coords(noisy_atom_coords, net_out, sigma)
        velocity = (noisy_atom_coords - denoised_atom_coords) / padded_sigma
        
        return velocity
    
    def train_sigma_distribution(self, batch_size):
        """Sample noise levels for training."""
        if self.train_sigma_distribution_type in ["edm", "af3"]:
            sigmas = (
                self.train_sigma_args.edm_P_mean
                + self.train_sigma_args.edm_P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
            if self.train_sigma_distribution_type == "af3":
                sigmas = sigmas * self.edm_args.sigma_data
        else:
            raise ValueError(
                f"Unknown train_sigma_distribution_type: {self.train_sigma_distribution_type}"
            )
        return sigmas
    
    def forward(
        self,
        model_inputs: Dict[str, torch.Tensor],
        atom_coords: torch.Tensor,
        multiplicity: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass: add noise and predict denoised coordinates.
        """
        batch_size = atom_coords.shape[0]

        if self.synchronize_sigmas:
            sigmas = self.train_sigma_distribution(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            sigmas = self.train_sigma_distribution(batch_size * multiplicity)

        atom_coords = atom_coords.repeat_interleave(multiplicity, 0)
        atom_mask = model_inputs["atom_mask"].repeat_interleave(multiplicity, 0)
        
        if self.coordinate_transformation_when_training == "centering_rotation_translation":
            atom_coords = center_random_augmentation(
                atom_coords,
                atom_mask,
                centering=True,
                augmentation=True,
            )

        noise = torch.randn_like(atom_coords)
        noisy_atom_coords = self.interpolate(atom_coords, noise, sigmas)

        net_out = self.neural_network_forward(
            noisy_atom_coords,
            sigmas,
            network_condition_kwargs=dict(
                model_inputs=model_inputs,
                multiplicity=multiplicity,
                guidance_scale=0.0,  # No guidance during training
            ),
        )
        
        denoised_atom_coords = self.predict_denoised_atom_coords(
            noisy_atom_coords,
            net_out,
            sigmas,
        )

        return dict(
            noisy_atom_coords=noisy_atom_coords,
            denoised_atom_coords=denoised_atom_coords,
            sigmas=sigmas,
            aligned_true_atom_coords=atom_coords,
            net_out=net_out["r_update"],
            noise=noise,
        )
    
    def compute_loss(
        self,
        model_inputs: Dict[str, torch.Tensor],
        dict_out: Dict[str, torch.Tensor],
        multiplicity: int = 1,
        add_smooth_lddt_loss: bool = True,
        lddt_loss_threshold: list = [0.5, 1.0, 2.0, 4.0],
    ) -> Dict[str, Any]:
        """
        Compute the training loss (MSE + optional Smooth LDDT).
        """
        denoised_atom_coords = dict_out["denoised_atom_coords"]
        sigmas = dict_out["sigmas"]
        atom_mask = model_inputs["atom_mask"].repeat_interleave(multiplicity, 0)
        align_weights = denoised_atom_coords.new_ones(denoised_atom_coords.shape[:2]) # Weight for each atom is 1 in small molecule case

        atom_coords_aligned_ground_truth = dict_out["aligned_true_atom_coords"]

        # Cast back
        atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
            denoised_atom_coords
        )

        # Compute MSE Loss
        net_target = self.net_target(atom_coords_aligned_ground_truth, dict_out["noise"], sigmas)
        mse_loss = ((dict_out["net_out"] - net_target) ** 2).sum(dim=-1)

        mse_loss = torch.sum(
            mse_loss * align_weights * atom_mask, dim=-1
        ) / torch.sum(3 * align_weights * atom_mask, dim=-1)

        # weight by sigma factor (implicit in EDM formulation if preconditioning is used correctly)
        mse_loss = mse_loss.mean()

        total_loss = mse_loss

        # Proposed auxiliary smooth lddt loss
        lddt_loss = self.zero
        if add_smooth_lddt_loss:
            lddt_loss = smooth_lddt_loss(
                pred_coords=denoised_atom_coords,
                true_coords=dict_out["aligned_true_atom_coords"],
                is_nucleotide=torch.zeros_like(atom_mask),
                coords_mask=atom_mask,
                lddt_loss_threshold=lddt_loss_threshold,
                multiplicity=multiplicity,
            )

            total_loss = total_loss + lddt_loss

        loss_breakdown = dict(
            mse_loss=mse_loss,
            smooth_lddt_loss=lddt_loss,
        )

        return dict(loss=total_loss, loss_breakdown=loss_breakdown)