import torch
from torch import nn
from einops import rearrange
from typing import Optional
import numpy as np
import math
from src.model.modules.utils import get_1d_sincos_pos_embed_from_grid


# -----------------------------------------------------------------------------
# Tokenizers
# -----------------------------------------------------------------------------
class SpectraTokenizerPatch1D(nn.Module):
    """
    1-D Patch Tokenizer.
    
    Splits a 1D sequence into fixed-size patches and projects them linearly.
    Implemented using `torch.unfold`.

    Args:
        patch_size (int): The size of each patch (P).
        stride (int): The stride between patches.
        hidden_dim (int): The dimension of the projected tokens (D).
    """
    def __init__(self, patch_size: int, stride: int, hidden_dim: int):
        super().__init__()
        self.patch_size = patch_size # P
        self.stride = stride
        self.proj = nn.Linear(patch_size, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, L).
        
        Returns:
            torch.Tensor: Output tokens of shape (B, T, D).
        """
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)  # (B, T, P) - T: num tokens, P: patch size
        x = self.proj(patches)                                                  # (B, T, D) - D: hidden dim
        return x

    @staticmethod
    def num_tokens(input_size: int, patch_size: int, stride: int) -> int:
        """Calculates the number of tokens (T) produced by the patch embedding."""
        return (input_size - patch_size) // stride + 1
    
# Modify from https://github.com/MarklandGroup/NMR2Struct/blob/3b885934912489dbb2d494ce5bd462d6e46e422d/nmr/networks/embeddings.py#L95
class SpectraTokenizerConv1D(nn.Module):
    """
    1-D Convolutional Tokenizer.
    
    Applies a sequence of Conv1d -> ReLU -> MaxPool1d blocks, followed by a 
    linear projection to the hidden dimension.

    Args:
        input_size (int): Length of the input sequence (L).
        hidden_dim (int): Dimension of the output tokens (D).
        pool_sizes (List[int]): Pooling kernel sizes for each layer.
        kernel_sizes (List[int]): Convolution kernel sizes for each layer.
        out_channels (List[int]): Output channel dimensions for each layer.
    """
    def __init__(self, input_size, hidden_dim, pool_sizes, kernel_sizes, out_channels):
        super().__init__()

        # Validate parameters
        num_layers = len(pool_sizes)
        assert num_layers > 0, "Must specify at least one convolutional layer."
        assert len(kernel_sizes) == num_layers, "kernel_sizes must have the same length as pool_sizes."
        assert len(out_channels) == num_layers, "out_channels must have the same length as pool_sizes."

        self.conv_blocks = nn.ModuleList()
        in_channel = 1 # Initial input channel for the first conv layer
        for i in range(num_layers):
            block = nn.Sequential(
                nn.Conv1d(in_channels=in_channel,          # Input channels
                          out_channels=out_channels[i],    # Output channels (C_out)
                          kernel_size=kernel_sizes[i],     # Kernel size (K)
                          stride=1,                        # Stride (S)
                          padding="valid"),                # Padding (P=0)
                nn.ReLU(),
                nn.MaxPool1d(pool_sizes[i])                # Max pooling kernel size
            )
            self.conv_blocks.append(block)
            in_channel = out_channels[i] # Update in_channel for the next block

        self.linear_after_conv = nn.Linear(out_channels[-1], hidden_dim) # Project C_last -> D

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, L) or (B, 1, L).
            
        Returns:
            torch.Tensor: Output tokens of shape (B, T, D).
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1) # Add channel dimension: (B, L) -> (B, 1, L)

        # Apply convolutional blocks sequentially
        for block in self.conv_blocks:
            x = block(x) # Output of each block: (B, C_i, L_i)

        # x shape is now (B, C_last, L_out) - C_last: channels from last conv, L_out: final length
        x = torch.transpose(x, 1, 2) # (B, L_out, C_last)
        x = self.linear_after_conv(x) # (B, L_out, D) - Project C_last to D (hidden_dim)
        # Output x: (B, T, D) where T = L_out
        return x
    
    # From https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    @staticmethod
    def _calculate_dim_after_conv( 
                                  L_in: int,      # Input length
                                  kernel: int,    # Kernel size (K)
                                  padding: int,   # Padding (P)
                                  dilation: int,  # Dilation (Dil)
                                  stride: int     # Stride (S)
                                  ) -> int:       # Output length (L_out)
        """Calculates the output length after a Conv1d layer."""
        # Formula: floor(((L_in + 2*P - Dil*(K-1) - 1) / S) + 1)
        numerator = L_in + (2 * padding) - (dilation * (kernel - 1)) - 1
        return math.floor(
            (numerator/stride) + 1
        )
    
    # From https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
    # and https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html
    @staticmethod
    def _calculate_dim_after_pool(
                                  pool_variation: str, # 'max' or 'avg'
                                  L_in: int,           # Input length
                                  kernel: int,         # Kernel size (K)
                                  padding: int,        # Padding (P)
                                  dilation: int,       # Dilation (Dil)
                                  stride: int          # Stride (S)
                                  ) -> int:            # Output length (L_out)
        """Calculates the output length after a MaxPool1d or AvgPool1d layer."""
        if pool_variation == 'max':
            # Formula: floor(((L_in + 2*P - Dil*(K-1) - 1) / S) + 1)
            numerator = L_in + (2 * padding) - (dilation * (kernel - 1)) - 1
            return math.floor(
                (numerator/stride) + 1
            )
        elif pool_variation == 'avg':
            # Formula: floor(((L_in + 2*P - K) / S) + 1)
            numerator = L_in + (2 * padding) - kernel
            return math.floor(
                (numerator/stride) + 1
            )
    
    @staticmethod
    def num_tokens(input_size: int, kernel_sizes: list, pool_sizes: list) -> int:
        '''Computes the final sequence length (T) after all convolution + pooling operations.'''
        L_current = input_size # Start with initial length L
        block_args = zip(kernel_sizes, pool_sizes)
        for conv_kernel, pool_kernel in block_args:
            # After Conv1d
            L_current = SpectraTokenizerConv1D._calculate_dim_after_conv(
                L_in=L_current, 
                kernel=conv_kernel, 
                padding=0, 
                dilation=1, 
                stride=1
            )
            # After MaxPool1d
            L_current = SpectraTokenizerConv1D._calculate_dim_after_pool(
                pool_variation='max', 
                L_in=L_current, 
                kernel=pool_kernel, 
                padding=0, 
                dilation=1, 
                stride=pool_kernel
            )
        return L_current # Final output sequence length T


# -----------------------------------------------------------------------------
# Transformer primitives
# -----------------------------------------------------------------------------
class PreNorm(nn.Module):
    """Applies LayerNorm before the function."""
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class Attention(nn.Module):
    """Multi-head Self Attention."""
    def __init__(self, dim: int, heads: int , dim_head: Optional[int] = None):
        super().__init__()
        if dim_head is None:
            assert dim % heads == 0, "Dimension must be divisible by number of heads"
            dim_head = dim // heads
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.attn = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (B, T, D)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attn(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FeedForward(nn.Module):
    """Point-wise FeedForward Network (MLP)."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoder(nn.Module):
    """Standard Transformer Encoder Stack."""
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_ratio: int, dropout: float):
        super().__init__()
        mlp_dim = dim * mlp_ratio
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                    ]
                )
            )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return self.final_norm(x)


# -----------------------------------------------------------------------------
# Pooling
# -----------------------------------------------------------------------------
class AttnPoolToken(nn.Module):
    """
    Attention-based pooling using a learnable CLS token.
    
    Appends a CLS token to the sequence, applies attention, and extracts the CLS token.
    """
    def __init__(self, dim, out_dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)
        self.attn = Attention(dim, heads, dim_head)
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, out_dim)
        )

    def forward(self, x):
        cls = self.cls.expand(x.size(0), -1, -1)
        y = torch.cat([cls, x], dim=1)
        y = self.attn(y)[:, 0]       # pooled token
        return self.proj(y)

# -----------------------------------------------------------------------------
# Main embedding module
# -----------------------------------------------------------------------------
class NMRSpectraEmbedder(nn.Module):
    """
    Embedder for NMR Spectra (1H and/or 13C).
    
    Processes 1D NMR spectra using configurable tokenizers (Conv1D, Patch, or Embedding),
    adds positional and type encodings, processes with a Transformer Encoder, and pools
    the output to a fixed-size vector.
    """
    def __init__(
        self,
        *,
        use_hnmr: bool = True,
        use_cnmr: bool = False,
        hnmr_dim: int = 10000, # 10000 or 28000
        cnmr_dim: int = 10000, # 80 or 10000
        hidden_dim: int = 256,
        output_dim: int = 768,
        dropout: float = 0.1,
        pooling: str = "flatten", # "flatten", "meanmlp", "attn", "conv"
        tokenizer_args: dict = None,
        transformer_args: dict = None,
    ):
        super().__init__()
        self.use_hnmr = use_hnmr
        self.use_cnmr = use_cnmr
        self.hnmr_dim = hnmr_dim
        self.cnmr_dim = cnmr_dim
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        self.hidden_dim = hidden_dim

        # --- Set default arguments if not provided ---
        default_tokenizer_args = {
            "h_tokenizer": "conv", "c_tokenizer": "conv",
            "h_pool_sizes": [8, 12], "h_kernel_sizes": [5, 9], "h_out_channels": [64, 128],
            "c_pool_sizes": [8, 12], "c_kernel_sizes": [5, 9], "c_out_channels": [64, 128],
            "h_patch_size": 256, "h_patch_stride": 128,
            "c_patch_size": 256, "c_patch_stride": 128,
            "h_mask_token": False,  # Use mask token for missing H1 NMR spectra
            "c_mask_token": False,  # Use mask token for missing C13 NMR spectra
        }
        tokenizer_args = {**default_tokenizer_args, **(tokenizer_args or {})}

        default_transformer_args = {
            "pos_enc": "learnable", "type_enc": True,
            "depth": 4, "heads": 8,
            "dim_head": None, "mlp_ratio": 4,
        }
        transformer_args = {**default_transformer_args, **(transformer_args or {})}

        # --- Extract arguments ---
        self.h_tokenizer = tokenizer_args['h_tokenizer']
        self.c_tokenizer = tokenizer_args['c_tokenizer']
        self.use_h_mask_token = tokenizer_args['h_mask_token']
        self.use_c_mask_token = tokenizer_args['c_mask_token']

        pos_enc = transformer_args['pos_enc']
        type_enc = transformer_args['type_enc']
        depth = transformer_args['depth']
        heads = transformer_args['heads']
        dim_head = transformer_args['dim_head']
        mlp_ratio = transformer_args['mlp_ratio']

        # Tokenizer initialization using helper method
        self.h_embed, self.h_token_num = None, 0
        self.c_embed, self.c_token_num = None, 0

        if use_hnmr:
            self.h_embed, self.h_token_num = self._initialize_tokenizer(
                tokenizer_type=self.h_tokenizer,
                input_dim=hnmr_dim,
                hidden_dim=hidden_dim,
                pool_sizes=tokenizer_args['h_pool_sizes'],
                kernel_sizes=tokenizer_args['h_kernel_sizes'],
                out_channels=tokenizer_args['h_out_channels'],
                patch_size=tokenizer_args['h_patch_size'],
                stride=tokenizer_args['h_patch_stride']
            )
            if self.use_h_mask_token:
                # Initialize learnable [MASK] token for HNMR
                self.h_mask_token = nn.Embedding(2, hidden_dim)
                nn.init.normal_(self.h_mask_token.weight, std=0.02)
                self.h_token_num += 1  # Add mask token to HNMR token count

        if use_cnmr:
            self.c_embed, self.c_token_num = self._initialize_tokenizer(
                tokenizer_type=self.c_tokenizer,
                input_dim=cnmr_dim,
                hidden_dim=hidden_dim,
                pool_sizes=tokenizer_args['c_pool_sizes'],
                kernel_sizes=tokenizer_args['c_kernel_sizes'],
                out_channels=tokenizer_args['c_out_channels'],
                patch_size=tokenizer_args['c_patch_size'],
                stride=tokenizer_args['c_patch_stride']
            )
            # Initialize learnable [MASK] token for CNMR
            if self.use_c_mask_token:
                self.c_mask_token = nn.Embedding(2, hidden_dim)
                nn.init.normal_(self.c_mask_token.weight, std=0.02)
                self.c_token_num += 1  # Add mask token to CNMR token count

        # Positional encoding initialization
        total_tokens = self.h_token_num + self.c_token_num
        if pos_enc == "sincos":
            self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, hidden_dim), requires_grad=False)
            pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.pos_embed.shape[-1], 
                np.arange(total_tokens)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            self.pos_encode = lambda x: x + self.pos_embed
        elif pos_enc == "learnable":
            self.learnable_pos_embed = nn.Parameter(torch.zeros(1, total_tokens, hidden_dim))
            nn.init.normal_(self.learnable_pos_embed, std=0.02)
            self.pos_encode = lambda x: x + self.learnable_pos_embed
        elif pos_enc is None:
            # No positional encoding
            self.pos_encode = nn.Identity()
        else:
            raise ValueError(f"Unknown positional encoding: {pos_enc}")

        # Type embedding to distinguish 1H / 13C
        if use_hnmr and use_cnmr and type_enc:
            # Type embedding for both HNMR and CNMR
            self.type_embedding = nn.Embedding(2, hidden_dim)
            nn.init.normal_(self.type_embedding.weight, std=0.02)
        else:
            self.type_embedding = None

        # Transformer encoder
        if depth > 0:
            self.transformer = TransformerEncoder(
                dim=hidden_dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
        else:
            self.transformer = nn.Identity()

        # Pool -> projection
        if pooling == "flatten":
            flatten_dim = (total_tokens) * hidden_dim
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(flatten_dim),
                nn.Dropout(dropout),
                nn.Linear(flatten_dim, output_dim),
            )
        elif pooling == "attn":
            self.head = AttnPoolToken(hidden_dim, output_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")


    def _initialize_tokenizer(self, tokenizer_type, input_dim, hidden_dim, **kwargs):
        """Helper function to initialize tokenizer and calculate token number."""
        if tokenizer_type == "conv":
            embed_layer = SpectraTokenizerConv1D(
                input_size=input_dim,
                hidden_dim=hidden_dim,
                pool_sizes=kwargs['pool_sizes'],
                kernel_sizes=kwargs['kernel_sizes'],
                out_channels=kwargs['out_channels']
            )
            num_tokens = SpectraTokenizerConv1D.num_tokens(
                input_size=input_dim,
                kernel_sizes=kwargs['kernel_sizes'],
                pool_sizes=kwargs['pool_sizes']
            )
        elif tokenizer_type == "patch":
            embed_layer = SpectraTokenizerPatch1D(
                patch_size=kwargs['patch_size'],
                stride=kwargs['stride'],
                hidden_dim=hidden_dim
            )
            num_tokens = SpectraTokenizerPatch1D.num_tokens(
                input_size=input_dim,
                patch_size=kwargs['patch_size'],
                stride=kwargs['stride']
            )
        elif tokenizer_type == "embed":
            # For binned binary input (e.g., 80-bin 13C NMR)
            embed_layer = nn.Embedding(input_dim + 1, hidden_dim, padding_idx=0)
            num_tokens = input_dim
            nn.init.normal_(embed_layer.weight, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        
        return embed_layer, num_tokens

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _embed_spectrum(self, x: torch.Tensor, tokenizer_type: str, embed_layer: nn.Module, input_dim: int) -> torch.Tensor:
        """Helper function to embed a single spectrum type based on tokenizer."""
        
        if tokenizer_type == "conv" or tokenizer_type == "patch":
            # Conv and Patch tokenizers expect (B, L) or (B, 1, L) and handle it internally
            return embed_layer(x)
        elif tokenizer_type == "embed":
            # Assumes x is a binary vector (B, L) where L = input_dim.
            # Example: For 80-bin 13C NMR, x is (B, 80) with 0s and 1s.
            # Create indices [1, 2, ..., input_dim]
            indices = torch.arange(1, input_dim + 1, device=x.device, dtype=torch.long) # Shape (L,)
            # Since x is binary (0s and 1s), multiplication acts as a selector:
            # - If x[i] is 0, result is 0.
            # - If x[i] is 1, result is indices[i] (which is i+1).
            # The result is a tensor where non-zero entries correspond to the 1-based index of the 'active' bins.
            # Shape remains (B, L).
            x = x.long() * indices 
            # Pass the resulting indices (including 0s for inactive bins) to the embedding layer.
            # The embedding layer uses padding_idx=0, so 0s map to a padding vector.
            return embed_layer(x)
        else:
            # This path should ideally not be reached due to checks in __init__
            raise ValueError(f"Unknown tokenizer type '{tokenizer_type}' during embedding.")

    def _embed_hnmr(self, x: torch.Tensor) -> torch.Tensor:
        """Embed the HNMR spectrum using the appropriate tokenizer."""
        if self.h_embed is None:
             raise RuntimeError("HNMR embedding layer is not initialized. Check use_hnmr flag.")
        h_embed = self._embed_spectrum(x, self.h_tokenizer, self.h_embed, self.hnmr_dim)
        if not self.use_h_mask_token:
            return h_embed  # No mask token, return directly
        else:
            h_missing_mask = (x == 0).all(dim=1) # ! (B,) - True if all bins are 0
            h_missing_tokens = self.h_mask_token(h_missing_mask.long()) # (B, D)
            h_missing_tokens = h_missing_tokens.unsqueeze(1)  # (B, 1, D)
            h_embed = torch.cat([h_missing_tokens, h_embed], dim=1)
            return h_embed  # (B, N_h + 1, D) where N_h is the number of HNMR tokens
        

    def _embed_cnmr(self, x: torch.Tensor) -> torch.Tensor:
        """Embed the CNMR spectrum using the appropriate tokenizer."""
        if self.c_embed is None:
             raise RuntimeError("CNMR embedding layer is not initialized. Check use_cnmr flag.")
        c_embed = self._embed_spectrum(x, self.c_tokenizer, self.c_embed, self.cnmr_dim)
        if not self.use_c_mask_token:
            return c_embed
        else:
            c_missing_mask = (x == 0).all(dim=1) # ! (B,) - True if all bins are 0
            c_missing_tokens = self.c_mask_token(c_missing_mask.long()) # (B, D)
            c_missing_tokens = c_missing_tokens.unsqueeze(1)  # (B, 1, D)
            c_embed = torch.cat([c_missing_tokens, c_embed], dim=1)
            return c_embed  # (B, N_c + 1, D) where N_c is the number of CNMR tokens

    def _separate_spectra_components(self, x: torch.Tensor):
        """Splits concatenated input into HNMR and CNMR components."""
        hnmr_x = x[:, :self.hnmr_dim]
        cnmr_x = x[:, self.hnmr_dim:self.hnmr_dim + self.cnmr_dim]
        return hnmr_x, cnmr_x

    def forward(
        self,
        x
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L_h + L_c), containing 
                              concatenated HNMR and CNMR spectra.
        
        Returns:
            torch.Tensor: Embedding vector of shape (B, output_dim).
        """
        hnmr_spectra, cnmr_spectra = self._separate_spectra_components(x)
        
        tokens = []
        type_ids = []
        if self.use_hnmr:
            t = self._embed_hnmr(hnmr_spectra)  # (B, N_h, D)
            tokens.append(t)
            type_ids.append(torch.zeros(t.size(1), device=t.device, dtype=torch.long))
        if self.use_cnmr:
            t = self._embed_cnmr(cnmr_spectra) # (B, N_c, D)
            tokens.append(t)
            type_ids.append(torch.ones(t.size(1), device=t.device, dtype=torch.long))

        x = torch.cat(tokens, dim=1)  # (B, N, D)
        x = self.pos_encode(x)

        if self.type_embedding is not None:
            type_ids_tensor = torch.cat(type_ids)  # (N,)
            type_emb = self.type_embedding(type_ids_tensor)  # (N, D)
            batch_size = x.size(0)
            type_emb = type_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, D)
            x = x + type_emb

        x = self.dropout(x)
        x = self.transformer(x)

        return self.head(x)
