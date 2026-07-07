"""Vision Transformer classifier for value/mask time-series grids."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


@dataclass
class ViTConfig(ModelConfig):
    """Full model shape and architecture configuration."""

    num_variables: int = 1
    num_timesteps: int = 1
    num_classes: int = 2
    in_channels: int = 2


class TransformerEncoderLayerWithAttention(nn.Module):
    """Transformer encoder block that keeps attention weights for rollout."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.last_attn: torch.Tensor | None = None
        self.last_input: torch.Tensor | None = None
        self.last_output: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x
        if x.requires_grad:
            x.retain_grad()
        attn_input = self.norm1(x)
        attn_out, attn_weights = self.attn(
            attn_input,
            attn_input,
            attn_input,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attn = attn_weights
        if attn_weights.requires_grad:
            attn_weights.retain_grad()
        x = x + attn_out
        output = x + self.mlp(self.norm2(x))
        self.last_output = output
        return output


class ViTTimeSeriesClassifier(nn.Module):
    """Classify binned multivariable time series with a ViT-style encoder.

    Inputs must have shape `[batch, channels, variables, timesteps]`. The model
    pads the variable/time grid internally when the configured patch size does
    not divide the grid exactly, then crops explanation maps back to the
    original dimensions.
    """

    def __init__(self, config: ViTConfig | ModelConfig, **kwargs) -> None:
        """Build the patch embedder, transformer stack, and classification head."""
        super().__init__()
        if not isinstance(config, ViTConfig):
            config = ViTConfig(**{**config.__dict__, **kwargs})
        else:
            for key, value in kwargs.items():
                setattr(config, key, value)
        self.config = config
        patch_vars, patch_steps = config.patch_size
        self.patch_vars = patch_vars
        self.patch_steps = patch_steps
        self.patch_dim = config.in_channels * patch_vars * patch_steps
        self.padded_variables = ((config.num_variables + patch_vars - 1) // patch_vars) * patch_vars
        self.padded_timesteps = ((config.num_timesteps + patch_steps - 1) // patch_steps) * patch_steps
        self.num_patches = (self.padded_variables // patch_vars) * (self.padded_timesteps // patch_steps)
        self.patch_embed = nn.Linear(self.patch_dim, config.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, config.embed_dim))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayerWithAttention(
                    config.embed_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of binned tensors."""
        features = self.forward_features(x)
        return self.head(features)

    def forward_features(self, x: torch.Tensor, return_tokens: bool = False) -> torch.Tensor:
        """Return final transformer features before the classification head.

        By default this returns the final CLS-token embedding for each patient.
        Set `return_tokens=True` to return all normalized tokens.
        """
        patches = self.patchify(x)
        tokens = self.patch_embed(patches)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.dropout(tokens + self.pos_embed)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens if return_tokens else tokens[:, 0]

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten variable-time patches into transformer tokens."""
        bsz, channels, variables, timesteps = x.shape
        if channels != self.config.in_channels:
            raise ValueError(f"Expected {self.config.in_channels} channels, got {channels}.")
        pad_vars = self.padded_variables - variables
        pad_steps = self.padded_timesteps - timesteps
        if pad_vars or pad_steps:
            x = F.pad(x, (0, pad_steps, 0, pad_vars))
        x = x.unfold(2, self.patch_vars, self.patch_vars).unfold(3, self.patch_steps, self.patch_steps)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        return x.view(bsz, self.num_patches, self.patch_dim)

    def patch_scores_to_grid(self, patch_scores: torch.Tensor) -> torch.Tensor:
        """Expand one score per patch back to `[batch, variables, timesteps]`."""
        bsz = patch_scores.shape[0]
        vars_p = self.padded_variables // self.patch_vars
        steps_p = self.padded_timesteps // self.patch_steps
        grid = patch_scores.view(bsz, vars_p, steps_p)
        grid = grid.repeat_interleave(self.patch_vars, dim=1).repeat_interleave(self.patch_steps, dim=2)
        return grid[:, : self.config.num_variables, : self.config.num_timesteps]
