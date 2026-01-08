from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SlideViTConfig:
    input_dim: int = 768
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    mlp_act: str = "swiglu"
    norm_type: str = "rmsnorm"
    drop_path_rate: float = 0.1
    layer_scale_init: float = 1e-5
    max_grid_size: Tuple[int, int] = (64, 64)
    pos_embed_type: str = "sin2d"
    num_obj_ids: int = 0
    attn_drop: float = 0.0
    proj_drop: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class LearnedPositionEmbedding2D(nn.Module):
    def __init__(self, dim: int, max_h: int, max_w: int) -> None:
        super().__init__()
        self.row_embed = nn.Embedding(max_h, dim)
        self.col_embed = nn.Embedding(max_w, dim)
        nn.init.normal_(self.row_embed.weight, std=0.02)
        nn.init.normal_(self.col_embed.weight, std=0.02)

    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        row = self.row_embed(torch.arange(h, device=device))
        col = self.col_embed(torch.arange(w, device=device))
        pos = row[:, None, :] + col[None, :, :]
        return pos.reshape(h * w, -1)


def sincos_position_embedding_2d(
    h: int,
    w: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for sincos 2D embedding.")
    dim_quarter = dim // 4
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_quarter, device=device, dtype=dtype) / dim_quarter))
    pos_x = torch.arange(w, device=device, dtype=dtype)
    pos_y = torch.arange(h, device=device, dtype=dtype)
    freqs_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    freqs_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    emb_x = torch.cat([torch.sin(freqs_x), torch.cos(freqs_x)], dim=1)
    emb_y = torch.cat([torch.sin(freqs_y), torch.cos(freqs_y)], dim=1)
    emb = torch.cat(
        [
            emb_y[:, None, :].expand(h, w, -1),
            emb_x[None, :, :].expand(h, w, -1),
        ],
        dim=-1,
    )
    return emb.reshape(h * w, dim)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(x2) * x1)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, act: str = "swiglu") -> None:
        super().__init__()
        act = act.lower()
        if act == "swiglu":
            self.block = SwiGLU(dim, hidden_dim)
        else:
            self.block = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float, proj_drop: float) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        if dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias.unsqueeze(0)
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(torch.bool)
            attn = attn.masked_fill(mask, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return self.proj_drop(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop_path: float,
        layer_scale_init: float,
        norm_layer: nn.Module,
        mlp_act: str,
        attn_drop: float,
        proj_drop: float,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiheadSelfAttention(dim, num_heads, attn_drop, proj_drop)
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim, act=mlp_act)
        self.drop_path = DropPath(drop_path)
        if layer_scale_init > 0:
            self.ls1 = LayerScale(dim, layer_scale_init)
            self.ls2 = LayerScale(dim, layer_scale_init)
        else:
            self.ls1 = nn.Identity()
            self.ls2 = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), attn_bias, key_padding_mask)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class SlideViT(nn.Module):
    def __init__(self, config: SlideViTConfig) -> None:
        super().__init__()
        self.config = config
        self.input_dim = int(config.input_dim)
        self.embed_dim = int(config.embed_dim)
        self.num_heads = int(config.num_heads)
        self.pos_embed_type = str(config.pos_embed_type).lower()
        self.token_proj = (
            nn.Identity() if self.input_dim == self.embed_dim else nn.Linear(self.input_dim, self.embed_dim)
        )
        self.pad_token = nn.Parameter(torch.zeros(self.embed_dim))
        if self.pos_embed_type == "learned":
            self.pos_embed = LearnedPositionEmbedding2D(
                self.embed_dim,
                int(config.max_grid_size[0]),
                int(config.max_grid_size[1]),
            )
        else:
            self.pos_embed = None
        self.obj_embed = (
            nn.Embedding(int(config.num_obj_ids), self.embed_dim) if config.num_obj_ids > 0 else None
        )
        self.mix_alpha = nn.Parameter(torch.ones(self.embed_dim))
        self.mix_beta = nn.Parameter(torch.ones(self.embed_dim))
        self.mix_gamma = nn.Parameter(torch.ones(self.embed_dim))
        self.alibi_x = nn.Parameter(torch.ones(self.num_heads))
        self.alibi_y = nn.Parameter(torch.ones(self.num_heads))

        norm_layer = RMSNorm if config.norm_type.lower() == "rmsnorm" else nn.LayerNorm
        dpr = np.linspace(0.0, float(config.drop_path_rate), int(config.depth)).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=float(config.mlp_ratio),
                    drop_path=float(dpr[i]),
                    layer_scale_init=float(config.layer_scale_init),
                    norm_layer=norm_layer,
                    mlp_act=config.mlp_act,
                    attn_drop=float(config.attn_drop),
                    proj_drop=float(config.proj_drop),
                )
                for i in range(int(config.depth))
            ]
        )
        self.norm = norm_layer(self.embed_dim)
        self._alibi_cache = {}

    def _prepare_inputs(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        obj_ids: Optional[torch.Tensor],
        grid_size: Optional[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        if x.ndim == 4:
            b, h, w, c = x.shape
        elif x.ndim == 3:
            if grid_size is None:
                raise ValueError("grid_size is required when providing a flattened token sequence.")
            b, n, c = x.shape
            h, w = int(grid_size[0]), int(grid_size[1])
            if h * w != n:
                raise ValueError("grid_size does not match flattened token count.")
            x = x.view(b, h, w, c)
        else:
            raise ValueError("Expected input tokens with shape [B,H,W,C] or [B,N,C].")

        if valid_mask is None:
            valid_mask = torch.ones((b, h, w), dtype=torch.bool, device=x.device)
        else:
            if valid_mask.ndim == 2:
                valid_mask = valid_mask.view(b, h, w)
            valid_mask = valid_mask.to(torch.bool)

        token = self.token_proj(x)
        if (~valid_mask).any():
            token = token.clone()
            token[~valid_mask] = self.pad_token

        token = token.view(b, h * w, -1)
        pos = None
        if self.pos_embed_type == "learned":
            if h > int(self.config.max_grid_size[0]) or w > int(self.config.max_grid_size[1]):
                raise ValueError("Grid exceeds max_grid_size for learned position embeddings.")
            pos = self.pos_embed(h, w, x.device).to(token.dtype)
        elif self.pos_embed_type == "sin2d":
            pos = sincos_position_embedding_2d(h, w, self.embed_dim, token.device, token.dtype)
        if pos is not None:
            pos = pos.unsqueeze(0).expand(b, -1, -1)
        if obj_ids is not None and self.obj_embed is not None:
            if obj_ids.ndim == 2:
                obj_ids = obj_ids.view(b, h, w)
            obj = self.obj_embed(obj_ids.to(torch.long)).view(b, h * w, -1)
        else:
            obj = None

        alpha = self.mix_alpha.view(1, 1, -1)
        if pos is not None:
            beta = self.mix_beta.view(1, 1, -1)
            h0 = alpha * token + beta * pos
        else:
            h0 = alpha * token
        if obj is not None:
            gamma = self.mix_gamma.view(1, 1, -1)
            h0 = h0 + gamma * obj

        pad_mask = (~valid_mask).view(b, h * w)
        return h0, valid_mask, pad_mask, h, w

    def _alibi_bias(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w, device, dtype)
        cached = self._alibi_cache.get(key)
        if cached is not None:
            return cached
        coords = torch.stack(
            torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"),
            dim=-1,
        ).view(-1, 2)
        dx = coords[:, 0][:, None] - coords[:, 0][None, :]
        dy = coords[:, 1][:, None] - coords[:, 1][None, :]
        dx = dx.abs().to(dtype)
        dy = dy.abs().to(dtype)
        ax = F.softplus(self.alibi_x).view(-1, 1, 1).to(dtype)
        ay = F.softplus(self.alibi_y).view(-1, 1, 1).to(dtype)
        bias = -(ax * dx + ay * dy)
        self._alibi_cache[key] = bias
        return bias

    def forward_features(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        obj_ids: Optional[torch.Tensor] = None,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, valid_mask, pad_mask, h, w = self._prepare_inputs(x, valid_mask, obj_ids, grid_size)
        attn_bias = self._alibi_bias(h, w, tokens.device, tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, attn_bias, pad_mask)
        tokens = self.norm(tokens)
        pooled = self._pool(tokens, valid_mask)
        return tokens, pooled

    def _pool(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = tokens.shape
        mask = valid_mask.view(b, n).to(tokens.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (tokens * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        obj_ids: Optional[torch.Tensor] = None,
        grid_size: Optional[Tuple[int, int]] = None,
        return_tokens: bool = True,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        tokens, pooled = self.forward_features(x, valid_mask, obj_ids, grid_size)
        if return_tokens:
            return tokens, pooled
        return pooled
