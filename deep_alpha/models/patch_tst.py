"""
deep_alpha/models/patch_tst.py
--------------------------------------------------
PatchTST (Patch Time Series Transformer for Quantitative Alpha)
特性：
1. 子序列分块 (Patching): 将 60 步长的时序切为小 Patch (如 12 步)，保留局部微观结构
2. 将 Attention 复杂度从 O(T^2) 降至 O((T/P)^2)，极致压低 RTX 5070 显存占用
3. 学习跨时间 Patch 之间的关联与非线性动量
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(
        self,
        feat_dim: int = 158,
        seq_len: int = 60,
        patch_len: int = 12,
        stride: int = 12,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride

        # 计算 Patch 数量
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Patch 线性投影
        self.patch_proj = nn.Linear(patch_len * feat_dim, d_model)

        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 头部回归打分器
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * self.num_patches),
            nn.Linear(d_model * self.num_patches, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x [Batch, seq_len, feat_dim]
        输出: score [Batch, 1]
        """
        B, T, D = x.shape
        # 展平切片为 Patches
        # x_unfold: [B, num_patches, patch_len, D]
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patches.append(x[:, start:end, :].reshape(B, 1, -1))

        x_patches = torch.cat(patches, dim=1) # [B, num_patches, patch_len * D]

        # 投影到隐空间并加位置编码
        h = self.patch_proj(x_patches) + self.pos_embed # [B, num_patches, d_model]

        # Transformer 自注意力
        out = self.encoder(h) # [B, num_patches, d_model]

        # 展平并打分
        flat = out.reshape(B, -1)
        score = self.head(flat) # [B, 1]
        return score
