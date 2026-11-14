"""
deep_alpha/models/itransformer.py
--------------------------------------------------
iTransformer (Inverted Transformer for Factor Cross-Attention)
核心思想：
反转 Conventional Transformer 的 Token 概念：
- 不再把“时间步”当 Token，而是把“每个 Alpha 因子特征”当作一个独立的 Token
- 每个因子在过去 60 步的整段走势被编码为 Token 的 Embedding
- 利用 Self-Attention 直接建模 158 维因子之间的跨特征非线性相关性与交互作用
"""

from __future__ import annotations
import torch
import torch.nn as nn


class iTransformer(nn.Module):
    def __init__(
        self,
        feat_dim: int = 158,
        seq_len: int = 60,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len

        # 时间序列反转投影: 将整个时间序列映射为因子的特征表示
        self.temporal_proj = nn.Linear(seq_len, d_model)

        # 因子跨特征注意力 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.cross_factor_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 最终截面预测头
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * feat_dim),
            nn.Linear(d_model * feat_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x [Batch, seq_len, feat_dim]
        输出: score [Batch, 1]
        """
        # 转置维度: [B, feat_dim, seq_len]
        # 现在每个因子是一个 Token，序列长度为 seq_len
        x_inv = x.transpose(1, 2)

        # 将整段时间走势投影到隐空间: [B, feat_dim, d_model]
        tokens = self.temporal_proj(x_inv)

        # 计算因子与因子之间的自注意力关联: [B, feat_dim, d_model]
        factor_repr = self.cross_factor_encoder(tokens)

        # 展平所有因子的交互表征并输出预测
        B = x.shape[0]
        score = self.head(factor_repr.reshape(B, -1)) # [B, 1]
        return score
