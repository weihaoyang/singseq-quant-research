"""
deep_alpha/models/alphanet.py
--------------------------------------------------
AlphaNet v3 (时空量价卷积与记忆网络)
架构：
1. 1D 卷积层提取微观时序动量与价差模式
2. 双向 GRU / LSTM 捕捉宏观时序周期与长记忆
3. 注意力加权池化输出 Alpha 截面排序得分
"""

from __future__ import annotations
import torch
import torch.nn as nn


class AlphaNet(nn.Module):
    def __init__(
        self,
        feat_dim: int = 158,
        seq_len: int = 60,
        conv_channels: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len

        # 1D 卷积提取局部特征
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=feat_dim, out_channels=conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 双向 GRU
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 时间注意力权重
        self.att = nn.Linear(hidden_dim * 2, 1)

        # 线性输出层
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, seq_len, feat_dim]
        """
        # 转置为 [B, feat_dim, seq_len] 供 Conv1d 计算
        h = x.transpose(1, 2)
        h = self.conv(h) # [B, conv_channels, seq_len]

        # 转置回 [B, seq_len, conv_channels] 供 GRU 计算
        h = h.transpose(1, 2)
        out, _ = self.gru(h) # [B, seq_len, hidden_dim * 2]

        # 注意力池化
        weights = torch.softmax(self.att(out), dim=1) # [B, seq_len, 1]
        pooled = torch.sum(out * weights, dim=1)      # [B, hidden_dim * 2]

        score = self.head(pooled) # [B, 1]
        return score
