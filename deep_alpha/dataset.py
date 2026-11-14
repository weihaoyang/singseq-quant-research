"""
deep_alpha/dataset.py
--------------------------------------------------
3D 时序张量流式数据管道 (Memory-Mapped Quant Tensor Dataset)
功能：
1. 构建三维金融时序张量: [Batch/Samples, Seq_Len=60, Feat_Dim=158]
2. 支持磁盘内存映射；构建时仍需容纳输入矩阵，不承诺固定内存上限
3. 适配 PyTorch DataLoader 与多进程并行预取
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from pathlib import Path


class QuantTensorDataset:
    """流式多标的时序 3D 张量生成器与缓存器"""

    def __init__(
        self,
        features_array: np.ndarray,      # 形状: [Total_Records, Feature_Dim]
        labels_array: np.ndarray,        # 形状: [Total_Records] (次日收益率)
        indices_map: List[Tuple[int, int]], # 有效切片索引: [(start_idx, end_idx), ...]
        seq_len: int = 60,
        sample_metadata: Optional[pd.DataFrame] = None
    ):
        self.features = features_array
        self.labels = labels_array
        self.indices_map = indices_map
        self.seq_len = seq_len
        self.sample_metadata = sample_metadata

    def __len__(self) -> int:
        return len(self.indices_map)

    def __getitem__(self, idx: int):
        import torch
        start_idx, end_idx = self.indices_map[idx]
        x_window = self.features[start_idx:end_idx] # [seq_len, feat_dim]
        y_val = self.labels[end_idx - 1]           # target_ret
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)


def build_streaming_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_ret",
    seq_len: int = 60,
    mmap_path: Optional[str] = None,
    require_label: bool = True
) -> QuantTensorDataset:
    """根据面板数据构建流式 3D 张量数据集"""
    df = df.sort_values(by=["ts_code", "trade_date"]).reset_index(drop=True)

    if seq_len < 1 or df.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("invalid sequence length or duplicate panel keys")
    # Missing labels are unknown, never zero. Preprocessing belongs to the
    # train-only fitted experiment processor, not the dataset constructor.
    feat_matrix = df[feature_cols].values.astype(np.float32)
    labels = df[target_col].values.astype(np.float32)

    # 如果指定 mmap_path，使用磁盘内存映射
    if mmap_path:
        Path(mmap_path).parent.mkdir(parents=True, exist_ok=True)
        mmap_feat = np.memmap(mmap_path, dtype="float32", mode="w+", shape=feat_matrix.shape)
        mmap_feat[:] = feat_matrix[:]
        mmap_feat.flush()
        feat_matrix = mmap_feat

    # 构建每个股票内部的合法切片索引 (防止跨标的切片)
    indices_map: List[Tuple[int, int]] = []

    # 获取每个 ts_code 的起止下标
    code_changes = np.where(df["ts_code"].values[:-1] != df["ts_code"].values[1:])[0] + 1
    split_indices = [0] + code_changes.tolist() + [len(df)]

    for i in range(len(split_indices) - 1):
        c_start = split_indices[i]
        c_end = split_indices[i + 1]
        c_len = c_end - c_start
        if c_len >= seq_len:
            for s in range(c_start, c_end - seq_len + 1):
                e = s + seq_len
                continuous = ("session_id" not in df or
                              df["session_id"].iloc[e - 1] - df["session_id"].iloc[s] == seq_len - 1)
                if continuous and (not require_label or np.isfinite(labels[e - 1])) and np.isfinite(feat_matrix[s:e]).all():
                    indices_map.append((s, e))

    columns = [c for c in ["ts_code", "trade_date", "label_end_date"] if c in df]
    metadata = df.iloc[[e - 1 for _, e in indices_map]][columns].reset_index(drop=True)
    return QuantTensorDataset(feat_matrix, labels, indices_map, seq_len=seq_len, sample_metadata=metadata)
