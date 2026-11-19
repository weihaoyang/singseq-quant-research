"""
quant_sdk/preflight.py
--------------------------------------------------
因子 AST 安全校验与本地预审入口 (Factor Preflight)
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

from quant_sdk.local_research import preflight as local_preflight


def test_factor(factor_target: Callable | str | Any, data: str | Path | None = None) -> dict[str, Any]:
    """预检因子。

    ``data`` 必须是研究者本机的 CSV、Parquet、DuckDB 或 DB 面板。这个入口
    不会把源码或数据发送到平台；没有本地面板时明确失败，避免把已经退役的
    服务端研究接口伪装成可用能力。
    """
    if data is not None:
        result = local_preflight(data, factor_target)
        print(f"[Preflight] [{'PASS' if result['ok'] else 'FAIL'}] 本地真实面板检查: "
              f"{result['duration_ms']}ms | rows={result['sample_rows']}")
        return result

    raise ValueError(
        "test_factor 必须提供本机 --data 面板；服务端预检已退役。"
        " 请使用 quant-agent local.research.check 或 factor.lint。"
    )


def submit_factor(
    factor_target: Callable | str | Any,
    author: str = "researcher",
    start_date: str = "20230101",
    end_date: str = "20231231",
    mode: str = "sandbox",
    wait: bool = True
) -> dict[str, Any]:
    """Fail closed for the retired server-side submission API.

    Competition submission is an explicit client action bound to an immutable
    local source version and result artifact. It is intentionally not available
    through this legacy SDK function.
    """
    raise RuntimeError(
        "submit_factor 已停用：普通因子研究必须在本机执行。"
        " 已完成的本机结果请使用 competition.local.submit 在客户端显式登记。"
    )


def _extract_source(target: Callable | str | Any) -> str:
    if isinstance(target, str):
        p = Path(target)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8")
        return target
    if callable(target) or hasattr(target, "__class__"):
        try:
            return inspect.getsource(target)
        except Exception as exc:
            raise ValueError("无法自动提取函数或类源码，请直接传入源码字符串或文件路径") from exc
    raise TypeError("不支持的因子目标类型")
