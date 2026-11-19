"""
quant_sdk/client.py
--------------------------------------------------
标准量化投研与数据分发 SDK 核心客户端 (Standard Quant SDK Client Engine)

设计特性：
1. 极简开箱即用：零配置自动嗅探本地 DuckDB 资产湖与远程 API 网关。
   - 本地模式 (Local Mode)：同一局域网或主机直接零拷贝毫秒级查询 185+ 数据表。
   - 远程模式 (Remote Mode)：无缝直连 HTTP API 网关，支持 Token 鉴权与配额流控。
2. 语法完全兼容 Tushare Pro：
   - pro = ts.pro_api()
   - df = pro.daily(ts_code="000001.SZ", start_date="20230101", end_date="20231231")
   - df = pro.query("daily_basic", trade_date="20231229")
3. 线程安全与自动重试保护。
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# 全局默认客户端实例
_DEFAULT_CLIENT: QuantClient | None = None


def init(token: str | None = None, url: str | None = None) -> QuantClient:
    """
    初始化全局 Quant SDK 配置
    :param token: 商业授权 Token 或大赛参赛者 Key
    :param url: 远程 API 网关地址 (默认 http://127.0.0.1:8000)
    :return: QuantClient 实例
    """
    global _DEFAULT_CLIENT
    _DEFAULT_CLIENT = QuantClient(base_url=url, token=token)
    return _DEFAULT_CLIENT


def pro_api(token: str | None = None, url: str | None = None) -> ProAPI:
    """
    获取兼容 Tushare Pro 的数据客户端对象 (Drop-in Replacement for ts.pro_api())
    """
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None or token is not None or url is not None:
        _DEFAULT_CLIENT = QuantClient(base_url=url, token=token)
    return _DEFAULT_CLIENT.pro_api()


class ProAPI:
    """Tushare Pro 兼容数据查询代理类"""

    def __init__(self, client: QuantClient):
        self._client = client

    def query(self, api_name: str, **kwargs) -> pd.DataFrame:
        """
        通用数据表查询：pro.query('daily', ts_code='000001.SZ', start_date='20230101')
        """
        return self._client.query(api_name, **kwargs)

    def __getattr__(self, name: str):
        """
        支持属性式动态调用：
        pro.daily(...)
        pro.daily_basic(...)
        pro.balancesheet(...)
        pro.fina_indicator(...)
        pro.moneyflow(...)
        pro.stock_basic(...)
        """
        if name.startswith("_"):
            raise AttributeError(name)
        def _caller(**kwargs):
            return self.query(name, **kwargs)
        return _caller


class QuantClient:
    """标准量化客户端核心类"""

    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base_url = (base_url or os.environ.get("QUANT_SERVER_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.token = token or os.environ.get("QUANT_API_TOKEN", "")
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

        self._local_client = None
        self._mode = "remote"

        # 智能嗅探本地 DuckDB 是否存在 (如果存在且未显式指定外部远程地址，启用本地高速引擎)
        if base_url is None and os.environ.get("QUANT_FORCE_REMOTE", "0") != "1":
            possible_paths = [
                Path("tushare_data.duckdb"),
                Path(__file__).resolve().parent.parent / "tushare_data.duckdb",
                Path(os.environ.get("QUANT_DATA_PATH", "")) / "tushare_data.duckdb"
            ]
            for p in possible_paths:
                if p.exists() and p.is_file():
                    try:
                        from data_foundation.access import UniversalDataClient
                        self._local_client = UniversalDataClient(str(p))
                        self._mode = "local"
                        break
                    except Exception:
                        pass

        self._pro = ProAPI(self)

    @property
    def mode(self) -> str:
        """当前数据访问引擎模式: 'local' (零拷贝极速) 或 'remote' (HTTP网关)"""
        return self._mode

    def pro_api(self) -> ProAPI:
        return self._pro

    def query(self, api_name: str, **kwargs) -> pd.DataFrame:
        """核心查询路由：本地直连或远程网关转发"""
        if self._mode == "local" and self._local_client is not None:
            try:
                return self._local_client.query(api_name, **kwargs)
            except Exception as exc:
                # A silent remote fallback changes the data snapshot while the
                # client still reports ``mode='local'``. Fail closed so callers
                # can retry, repair, or explicitly construct a remote client.
                raise RuntimeError(
                    f"本地数据查询失败（{api_name}）：拒绝静默切换远程数据源"
                ) from exc

        url = f"{self.base_url}/api/v1/data/query"
        limit = kwargs.pop("limit", 1000)
        payload = {
            "api": api_name,
            "params": kwargs,
            "limit": limit
        }
        res = self.session.post(url, json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()
        if data.get("ok") is True:
            payload = data.get("data") or {}
            records = payload.get("rows", [])
            cols = payload.get("columns", [])
            return pd.DataFrame(records, columns=cols)
        if data.get("code") == 0:
            records = data.get("data", [])
            cols = data.get("columns", [])
            return pd.DataFrame(records, columns=cols)
        error = data.get("error") or data.get("message") or data
        raise RuntimeError(f"API [{api_name}] 查询失败: {error}")

    def list_apis(self) -> list[str]:
        """获取所有可用数据 API 列表 (185+ 张表)"""
        if self._mode == "local" and self._local_client is not None:
            return self._local_client.list_apis()
        url = f"{self.base_url}/api/v1/data/apis"
        res = self.session.get(url, timeout=10)
        res.raise_for_status()
        body = res.json()
        if body.get("ok") is True:
            return (body.get("data") or {}).get("apis", [])
        return body.get("apis", [])

    def check_health(self) -> dict[str, Any]:
        """探测服务端硬件拓扑与算力状态 (20核CPU + RTX 5070)"""
        url = f"{self.base_url}/api/v1/health"
        res = self.session.get(url, timeout=5)
        res.raise_for_status()
        body = res.json()
        return body.get("data", body) if body.get("ok") is True else body

    def get_quota(self) -> dict[str, Any]:
        """查询当前账号商业配额水表与 QPS 并发度"""
        url = f"{self.base_url}/api/v1/auth/me"
        res = self.session.get(url, timeout=5)
        res.raise_for_status()
        body = res.json()
        payload = body.get("data", body) if body.get("ok") is True else body
        return payload.get("tenant_info", {})

    def upload_result(self, factor_id: str, version_id: str,
                      result_file: str | Path, *,
                      package_version: str | None = None,
                      package_hash: str | None = None) -> dict[str, Any]:
        """Upload a local UTF-8 factor-result CSV bound to one immutable version.

        The server remains the authority for schema, duplicate-key and hash
        validation. This method only reads the selected local file and sends
        its bytes; it never uploads market data.
        """
        path = Path(result_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() != ".csv":
            raise ValueError("result_file must be a UTF-8 .csv file")
        content = path.read_bytes()
        if not content or len(content) > 50 * 1024 * 1024:
            raise ValueError("result_file must be between 1 byte and 50MB")
        payload: dict[str, Any] = {
            "filename": path.name,
            "content_b64": base64.b64encode(content).decode("ascii"),
        }
        if package_version:
            payload["package_version"] = package_version
        if package_hash:
            payload["package_hash"] = package_hash
        url = (f"{self.base_url}/api/v1/factors/{factor_id}/versions/"
               f"{version_id}/results")
        res = self.session.post(url, json=payload, timeout=60)
        res.raise_for_status()
        body = res.json()
        if body.get("ok") is True:
            return body.get("data") or {}
        if body.get("code") == 0:
            return body.get("data") or body
        raise RuntimeError(f"结果上传失败: {body.get('error') or body.get('message') or body}")
