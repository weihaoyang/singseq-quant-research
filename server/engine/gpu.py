"""GPUTensorEngine - real torch/CUDA acceleration with OOM fallback (H8).

Scope (what actually runs on the GPU):

* ``batched_spearman_rank_ic``  - per-row rank (argsort twice) + Pearson
  IC of factor rank vs forward-return rank for every date at once.
* ``batched_ic_decay``          - same kernel over lag-shifted returns.
* ``batched_residualize``       - cross-sectional residualisation of a
  factor matrix vs a baseline factor pool via one batched least-squares
  (``torch.linalg.lstsq``) per date.

Fallback policy (H8 + ARCHITECTURE 5):

* ``backend="cpu"`` or no CUDA            -> ``"cpu"``
* every GPU op succeeded                  -> ``"cuda"``
* memory *probe* declined (free VRAM <
  needed bytes) or a real ``torch.OutOfMemoryError``
  was raised -> log a warning, drop to numpy kernel,
  report ``"cpu-fallback"``.

Purity: torch + numpy + stdlib only.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger("engine.gpu")

try:  # torch is optional at import time (engine/__init__ stays lazy)
    import torch
    _TORCH_OK = True
except Exception:  # pragma: no cover - environment without torch
    torch = None  # type: ignore[assignment]
    _TORCH_OK = False

FALLBACK_CPU = "cpu"
FALLBACK_CUDA = "cuda"
FALLBACK_OOM = "cpu-fallback"


class GPUBackendUnavailable(RuntimeError):
    """Raised when a CUDA backend is requested but unavailable."""


def cuda_available() -> bool:
    return bool(_TORCH_OK and torch.cuda.is_available())


def _dtype():
    return torch.float32


class GPUTensorEngine:
    """Thin wrapper owning device placement + the guarded fallback logic."""

    def __init__(self, device: Optional[str] = None, probe: bool = True):
        if not _TORCH_OK:
            raise GPUBackendUnavailable("torch is not installed")
        if not torch.cuda.is_available():
            raise GPUBackendUnavailable("CUDA is not available")
        self.device = torch.device(device or "cuda:0")
        self.probe = probe
        self._mode = FALLBACK_CUDA
        self._last_free: Optional[int] = None
        self._warm = False
        self.uploads: List[str] = []

    # -- housekeeping ------------------------------------------------------- #
    @property
    def mode(self) -> str:
        return self._mode

    def _warn_fallback(self, reason: str) -> None:
        if self._mode != FALLBACK_OOM:
            logger.warning("GPUTensorEngine falling back to CPU: %s", reason)
        self._mode = FALLBACK_OOM
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    def _probe_declines(self, nbytes: int) -> bool:
        """True when free VRAM cannot hold ``nbytes`` (+5% headroom)."""
        if not self.probe:
            return False
        try:
            free, total = torch.cuda.mem_get_info(self.device)
        except Exception:  # pragma: no cover
            return False
        self._last_free = int(free)
        return int(free) < int(nbytes * 1.05)

    def _guarded(self, name: str, nbytes: int,
                 cpu_fn: Callable[[], np.ndarray],
                 gpu_fn: Callable[[], "torch.Tensor"]) -> np.ndarray:
        """Run ``gpu_fn``; on probe-decline or OOM run ``cpu_fn`` instead."""
        if self._mode == FALLBACK_OOM:
            # already degraded for this engine instance
            return cpu_fn()
        if self._probe_declines(nbytes):
            self._warn_fallback(
                f"{name} probe declined (need {nbytes/1e6:.1f} MB, "
                f"free {0 if self._last_free is None else self._last_free/1e6:.1f} MB)")
            return cpu_fn()
        try:
            out = gpu_fn()
            self.uploads.append(name)
            return out
        except torch.OutOfMemoryError as exc:
            self._warn_fallback(f"{name} raised OutOfMemoryError: {exc}")
            return cpu_fn()

    def reset(self) -> None:
        """New backtest run -> allow CUDA again (probe re-checks each op)."""
        self._mode = FALLBACK_CUDA
        self.uploads.clear()
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    # -- kernels ------------------------------------------------------------ #
    def rank_matrix(self, x: np.ndarray) -> np.ndarray:
        """[T, N] -> row-wise average-tie ranks (NaN stays NaN), computed on GPU.

        Kept in float64 so tie groups and averaged ranks are bit-identical
        with ``engine.metrics.rank_rows`` (quintile membership must not
        depend on the backend).  Falls back to the numpy kernel through
        the same probe/OOM guard as every other op.
        """
        x = np.asarray(x, dtype=np.float64)
        T, N = x.shape
        nbytes = x.nbytes + T * N * 8 * 4

        def cpu() -> np.ndarray:
            from . import metrics as M
            return M.rank_rows(x)

        def gpu() -> np.ndarray:
            t0 = time.perf_counter()
            t = torch.as_tensor(np.ascontiguousarray(x),
                                device=self.device, dtype=torch.float64)
            finite = torch.isfinite(t)
            # NaN sentinel sorts LAST (+inf): valid entries then keep their
            # true 1..n_valid positions.  A -inf sentinel shifted every rank
            # by the row's NaN count, which Pearson on ranks tolerated but
            # rank_normalize()/quintile binning did NOT (backend-dependent
            # group membership).  Fixed 2026-09-26 (S3).
            safe = torch.where(finite, t, torch.full_like(t, float("inf")))
            ranks = self._rank2d(safe)
            ranks = torch.where(finite, ranks,
                                torch.full_like(ranks, float("nan")))
            out = ranks.cpu().numpy()
            self.uploads.append(f"rank_matrix[{time.perf_counter()-t0:.4f}s]")
            return out

        return self._guarded("rank_matrix", nbytes, cpu, gpu)

    @staticmethod
    def _rank2d(x: "torch.Tensor") -> "torch.Tensor":
        """Row-wise average-tie ranks of a [B, N] tensor."""
        n = x.shape[-1]
        order = torch.argsort(x, dim=-1)
        sorted_x = torch.gather(x, -1, order)
        # group starts in sorted order
        starts = torch.ones_like(sorted_x, dtype=torch.bool)
        if n > 1:
            starts[:, 1:] = sorted_x[:, 1:] != sorted_x[:, :-1]
        pos = torch.arange(n, device=x.device, dtype=x.dtype).expand_as(sorted_x)
        # first index of the tie group: largest start <= i
        first = torch.cummax(torch.where(starts, pos, torch.zeros_like(pos)),
                             dim=-1).values
        # last index of the tie group: first group-end >= i
        ends = torch.ones_like(sorted_x, dtype=torch.bool)
        if n > 1:
            ends[:, :-1] = starts[:, 1:]
        marked = torch.where(ends, pos, torch.full_like(pos, float("inf")))
        last = torch.flip(torch.cummin(torch.flip(marked, [-1]), dim=-1).values, [-1])
        avg_sorted = (first + last) / 2.0 + 1.0
        ranks = torch.empty_like(sorted_x)
        ranks.scatter_(-1, order, avg_sorted)
        return ranks

    @staticmethod
    def _pearson(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
        """Row-wise Pearson of two [B, N] tensors, NaN-safe.

        Invalid pairs are zero-filled *after* the per-row means are taken
        over the same valid subset - a naive ``mean()`` over NaN poisons
        the whole row (that bug produced all-NaN IC series).
        """
        both = torch.isfinite(a) & torch.isfinite(b)
        cnt = both.sum(dim=-1)
        a0 = torch.where(both, a, torch.zeros_like(a))
        b0 = torch.where(both, b, torch.zeros_like(b))
        c = torch.clamp(cnt.to(a.dtype), min=1.0)
        sa = a0.sum(dim=-1)
        sb = b0.sum(dim=-1)
        saa = (a0 * a0).sum(dim=-1)
        sbb = (b0 * b0).sum(dim=-1)
        sab = (a0 * b0).sum(dim=-1)
        va = saa - sa * sa / c
        vb = sbb - sb * sb / c
        denom = torch.sqrt(va * vb)
        r = (sab - sa * sb / c) / torch.where(
            denom > 0, denom, torch.full_like(denom, float("nan")))
        r = torch.where((cnt >= 3) & torch.isfinite(r), r,
                        torch.full_like(r, float("nan")))
        return r

    def batched_spearman_rank_ic(self, factor: np.ndarray, forward: np.ndarray,
                                 workers: int = 0) -> np.ndarray:
        """[T, N] factor & forward -> [T] Spearman IC per date (GPU)."""
        factor = np.asarray(factor, dtype=np.float32)
        forward = np.asarray(forward, dtype=np.float32)
        T, N = factor.shape
        nbytes = factor.nbytes + forward.nbytes + 2 * T * N * 4

        def cpu() -> np.ndarray:
            from . import metrics as M
            return M.spearman_ic_rows(factor.astype(np.float64),
                                      forward.astype(np.float64), workers=20)

        def gpu() -> np.ndarray:
            t0 = time.perf_counter()
            f = torch.as_tensor(np.ascontiguousarray(factor),
                                device=self.device, dtype=_dtype())
            r = torch.as_tensor(np.ascontiguousarray(forward),
                                device=self.device, dtype=_dtype())
            # NaN -> +inf so they sort LAST (true 1..n_valid positions),
            # then mask them out of the IC
            f_mask = torch.isfinite(f)
            r_mask = torch.isfinite(r)
            f = torch.where(f_mask, f, torch.full_like(f, float("inf")))
            r = torch.where(r_mask, r, torch.full_like(r, float("inf")))
            fr = self._rank2d(f)
            rr = self._rank2d(r)
            both = f_mask & r_mask
            # NaNs get rank replaced by NaN before Pearson
            fr = torch.where(both, fr, torch.full_like(fr, float("nan")))
            rr = torch.where(both, rr, torch.full_like(rr, float("nan")))
            # rows with <3 valid points -> NaN
            cnt = both.sum(dim=-1)
            ic = self._pearson(fr, rr)
            ic = torch.where(cnt >= 3, ic, torch.full_like(ic, float("nan")))
            out = ic.cpu().numpy().astype(np.float64)
            self.uploads.append(f"rank_ic[{time.perf_counter()-t0:.4f}s]")
            return out

        return self._guarded("rank_ic", nbytes, cpu, gpu)

    def batched_ic_decay(self, factor: np.ndarray, prices: np.ndarray,
                         lags: Sequence[int] = (1, 2, 3, 5, 10, 20),
                         workers: int = 0) -> Dict[int, float]:
        """IC of factor at t vs return over [t, t+lag], common row mask."""
        factor = np.asarray(factor, dtype=np.float32)
        prices = np.asarray(prices, dtype=np.float32)
        T, N = factor.shape
        max_lag = int(max(lags))
        nbytes = factor.nbytes + prices.nbytes

        def cpu() -> Dict[int, float]:
            from . import metrics as M
            return M.ic_decay_map(factor.astype(np.float64),
                                  prices.astype(np.float64), lags=lags)

        def gpu() -> Dict[int, float]:
            out: Dict[int, float] = {}
            f = torch.as_tensor(np.ascontiguousarray(factor),
                                device=self.device, dtype=_dtype())
            p = torch.as_tensor(np.ascontiguousarray(prices),
                                device=self.device, dtype=_dtype())
            n_rows = max(T - max_lag, 0)
            if n_rows <= 0:
                return {int(l): float("nan") for l in lags}
            f_part = f[:n_rows]
            # lag returns + common mask (same discipline as the CPU kernel)
            fwd_by_lag: Dict[int, "torch.Tensor"] = {}
            common = torch.ones_like(f_part, dtype=torch.bool)
            for lag in sorted({int(l) for l in lags}):
                if lag <= 0 or lag >= T:
                    continue
                fwd = p[lag:lag + n_rows] / p[:n_rows] - 1.0
                fwd_by_lag[lag] = fwd
                common &= torch.isfinite(fwd)
            fr = self._rank2d(f_part)
            common &= torch.isfinite(fr)
            a = torch.where(common, fr, torch.full_like(fr, float("nan")))
            for lag in sorted({int(l) for l in lags}):
                if lag <= 0 or lag >= T or lag not in fwd_by_lag:
                    out[lag] = float("nan")
                    continue
                fwr = self._rank2d(fwd_by_lag[lag])
                b = torch.where(common, fwr, torch.full_like(fwr, float("nan")))
                ic = self._pearson(a, b)
                finite = ic[torch.isfinite(ic)]
                out[lag] = float(finite.mean().cpu()) if finite.numel() else float("nan")
            self.uploads.append("ic_decay")
            return out

        return self._guarded("ic_decay", nbytes, cpu, gpu)

    def batched_residualize(self, factor: np.ndarray, baselines: np.ndarray,
                            workers: int = 0) -> np.ndarray:
        """Cross-sectional residual of ``factor`` vs ``baselines`` per date.

        Uses one batched least-squares on the [T, N, K] design; when CUDA
        cannot allocate, falls back to a numpy lstsq (still real maths,
        no synthetic values).
        """
        factor = np.asarray(factor, dtype=np.float32)
        baselines = np.asarray(baselines, dtype=np.float32)
        if baselines.ndim != 3:
            raise ValueError("baselines must be [T, N, K]")
        T, N = factor.shape
        K = baselines.shape[-1]
        valid_mask = np.isfinite(factor) & np.isfinite(baselines).all(axis=-1)
        valid_counts = valid_mask.sum(axis=1)
        if np.any(valid_counts <= K):
            bad = ",".join(str(i) for i in np.flatnonzero(valid_counts <= K)[:8])
            raise ValueError(
                "insufficient finite observations for residualization on day(s): "
                f"{bad}")
        nbytes = factor.nbytes + baselines.nbytes + T * N * 4

        def cpu() -> np.ndarray:
            X = baselines.astype(np.float64)
            y = factor.astype(np.float64)
            out = np.empty_like(y)
            for t in range(T):
                Xt = X[t]
                yt = y[t]
                m = np.isfinite(yt) & np.isfinite(Xt).all(axis=1)
                coef, *_ = np.linalg.lstsq(Xt[m], yt[m], rcond=None)
                res = np.full(N, np.nan)
                res[m] = yt[m] - Xt[m] @ coef
                out[t] = res
            return out

        def gpu() -> np.ndarray:
            A = torch.as_tensor(np.ascontiguousarray(baselines),
                                device=self.device, dtype=_dtype())
            b = torch.as_tensor(np.ascontiguousarray(factor),
                                device=self.device, dtype=_dtype())
            invalid = ~torch.isfinite(b) | ~torch.isfinite(A).all(dim=-1)
            A = torch.where(invalid[..., None], torch.zeros_like(A), A)
            b = torch.where(invalid, torch.zeros_like(b), b)
            sol = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
            resid = b - (A @ sol.unsqueeze(-1)).squeeze(-1)
            resid = torch.where(invalid, torch.full_like(resid, float("nan")), resid)
            out = resid.cpu().numpy().astype(np.float64)
            self.uploads.append("residualize")
            return out

        return self._guarded("residualize", nbytes, cpu, gpu)

    # -- introspection ------------------------------------------------------ #
    def device_info(self) -> Dict[str, object]:
        idx = self.device.index or 0
        props = torch.cuda.get_device_properties(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        return {
            "name": torch.cuda.get_device_name(idx),
            "capability": f"{props.major}.{props.minor}",
            "total_memory_mb": round(total / 1e6, 1),
            "free_memory_mb": round(free / 1e6, 1),
            "mode": self._mode,
        }

    def peak_memory_mb(self) -> float:
        try:
            return round(torch.cuda.max_memory_allocated(self.device) / 1e6, 2)
        except Exception:  # pragma: no cover
            return 0.0

    def reset_peak_memory(self) -> None:
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except Exception:  # pragma: no cover
            pass
