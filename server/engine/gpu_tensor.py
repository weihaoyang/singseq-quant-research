"""gpu_tensor - RTX 5070 batched cross-sectional kernels (S3, deliverable 2).

All kernels share the same discipline:

* input: pandas / numpy (CPU); a **single** ``.to("cuda")`` transfer per
  input tensor, then every op stays on the device until one ``.cpu()``;
* ``torch.no_grad()`` + ``float32`` on the device (float64 only in the
  numpy fallback path when exact integer ranks are required);
* output: plain numpy (CPU) wrapped in a result object that always
  carries ``backend: "cuda" | "cpu"`` and ``fell_back_to_cpu: bool`` so
  callers know exactly which path ran (ARCHITECTURE 5.2);
* if CUDA is unavailable the **CPU fallback computes the same numerics**
  (slower, no GPU) - it is not a stub and never returns synthetic data (H7).

Kernels
-------
``batched_ols_residuals(factor, exposures, groups=None)``
    Per-day cross-sectional OLS of ``factor`` on ``exposures`` via one
    batched ``torch.linalg.lstsq`` over ``[Days, Stocks, Features]``.
    ``groups`` (optional integer labels ``[N]`` / ``[T, N]``) applies
    within-group demeaning to *both* y and X first (Frisch-Waugh-Lovell
    => residual is orthogonal to the group dummies as well), which is how
    industry neutralisation is done without materialising N x G dummies.

``gram_schmidt_orthogonal(X, risk_factors)``
    Classical Gram-Schmidt: the running orthonormal basis starts with the
    risk-factor columns and each column of ``X`` is projected out of it
    (numerically plain GS - for the well-conditioned Barra-style factor
    blocks used here; OLS residualisation is the numerically robust
    alternative and is used by the backtest facade).

``batched_spearman_ic(factor_df, forward_returns)``
    Per-day Spearman IC.  **Tie handling: average ranks** (the correct
    Spearman definition) computed with sort + ``cummax``/``cummin`` group
    boundaries - *not* ``argsort(argsort(x))``, which breaks ties by
    position.  Result: ``{ic_series, ic_mean, ic_ir, t_stat, ...}``.

``quintile_split(factor, n=5)``
    Cross-sectional binning from integer ranks: ``label = (rank0 * n) //
    n_valid`` computed in **int64** on both backends, so GPU and CPU
    labels are bit-identical (a float32/float64 floor could disagree at
    bin boundaries).  Invalid (NaN) entries get ``-1``.

OOM safety
----------
``torch.cuda.OutOfMemoryError`` (and RuntimeError "out of memory") ->
``torch.cuda.empty_cache()`` -> **retry once on CPU** ->
``fell_back_to_cpu=True`` (+ ``fallback_reason``).  Any other CUDA error
also falls back to CPU once but is recorded with its reason; errors from
the CPU retry itself propagate (we fail loudly rather than fake numbers).

CUDA context lifetime (documented choice)
-----------------------------------------
PyTorch exposes no public API that destroys a CUDA context from inside a
living process, so:

1. the context is created **lazily** - never touched until the first
   kernel actually runs on the GPU;
2. after ``IDLE_RELEASE_S`` (60 s) without a kernel,
   ``maybe_release_idle()`` synchronises and returns every cached
   allocator block to the driver (``empty_cache``), so other processes
   can claim the VRAM while we sleep;
3. true context teardown is delegated to the **process boundary**: the
   sandbox worker subprocess (``python -m server.sandbox.worker``)
   exits on idle/kill switch, which destroys the context with it.

Purity: torch + numpy + pandas (optional, for input coercion) + stdlib.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("engine.gpu_tensor")

try:  # torch is optional at import time (engine/__init__ stays lazy)
    import torch
    _TORCH_OK = True
except Exception:  # pragma: no cover - environment without torch
    torch = None  # type: ignore[assignment]
    _TORCH_OK = False

#: seconds of inactivity after which cached VRAM is released to the driver
IDLE_RELEASE_S = 60.0
#: annualisation factor used by ic_ir / t_stat conventions on this platform
TRADING_DAYS = 252

_last_use: float = 0.0


# --------------------------------------------------------------------------- #
# result containers
# --------------------------------------------------------------------------- #
@dataclass
class KernelResult:
    """Generic kernel output: the array + which backend produced it."""

    value: np.ndarray
    backend: str                      # "cuda" | "cpu"
    fell_back_to_cpu: bool = False
    ms: float = 0.0
    fallback_reason: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value, "backend": self.backend,
            "fell_back_to_cpu": self.fell_back_to_cpu,
            "ms": round(self.ms, 3), "fallback_reason": self.fallback_reason,
        }


@dataclass
class IcResult:
    """``batched_spearman_ic`` output (dict-style and attribute access)."""

    ic_series: np.ndarray              # [T] float64, NaN where undefined
    ic_mean: float
    ic_ir: float                       # mean/std*sqrt(252), NaN-safe
    t_stat: float                      # mean/(std/sqrt(n_valid))
    backend: str                       # "cuda" | "cpu"
    fell_back_to_cpu: bool = False
    ms: float = 0.0
    fallback_reason: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ic_series": self.ic_series, "ic_mean": self.ic_mean,
            "ic_ir": self.ic_ir, "t_stat": self.t_stat,
            "backend": self.backend, "fell_back_to_cpu": self.fell_back_to_cpu,
            "ms": round(self.ms, 3), "fallback_reason": self.fallback_reason,
        }


# --------------------------------------------------------------------------- #
# backend plumbing
# --------------------------------------------------------------------------- #
def torch_available() -> bool:
    return bool(_TORCH_OK)


def cuda_available() -> bool:
    return bool(_TORCH_OK and torch.cuda.is_available())


def maybe_release_idle(idle_seconds: float = IDLE_RELEASE_S) -> bool:
    """Release cached VRAM when the last kernel is older than ``idle_seconds``.

    Returns True when a release actually happened.  See the module
    docstring for why the context itself is only freed at process exit.
    """
    global _last_use
    if not cuda_available() or _last_use == 0.0:
        return False
    now = time.monotonic()
    if now - _last_use < idle_seconds:
        return False
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - defensive
        logger.warning("idle CUDA release failed", exc_info=True)
        return False
    _last_use = 0.0
    logger.info("released cached CUDA memory after %.0fs idle", idle_seconds)
    return True


def _is_oom(exc: BaseException) -> bool:
    if _TORCH_OK and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _run_guarded(cpu_fn: Callable[[], Any], gpu_fn: Callable[[], Any],
                 *, prefer_cuda: bool, name: str) -> Tuple[Any, str, bool, Optional[str]]:
    """Run ``gpu_fn`` on CUDA; on OOM/other CUDA error retry once on CPU.

    Returns ``(value, backend, fell_back_to_cpu, fallback_reason)``.
    """
    global _last_use
    if not prefer_cuda or not cuda_available():
        out = cpu_fn()
        return out, "cpu", False, None
    maybe_release_idle()
    try:
        with torch.no_grad():
            out = gpu_fn()
        _last_use = time.monotonic()
        return out, "cuda", False, None
    except Exception as exc:  # noqa: BLE001 - guarded per-op fallback
        if not _is_oom(exc):
            # unexpected CUDA-side failure: still retry CPU once, but say why
            logger.warning("GPU kernel %s failed (%s: %s) - retrying on CPU",
                           name, type(exc).__name__, exc)
            reason = f"{type(exc).__name__}: {exc}"
        else:
            reason = f"oom: {exc}"
            try:
                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass
        logger.warning("GPU kernel %s fell back to CPU (%s)", name, reason)
        out = cpu_fn()          # retry once, CPU only - errors here propagate
        _last_use = time.monotonic()
        return out, "cpu", True, reason


def _to_numpy(x: Any) -> np.ndarray:
    """DataFrame / Series / tensor / array -> contiguous numpy (CPU)."""
    if hasattr(x, "detach"):                      # torch tensor
        return x.detach().cpu().numpy()
    if hasattr(x, "values") and not isinstance(x, np.ndarray):
        return np.asarray(x.values)               # pandas DataFrame/Series
    return np.asarray(x)


def _as_2d(x: Any, name: str) -> np.ndarray:
    arr = _to_numpy(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D [days, stocks], got shape {arr.shape}")
    return arr


def _rank2d(x: "torch.Tensor") -> "torch.Tensor":
    """Row-wise **average-tie** ranks of a [B, N] tensor (NaN must be gone).

    sort -> tie-group start/end via cummax/cummin -> average of 1-based
    positions inside the group -> scatter back.  Integer-valued ranks are
    exact in float32 for N < 2**24, so GPU and CPU agree exactly.
    """
    n = x.shape[-1]
    order = torch.argsort(x, dim=-1)
    sorted_x = torch.gather(x, -1, order)
    pos = torch.arange(n, device=x.device, dtype=x.dtype).expand_as(sorted_x)
    starts = torch.ones_like(sorted_x, dtype=torch.bool)
    if n > 1:
        starts[:, 1:] = sorted_x[:, 1:] != sorted_x[:, :-1]
    first = torch.cummax(torch.where(starts, pos, torch.zeros_like(pos)),
                         dim=-1).values
    ends = torch.ones_like(sorted_x, dtype=torch.bool)
    if n > 1:
        ends[:, :-1] = starts[:, 1:]
    marked = torch.where(ends, pos, torch.full_like(pos, float("inf")))
    last = torch.flip(torch.cummin(torch.flip(marked, [-1]), dim=-1).values, [-1])
    avg_sorted = (first + last) / 2.0 + 1.0
    ranks = torch.empty_like(sorted_x)
    ranks.scatter_(-1, order, avg_sorted)
    return ranks


def _rank_numpy(x: np.ndarray) -> np.ndarray:
    """CPU twin of :func:`_rank2d` (float64, same average-tie definition)."""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    for i in range(x.shape[0]):
        row = x[i]
        finite = np.isfinite(row)
        vals = row[finite]
        if vals.size == 0:
            continue
        order = np.argsort(vals, kind="stable")
        sv = vals[order]
        new_grp = np.empty(vals.size, dtype=bool)
        new_grp[0] = True
        if vals.size > 1:
            np.not_equal(sv[1:], sv[:-1], out=new_grp[1:])
        gid = np.cumsum(new_grp, dtype=np.int64) - 1
        counts = np.bincount(gid)
        rank_sum = np.bincount(gid, weights=np.arange(1, vals.size + 1,
                                                      dtype=np.float64))
        avg = rank_sum / counts
        r = np.empty(vals.size, dtype=np.float64)
        r[order] = avg[gid]
        out[i][finite] = r
    return out


def _pearson_rows(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """Row-wise Pearson of two [B, N] tensors, NaN-safe (pairwise mask)."""
    both = torch.isfinite(a) & torch.isfinite(b)
    cnt = both.sum(dim=-1)
    a0 = torch.where(both, a, torch.zeros_like(a))
    b0 = torch.where(both, b, torch.zeros_like(b))
    c = torch.clamp(cnt.to(a.dtype), min=1.0)
    sa, sb = a0.sum(dim=-1), b0.sum(dim=-1)
    va = (a0 * a0).sum(dim=-1) - sa * sa / c
    vb = (b0 * b0).sum(dim=-1) - sb * sb / c
    cov = (a0 * b0).sum(dim=-1) - sa * sb / c
    denom = torch.sqrt(torch.clamp(va, min=0.0) * torch.clamp(vb, min=0.0))
    r = cov / torch.where(denom > 0, denom, torch.full_like(denom, float("nan")))
    return torch.where((cnt >= 3) & torch.isfinite(r), r,
                       torch.full_like(r, float("nan")))


def _ic_stats(ic: np.ndarray) -> Tuple[float, float, float]:
    """(ic_mean, annualised ic_ir, t_stat) from a finite IC series."""
    x = np.asarray(ic, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(x.mean())
    if x.size < 2:
        return mean, float("nan"), float("nan")
    sd = float(x.std(ddof=1))
    if sd <= 0:
        return mean, float("nan"), float("nan")
    ic_ir = mean / sd * np.sqrt(TRADING_DAYS)
    t_stat = mean / (sd / np.sqrt(x.size))
    return mean, float(ic_ir), float(t_stat)


def _coerce_exposures(exposures: Union[np.ndarray, Sequence[Any]]
                      ) -> np.ndarray:
    """[T,N,F] array or sequence of [T,N] arrays -> contiguous [T,N,F]."""
    if isinstance(exposures, np.ndarray) and exposures.ndim == 3:
        arr = exposures
    else:
        mats = [_as_2d(e, "exposures[i]") for e in exposures]
        if not mats:
            raise ValueError("exposures must contain at least one factor")
        arr = np.stack(mats, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"exposures must be [days, stocks, features], got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _group_demean_torch(y: "torch.Tensor", g: "torch.Tensor") -> "torch.Tensor":
    """Within-group demeaning per day (g: [B, N] int64, -1 allowed).

    ``y`` may be ``[B, N]`` or ``[B, N, F]``; the index is expanded to
    match so ``scatter_add_`` sees equal ranks.
    """
    G = int(g.max().item()) + 2                      # +1 for the -1 channel
    g_idx = (g + 1).clamp(min=0)                     # -1 -> 0 scratch bin
    if y.dim() == 3:
        g_sc = g_idx.unsqueeze(-1).expand_as(y)
        sums = torch.zeros((y.shape[0], G, y.shape[2]), dtype=y.dtype,
                           device=y.device)
        counts = torch.zeros_like(sums)
        vals = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
        sums.scatter_add_(1, g_sc, vals)
        counts.scatter_add_(1, g_sc, torch.isfinite(y).to(y.dtype))
        mean = sums / counts.clamp(min=1.0)
        return y - mean.gather(1, g_idx.unsqueeze(-1).expand_as(y))
    sums = torch.zeros((y.shape[0], G), dtype=y.dtype, device=y.device)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(1, g_idx, torch.where(torch.isfinite(y), y,
                                            torch.zeros_like(y)))
    counts.scatter_add_(1, g_idx, torch.isfinite(y).to(y.dtype))
    mean = sums / counts.clamp(min=1.0)
    return y - mean.gather(1, g_idx)


# --------------------------------------------------------------------------- #
# kernel 1: batched cross-sectional OLS residuals
# --------------------------------------------------------------------------- #
def batched_ols_residuals(factor: Any,
                          exposures: Union[np.ndarray, Sequence[Any]],
                          groups: Optional[Any] = None,
                          *, prefer_cuda: bool = True) -> KernelResult:
    """Per-day OLS residual of ``factor`` on ``exposures`` (``[T,N]``).

    One batched ``torch.linalg.lstsq`` over ``[Days, Stocks, Features]``.
    ``groups`` (``[N]`` or ``[T,N]`` int labels) -> within-group demeaning
    of y **and** X first (Frisch-Waugh-Lovell), so the residual is
    orthogonal to both the continuous exposures and the group dummies.
    Non-finite entries are zero-filled for the solve and stay NaN in the
    output (they never influence the coefficients).
    """
    t0 = time.perf_counter()
    y_np = _as_2d(factor, "factor").astype(np.float32, copy=False)
    x_np = _coerce_exposures(exposures)
    T, N = y_np.shape
    if x_np.shape[0] != T or x_np.shape[1] != N:
        raise ValueError(f"exposures shape {x_np.shape} incompatible with "
                         f"factor shape {y_np.shape}")

    g_np = None
    if groups is not None:
        g_np = _to_numpy(groups)
        if g_np.ndim == 1:
            g_np = np.broadcast_to(g_np, (T, N))
        if g_np.shape != (T, N):
            raise ValueError(f"groups shape {g_np.shape} != ({T}, {N})")
        g_np = np.ascontiguousarray(g_np, dtype=np.int64)

    def cpu() -> np.ndarray:
        y = y_np.astype(np.float64)
        X = x_np.astype(np.float64)
        # Same invalid mask as the torch path (bad target *or* bad row):
        # zeroing the X row too is what keeps a missing target from
        # influencing the fit - keeping it would regress the stock on a
        # fabricated 0 and bias every coefficient (see docstring).
        invalid = ~np.isfinite(y) | ~np.isfinite(X).all(axis=-1)
        # Preserve NaNs through the optional group demeaning pass so invalid
        # rows are excluded from both sums and counts.  Only after demeaning
        # are they replaced by zero for the linear solve; the published
        # residual restores NaN at the same positions.
        y_s = np.where(invalid, np.nan, y)
        X_s = np.where(invalid[..., None], np.nan, X)
        if g_np is not None:
            y_s = _group_demean_numpy(y_s, g_np)
            X_s = _group_demean_numpy(X_s, g_np)
        y_s = np.where(invalid, 0.0, y_s)
        X_s = np.where(invalid[..., None], 0.0, X_s)
        # solve every day in one stacked call where possible; numpy lstsq
        # has no batch dim -> loop over days (still pure numerics, no GPU)
        out = np.full((T, N), np.nan, dtype=np.float64)
        for t in range(T):
            sol, *_ = np.linalg.lstsq(X_s[t], y_s[t], rcond=None)
            res = y_s[t] - X_s[t] @ sol
            res[invalid[t]] = np.nan
            out[t] = res
        return out

    def gpu() -> np.ndarray:
        dev = torch.device("cuda")
        y = torch.as_tensor(np.ascontiguousarray(y_np), device=dev)
        X = torch.as_tensor(np.ascontiguousarray(x_np), device=dev)
        invalid = ~torch.isfinite(y) | ~torch.isfinite(X).all(dim=-1)
        y_s = torch.where(invalid, torch.full_like(y, float("nan")), y)
        X_s = torch.where(invalid[..., None],
                          torch.full_like(X, float("nan")), X)
        if g_np is not None:
            g = torch.as_tensor(g_np, device=dev)
            y_s = _group_demean_torch(y_s, g)
            X_s = _group_demean_torch(X_s, g)
        y_s = torch.where(invalid, torch.zeros_like(y_s), y_s)
        X_s = torch.where(invalid[..., None], torch.zeros_like(X_s), X_s)
        sol = torch.linalg.lstsq(X_s, y_s.unsqueeze(-1)).solution.squeeze(-1)
        res = y_s - torch.einsum("tnf,tf->tn", X_s, sol)
        res = torch.where(invalid, torch.full_like(res, float("nan")), res)
        return res.cpu().numpy().astype(np.float64)

    value, backend, fell, reason = _run_guarded(cpu, gpu,
                                                prefer_cuda=prefer_cuda,
                                                name="batched_ols_residuals")
    return KernelResult(value=value, backend=backend, fell_back_to_cpu=fell,
                        ms=(time.perf_counter() - t0) * 1000.0,
                        fallback_reason=reason)


def _group_demean_numpy(y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """NumPy twin of :func:`_group_demean_torch` (2-D or 3-D ``y``).

    Uses flat ``bincount`` accumulation - ``np.add.at`` with fewer index
    arrays than the target has dimensions silently produces an extra
    trailing axis (broadcast error), which bit us once already.
    """
    T = y.shape[0]
    G = int(g.max()) + 2
    gi = np.clip(np.asarray(g, dtype=np.int64) + 1, 0, None)   # -1 -> bin 0
    if y.ndim == 3:
        out = np.empty_like(y)
        for k in range(y.shape[2]):
            out[:, :, k] = _group_demean_numpy(y[:, :, k], g)
        return out
    idx = (np.arange(T, dtype=np.int64)[:, None] * G + gi).ravel()
    ok = np.isfinite(y)
    sums = np.bincount(idx, weights=np.where(ok, y, 0.0).ravel(),
                       minlength=T * G).reshape(T, G)
    cnts = np.bincount(idx, weights=ok.astype(np.float64).ravel(),
                       minlength=T * G).reshape(T, G)
    mean = sums / np.maximum(cnts, 1.0)
    return y - np.take_along_axis(mean, gi, axis=1)


# --------------------------------------------------------------------------- #
# kernel 2: Gram-Schmidt orthogonalisation
# --------------------------------------------------------------------------- #
def gram_schmidt_orthogonal(X: Any, risk_factors: Any,
                            *, prefer_cuda: bool = True) -> KernelResult:
    """Orthogonalise every column of ``X`` against ``risk_factors``.

    ``X``: ``[T,N]`` or ``[T,N,K]``; ``risk_factors``: ``[T,N]`` or
    ``[T,N,M]`` (or a sequence of ``[T,N]``).  Classical Gram-Schmidt
    **with a per-day orthonormal basis**: the risk-factor block is
    GS-orthogonalised and unit-normalised first, then each column of
    ``X`` is swept out of that basis (one sweep against orthonormal
    vectors is exact; a second sweep guards float32 rounding).  Only
    unit-normalising - without orthogonalising the risk block against
    itself - lets each later projection reintroduce the earlier
    component (measured 3.9e-6 normalised residual instead of ~1e-7).  Fully batched over days (no
    Python day loop).  NaN inputs are treated as zeros inside
    projections (their output entries stay NaN).
    """
    t0 = time.perf_counter()
    x_in = _to_numpy(X)
    squeeze = x_in.ndim == 2
    x_np = x_in.reshape(x_in.shape[0], x_in.shape[1], -1).astype(np.float32,
                                                                copy=False)
    if isinstance(risk_factors, np.ndarray) and risk_factors.ndim == 3:
        r_np = risk_factors
    else:
        r_np = np.stack([_as_2d(r, "risk_factors[i]") for r in risk_factors],
                        axis=-1)
    r_np = np.ascontiguousarray(r_np, dtype=np.float32)
    T, N = x_np.shape[0], x_np.shape[1]
    if r_np.shape[0] != T or r_np.shape[1] != N:
        raise ValueError("risk_factors incompatible with X")
    K = x_np.shape[2]

    def _normalize_np(v: np.ndarray) -> np.ndarray:
        n = np.sqrt((v * v).sum(axis=1, keepdims=True))
        return v / np.where(n > 0, n, 1.0)

    def _normalize_torch(v: "torch.Tensor") -> "torch.Tensor":
        n = torch.sqrt((v * v).sum(dim=1, keepdim=True))
        return v / torch.where(n > 0, n, torch.ones_like(n))

    def cpu() -> np.ndarray:
        rf0 = np.where(np.isfinite(r_np), r_np, 0.0).astype(np.float64)
        # true orthonormal basis: GS-orthogonalise the risk block first,
        # then each X column; a sweep over orthonormal q's is exact.
        Q: list = []
        for m in range(rf0.shape[-1]):
            v = rf0[:, :, m].copy()
            for _pass in range(2):
                for q in Q:
                    v = v - (v * q).sum(axis=1, keepdims=True) * q
            Q.append(_normalize_np(v))
        out_cols = []
        for k in range(K):
            v = x_np[:, :, k].astype(np.float64).copy()
            v0 = np.where(np.isfinite(v), v, 0.0)
            for _pass in range(2):            # second sweep guards rounding
                for q in Q:
                    v0 = v0 - (v0 * q).sum(axis=1, keepdims=True) * q
            v0[~np.isfinite(x_np[:, :, k])] = np.nan
            out_cols.append(v0)
            Q.append(_normalize_np(np.where(np.isfinite(v0), v0, 0.0)))
        return np.stack(out_cols, axis=-1)

    def gpu() -> np.ndarray:
        dev = torch.device("cuda")
        rf_t = torch.as_tensor(np.ascontiguousarray(r_np), device=dev)
        rf0 = torch.where(torch.isfinite(rf_t), rf_t, torch.zeros_like(rf_t))
        Q: list = []
        for m in range(rf0.shape[-1]):
            v = rf0[:, :, m]
            for _pass in range(2):
                for q in Q:
                    v = v - (v * q).sum(dim=1, keepdim=True) * q
            Q.append(_normalize_torch(v))
        x_t = torch.as_tensor(np.ascontiguousarray(x_np), device=dev)
        nan_mask = ~torch.isfinite(x_t)
        out_cols = []
        for k in range(K):
            v = torch.where(nan_mask[:, :, k], torch.zeros_like(x_t[:, :, k]),
                            x_t[:, :, k])
            for _pass in range(2):
                for q in Q:
                    v = v - (v * q).sum(dim=1, keepdim=True) * q
            v = torch.where(nan_mask[:, :, k], torch.full_like(v, float("nan")),
                            v)
            out_cols.append(v)
            Q.append(_normalize_torch(torch.where(torch.isfinite(v), v,
                                                  torch.zeros_like(v))))
        return torch.stack(out_cols, dim=-1).cpu().numpy().astype(np.float64)

    value, backend, fell, reason = _run_guarded(cpu, gpu,
                                                prefer_cuda=prefer_cuda,
                                                name="gram_schmidt_orthogonal")
    if squeeze:
        value = value[:, :, 0]
    return KernelResult(value=value, backend=backend, fell_back_to_cpu=fell,
                        ms=(time.perf_counter() - t0) * 1000.0,
                        fallback_reason=reason)


# --------------------------------------------------------------------------- #
# kernel 3: batched Spearman rank IC
# --------------------------------------------------------------------------- #
def batched_spearman_ic(factor_df: Any, forward_returns: Any,
                        *, prefer_cuda: bool = True) -> IcResult:
    """Per-day Spearman IC of factor vs forward returns.

    ``factor_df`` / ``forward_returns``: DataFrame (index=dates,
    columns=stocks) or ``[T, N]`` array.  DataFrames must already be
    aligned by the caller when mixed with arrays; two DataFrames are
    reindexed onto the factor's axes.

    **Ties are handled with average ranks** (correct Spearman); the
    double-argsort shortcut is explicitly not used.  Ranks are computed
    on the **pairwise-complete subset** of each row - the
    scipy/pandas ``.corr(method="spearman")`` convention (ranking each
    series first and masking afterwards differs exactly on rows where
    the two NaN patterns differ).  Days with <3 valid pairs give NaN.
    """
    t0 = time.perf_counter()
    f_np = _as_2d(factor_df, "factor_df")
    if hasattr(forward_returns, "index") and hasattr(factor_df, "index"):
        fwd = _as_2d(forward_returns.reindex(index=factor_df.index,
                                             columns=factor_df.columns),
                     "forward_returns")
    else:
        fwd = _as_2d(forward_returns, "forward_returns")
    if f_np.shape != fwd.shape:
        raise ValueError(f"shape mismatch: factor {f_np.shape} vs "
                         f"forward {fwd.shape}")
    f32 = np.ascontiguousarray(f_np, dtype=np.float32)
    r32 = np.ascontiguousarray(fwd, dtype=np.float32)

    def cpu() -> np.ndarray:
        both = np.isfinite(f32) & np.isfinite(r32)
        fr = _rank_numpy(np.where(both, f32, np.nan).astype(np.float64))
        rw = _rank_numpy(np.where(both, r32, np.nan).astype(np.float64))
        # pairwise-NaN Pearson on ranks (identical discipline as GPU path)
        both = np.isfinite(fr) & np.isfinite(rw)
        cnt = both.sum(axis=1)
        fr0 = np.where(both, fr, 0.0)
        rw0 = np.where(both, rw, 0.0)
        c = np.maximum(cnt.astype(np.float64), 1.0)
        va = (fr0 * fr0).sum(1) - fr0.sum(1) ** 2 / c
        vb = (rw0 * rw0).sum(1) - rw0.sum(1) ** 2 / c
        cov = (fr0 * rw0).sum(1) - fr0.sum(1) * rw0.sum(1) / c
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cov / np.sqrt(np.clip(va, 0, None) * np.clip(vb, 0, None))
        return np.where((cnt >= 3) & np.isfinite(r), r, np.nan)

    def gpu() -> np.ndarray:
        dev = torch.device("cuda")
        f = torch.as_tensor(f32, device=dev)
        r = torch.as_tensor(r32, device=dev)
        both = torch.isfinite(f) & torch.isfinite(r)
        # rank ONLY the pairwise-complete subset (+inf sentinel keeps the
        # common entries at positions 1..n_common, then non-common -> NaN)
        fs = torch.where(both, f, torch.full_like(f, float("inf")))
        rs = torch.where(both, r, torch.full_like(r, float("inf")))
        fr = _rank2d(fs)
        rw = _rank2d(rs)
        fr = torch.where(both, fr, torch.full_like(fr, float("nan")))
        rw = torch.where(both, rw, torch.full_like(rw, float("nan")))
        ic = _pearson_rows(fr, rw)
        return ic.cpu().numpy().astype(np.float64)

    ic, backend, fell, reason = _run_guarded(cpu, gpu,
                                             prefer_cuda=prefer_cuda,
                                             name="batched_spearman_ic")
    ic_mean, ic_ir, t_stat = _ic_stats(ic)
    return IcResult(ic_series=np.asarray(ic, dtype=np.float64),
                    ic_mean=ic_mean, ic_ir=ic_ir, t_stat=t_stat,
                    backend=backend, fell_back_to_cpu=fell,
                    ms=(time.perf_counter() - t0) * 1000.0,
                    fallback_reason=reason)


# --------------------------------------------------------------------------- #
# kernel 4: cross-sectional quintile split
# --------------------------------------------------------------------------- #
def quintile_split(factor: Any, n: int = 5, *,
                   prefer_cuda: bool = True) -> KernelResult:
    """Cross-sectional ``n``-way split -> int8 ``[T, N]`` labels (``-1``=NaN).

    ``label = min((rank0 * n) // n_valid, n-1)`` in **int64** on both
    backends, so GPU/CPU outputs are bit-identical.
    """
    t0 = time.perf_counter()
    if n < 2:
        raise ValueError("n must be >= 2")
    f_np = np.ascontiguousarray(_as_2d(factor, "factor"), dtype=np.float32)

    def cpu() -> np.ndarray:
        ranks = _rank_numpy(f_np.astype(np.float64))
        valid = np.isfinite(ranks)
        n_valid = valid.sum(axis=1, keepdims=True)
        r0i = np.rint(np.where(valid, ranks - 1.0, 0.0)).astype(np.int64)
        nv = np.maximum(n_valid, 1).astype(np.int64)
        labels = np.clip((r0i * int(n)) // nv, 0, n - 1).astype(np.int8)
        labels[~valid] = -1
        labels[np.broadcast_to(n_valid < 2, ranks.shape)] = -1
        return labels

    def gpu() -> np.ndarray:
        dev = torch.device("cuda")
        f = torch.as_tensor(f_np, device=dev)
        finite = torch.isfinite(f)
        # +inf sentinel sorts NaNs last -> valid ranks keep 1..n_valid
        safe = torch.where(finite, f, torch.full_like(f, float("inf")))
        ranks = _rank2d(safe)
        ranks = torch.where(finite, ranks, torch.full_like(ranks, float("nan")))
        valid = torch.isfinite(ranks)
        n_valid = valid.sum(dim=1, keepdim=True)
        r0 = torch.where(valid, ranks - 1.0, torch.zeros_like(ranks)).long()
        nv = n_valid.clamp(min=1).long()
        labels = torch.div(r0 * int(n), nv, rounding_mode="floor")
        labels = labels.clamp(0, n - 1).to(torch.int8)
        labels[~valid] = -1
        labels[(n_valid < 2).expand_as(labels)] = -1
        return labels.cpu().numpy()

    value, backend, fell, reason = _run_guarded(cpu, gpu,
                                                prefer_cuda=prefer_cuda,
                                                name="quintile_split")
    return KernelResult(value=value.astype(np.int8, copy=False),
                        backend=backend, fell_back_to_cpu=fell,
                        ms=(time.perf_counter() - t0) * 1000.0,
                        fallback_reason=reason)


__all__ = [
    "IDLE_RELEASE_S", "IcResult", "KernelResult",
    "batched_ols_residuals", "batched_spearman_ic", "cuda_available",
    "gram_schmidt_orthogonal", "maybe_release_idle", "quintile_split",
    "torch_available",
]
