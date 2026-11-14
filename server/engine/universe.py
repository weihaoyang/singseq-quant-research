"""Universe hygiene matrices (S3): PIT-ST, listing age, suspension, eligibility.

All inputs are the dense panel contract arrays (date-major ``[T, N]``);
outputs are boolean/float ``[T, N]`` matrices computed with numpy only.

Rules (ARCHITECTURE.md H7 - no synthetic stand-ins, real data only):

* ST  : point-in-time from ``namechange`` intervals (name contains
        ``ST``/``退``); when a code has no historical interval it is
        conservatively excluded by the canonical data service.
* Age : calendar days since listing >= 60 (``list_date`` [N]).
* Susp: ``vol <= 0`` or non-finite amount -> suspended.
* Eligible = ~st & ~suspended & age_ok & has_limit_prices.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

MIN_LIST_DAYS = 60


_BAD_DAY = 1_000_000_000      # sentinel day number: age = dd - 1e9 << 0


def yyyymmdd_to_days(values) -> np.ndarray:
    """YYYYMMDD integers -> day numbers since 1970-01-01 (int64).

    Invalid / missing entries (0, NaT-ish, out of range) map to
    ``_BAD_DAY`` so the resulting age is hugely negative and the name
    stays excluded until it actually has a listing date.
    """
    a = np.asarray(values)
    if a.dtype.kind not in "iu":
        a = np.nan_to_num(a.astype(np.float64), nan=0.0).astype(np.int64)
    else:
        a = a.astype(np.int64, copy=False)
    valid = (a >= 19000101) & (a <= 29991231)
    out = np.full(a.shape, _BAD_DAY, dtype=np.int64)
    sel = a[valid]
    if sel.size:
        # vectorised string build -> numpy datetime64 parse (exact calendar
        # days, unlike raw YYYYMMDD integer subtraction)
        strs = np.array([f"{x // 10000:04d}-{x // 100 % 100:02d}-{x % 100:02d}"
                         for x in sel], dtype="datetime64[D]")
        out[valid] = strs.astype(np.int64)
    return out


def listing_age_days(list_date: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Calendar days between listing date and each trade date -> ``[T, N]``."""
    ld = yyyymmdd_to_days(list_date)
    dd = yyyymmdd_to_days(dates)
    return dd[:, None] - ld[None, :]


def age_ok_matrix(list_date: np.ndarray, dates: np.ndarray,
                  min_days: int = MIN_LIST_DAYS) -> np.ndarray:
    age = listing_age_days(list_date, dates)
    return (age >= min_days) & (age < 100_000)


def suspension_mask(vol: np.ndarray, amount: np.ndarray) -> np.ndarray:
    """Suspended when no volume traded or amount unusable -> ``[T, N]``.

    Works directly on the stored dtype (the Arrow panel keeps these as
    float32): converting a [484, 5219] matrix to float64 costs a 20 MB
    copy per call for results that are bit-identical.
    """
    vol = np.asarray(vol)
    amount = np.asarray(amount)
    return (vol <= 0) | ~np.isfinite(vol) | ~np.isfinite(amount)


def st_mask_from_intervals(codes: np.ndarray, dates: np.ndarray,
                           intervals) -> np.ndarray:
    """Build PIT ST mask from ``(code, start_date, end_date, name)`` rows."""
    T, N = len(dates), len(codes)
    mask = np.zeros((T, N), dtype=bool)
    index = {c: i for i, c in enumerate(codes)}
    # Dates arrive in the project contract as YYYYMMDD integers/strings;
    # parsing integers directly as datetime64 would interpret them as epoch
    # day offsets and silently disable every PIT interval.
    raw_dates = np.asarray(dates)
    dd = np.array([
        f"{int(x) // 10000:04d}-{int(x) // 100 % 100:02d}-{int(x) % 100:02d}"
        for x in raw_dates
    ], dtype="datetime64[D]")
    for code, start, end, name in intervals:
        j = index.get(code)
        if j is None:
            continue
        st_name = isinstance(name, str) and ("ST" in name.upper() or "退" in name)
        if not st_name:
            continue
        try:
            sv = int(str(start)[:8])
            s = np.datetime64(f"{sv // 10000:04d}-{sv // 100 % 100:02d}-{sv % 100:02d}", "D")
        except (ValueError, TypeError):
            continue
        if end in (None, "", "NaT", "nan"):
            e = np.datetime64("2099-12-31", "D")
        else:
            try:
                ev = int(str(end)[:8])
                e = np.datetime64(f"{ev // 10000:04d}-{ev // 100 % 100:02d}-{ev % 100:02d}", "D")
            except (ValueError, TypeError):
                e = np.datetime64("2099-12-31", "D")
        if e < s:
            continue
        sel = (dd >= s) & (dd <= e)
        if sel.any():
            mask[sel, j] = True
    return mask


def st_mask_from_names(dates: np.ndarray, codes: np.ndarray,
                       names) -> np.ndarray:
    """Legacy helper; canonical replay never infers PIT status from current names.

    The canonical data service excludes names with no historical interval.
    """
    T, N = len(dates), len(codes)
    mask = np.zeros((T, N), dtype=bool)
    for j, nm in enumerate(names or []):
        if isinstance(nm, str) and ("ST" in nm.upper() or "退" in nm):
            mask[:, j] = True
    return mask


def eligibility_matrix(st: np.ndarray, suspended: np.ndarray, age_ok: np.ndarray,
                       has_limits: Optional[np.ndarray] = None) -> np.ndarray:
    ok = (~st) & (~suspended) & age_ok
    if has_limits is not None:
        ok &= has_limits
    return ok


def has_limit_prices(up_limit: np.ndarray, down_limit: np.ndarray) -> np.ndarray:
    up = np.asarray(up_limit, dtype=np.float64)
    dn = np.asarray(down_limit, dtype=np.float64)
    return np.isfinite(up) & np.isfinite(dn) & (up > 0) & (dn > 0)


def neutralize_matrix(factor: np.ndarray, eligible: np.ndarray,
                      industry_id: Optional[np.ndarray] = None,
                      circ_mv: Optional[np.ndarray] = None, *,
                      min_n: int = 20) -> np.ndarray:
    """Cross-sectional residualisation vs PIT industry + log market cap.

    For each date, regress the factor on industry dummies (drop-first) and the
    demeaned log market cap over eligible, finite rows, and return the
    residual as the neutralised factor. Rows with fewer than ``min_n`` valid
    observations are returned unchanged (never fabricated); NaN is preserved.
    Numpy-only.
    """
    f = np.asarray(factor, dtype=np.float64).copy()
    T, N = f.shape
    if industry_id is None and circ_mv is None:
        return f
    ind = None if industry_id is None else np.asarray(industry_id)
    size = None
    if circ_mv is not None:
        mv = np.asarray(circ_mv, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            size = np.where(mv > 0, np.log(mv), np.nan)
    elig = np.asarray(eligible, dtype=bool)
    for t in range(T):
        row = f[t]
        valid = np.isfinite(row) & elig[t]
        if size is not None:
            valid = valid & np.isfinite(size[t])
        n = int(valid.sum())
        if n < min_n:
            continue
        cols = [np.ones(n, dtype=np.float64)]
        if size is not None:
            s = size[t][valid]
            cols.append(s - s.mean())
        if ind is not None:
            labels = ind[t][valid].astype(str)
            uniq, inv = np.unique(labels, return_inverse=True)
            for k in range(1, len(uniq)):
                cols.append((inv == k).astype(np.float64))
        X = np.column_stack(cols)
        y = row[valid]
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:            # pragma: no cover - singular
            continue
        resid = y - X @ beta
        out = np.full(N, np.nan, dtype=np.float64)
        out[valid] = resid
        f[t] = out
    return f
