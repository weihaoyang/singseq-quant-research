"""Main backtest loop (S3) - ARCHITECTURE.md 3.4.

Contract::

    def run_backtest(
        panel: Panel,
        config: BacktestConfig,
        *,
        backend: Literal["auto", "cpu", "gpu"] = "auto",
    ) -> BacktestResult

``BacktestResult`` carries the exact fields mandated by 3.4:
``group_curves[5] ls_curve benchmark_curve ic_series ic_decay
underwater turnover_series friction_breakdown metrics
acceleration_mode blocked_orders``.

Economics (documented, deterministic) - **next-day-open matching**:

* signal formed at close of day *t* -> the order is executed at the
  **open of day t+1** (never at the same-day close); the weights held
  after that fill earn ``adj_open[t+2] / adj_open[t+1] - 1`` where
  ``adj_open = open * adj_close / close`` (adjustment-factor-consistent
  open price);
* limit-up/limit-down locks, suspension, ST and listing-age checks,
  the daily amount used by the square-root impact model **and** the
  stamp-duty policy date are all read from the **execution day t+1**
  (the day the order actually trades);
* costs are charged on the execution day *t+1*
  (stamp/commission/slippage plus financing and borrow fees from
  ``engine.friction``);
* the last signal (close of day *T-2*) still executes on day *T-1* but
  has no forward open inside the window -> its row carries costs with a
  0 return (position held beyond the sample, standard practice);
* the benchmark stays close-to-close but is shifted one row
  (``benchmark_ret[t+1]``, i.e. close t+1 -> close t+2) so both curves
  span the same calendar window (documented deviation: the index has no
  open price in the panel);
* ``ls_curve`` is **net of friction**; ``ls_curve_gross`` (extra field)
  and the 5 group curves are gross diagnostics;
* turnover = ``0.5 * sum(|executed delta|)``; the initial build is 1.0
  for the dollar-neutral long-short book (gross exposure 2);
* benchmark = index returns carried in ``panel.benchmark_ret``
  (沪深300, stored in the Arrow schema metadata); missing benchmark data is
  a hard replay failure and is never replaced with an equal-weight proxy.

GPU wiring (H8): rank-IC series and IC-decay run through
``engine.gpu.GPUTensorEngine`` when ``backend`` is ``auto``/``gpu``;
the day loop stays pure numpy.  ``acceleration_mode`` is ``"cpu"``,
``"cuda"`` or ``"cpu-fallback"`` (real OOM / probe decline).

Purity: numpy + stdlib only (torch is imported lazily, only when the
GPU path is actually requested).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from . import metrics as M
from . import universe as U
from .friction import FrictionConfig, TradingStateMachine

Backend = Literal["auto", "cpu", "gpu"]


# --------------------------------------------------------------------------- #
# input contract
# --------------------------------------------------------------------------- #
@dataclass
class Panel:
    """Dense date-major panel: every per-row array is ``[T, N]``."""

    dates: np.ndarray                 # [T] int32  YYYYMMDD
    codes: np.ndarray                 # [N] str
    factor: np.ndarray                # [T, N] float
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    adj_close: np.ndarray
    up_limit: np.ndarray
    down_limit: np.ndarray
    vol: np.ndarray                   # [T, N] float
    amount: np.ndarray                # [T, N] float (元)
    list_date: np.ndarray             # [N] YYYYMMDD or datetime64
    st: np.ndarray                    # [T, N] bool (point-in-time)
    benchmark_ret: Optional[np.ndarray] = None   # [T] float, index return
    # optional cross-sectional inputs used by alpha baselines (real columns)
    circ_mv: Optional[np.ndarray] = None         # [T, N] float, 万元
    pb: Optional[np.ndarray] = None              # [T, N] float
    turnover_rate: Optional[np.ndarray] = None   # [T, N] float, %
    industry_id: Optional[np.ndarray] = None     # [T, N] versioned PIT key
    pool_membership: Optional[np.ndarray] = None # [T, N] PIT index pool mask
    cash_dividend: Optional[np.ndarray] = None   # [T, N] after-tax CNY/share
    shortable: Optional[np.ndarray] = None      # [T, N] authoritative borrow flag
    borrow_rate: Optional[np.ndarray] = None     # [T, N] annualized per-name borrow rate
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        T = len(self.dates)
        N = len(self.codes)
        if self.factor.shape != (T, N):
            raise ValueError(f"factor shape {self.factor.shape} != ({T}, {N})")
        for name in ("open", "high", "low", "close", "adj_close",
                     "up_limit", "down_limit", "vol", "amount", "st",
                     "circ_mv", "pb", "turnover_rate", "industry_id", "pool_membership",
                     "cash_dividend", "shortable", "borrow_rate"):
            arr = getattr(self, name)
            if arr is not None and arr.shape != (T, N):
                raise ValueError(f"{name} shape {arr.shape} != ({T}, {N})")

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.dates), len(self.codes))

    def window(self, start: Optional[int], end: Optional[int]) -> "Panel":
        """Row-block view ``[start, end)`` (no copies of the big matrices)."""
        T = self.shape[0]
        s = 0 if start is None else int(start)
        e = T if end is None else int(end)
        if not (0 <= s < e <= T):
            raise ValueError(f"bad window [{s}, {e}) for T={T}")
        sl = slice(s, e)
        kw = {n: getattr(self, n) for n in
              ("factor", "open", "high", "low", "close", "adj_close",
               "up_limit", "down_limit", "vol", "amount", "st",
               "circ_mv", "pb", "turnover_rate", "industry_id", "pool_membership",
               "cash_dividend", "shortable", "borrow_rate")}

        for n in list(kw):
            if kw[n] is not None:
                kw[n] = kw[n][sl]
        bench = None if self.benchmark_ret is None else self.benchmark_ret[sl]
        return Panel(dates=self.dates[sl], codes=self.codes,
                     benchmark_ret=bench, list_date=self.list_date,
                     meta=dict(self.meta), **kw)

    def date_index(self, yyyymmdd: int) -> int:
        idx = int(np.searchsorted(self.dates, np.int32(yyyymmdd)))
        return min(idx, self.shape[0] - 1)


# --------------------------------------------------------------------------- #
# config / result
# --------------------------------------------------------------------------- #
@dataclass
class BacktestConfig:
    n_groups: int = 5
    start_date: Optional[int] = None
    end_date: Optional[int] = None
    rebalance: str = "daily"
    costs_enabled: bool = True
    lags: Sequence[int] = (1, 2, 3, 5, 10, 20)
    min_list_days: int = U.MIN_LIST_DAYS
    # Declared number of trials/hypotheses behind this result.  Feeds the
    # deflated-Sharpe multiple-testing adjustment; 1 means a single
    # pre-specified hypothesis (no search-driven Sharpe inflation to correct).
    n_trials: int = 1
    # Residualise the factor against PIT industry + log market cap before
    # ranking (cross-sectional neutralisation). Uses panel.industry_id and
    # panel.circ_mv; silently a no-op when neither is available.
    neutralize: bool = False
    friction: FrictionConfig = field(default_factory=FrictionConfig)

    def __post_init__(self) -> None:
        if self.n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if int(self.n_trials) < 1:
            raise ValueError("n_trials must be >= 1")
        if self.rebalance != "daily":
            raise ValueError("only daily rebalance is implemented (no synthetic fills)")
        try:
            lags = tuple(int(lag) for lag in self.lags)
        except (TypeError, ValueError) as exc:
            raise ValueError("lags must be a non-empty sequence of integers") from exc
        if not lags or any(lag <= 0 for lag in lags):
            raise ValueError("lags must contain positive values")
        self.lags = lags


@dataclass
class BacktestResult:
    dates: np.ndarray
    codes: np.ndarray
    group_curves: List[np.ndarray]        # exactly n_groups cumulative curves
    ls_curve: np.ndarray                  # net-of-friction equity curve
    benchmark_curve: np.ndarray
    ic_series: np.ndarray                 # [T] Spearman IC (last 2 days NaN)
    ic_decay: Dict[int, float]            # lag -> mean IC
    underwater: np.ndarray                # drawdown of ls_curve
    turnover_series: np.ndarray           # [T] 0.5*sum|delta| per trade date
    friction_breakdown: Dict[str, float]  # {stamp, commission, slippage}
    metrics: Dict[str, float]             # {sharpe,calmar,icir,maxdd,winrate,annualized}
    acceleration_mode: str                # cpu | cuda | cpu-fallback
    blocked_orders: Dict[str, int]        # {limit_up_buy, limit_down_sell}
    # --- extra (contract-compatible additions) ------------------------- #
    normal_ic_series: Optional[np.ndarray] = None
    cross_sectional_autocorrelation_series: Optional[np.ndarray] = None
    ls_curve_gross: Optional[np.ndarray] = None
    group_daily: Optional[np.ndarray] = None      # [G, T-1]
    ls_daily: Optional[np.ndarray] = None         # [T-1] net
    ls_daily_gross: Optional[np.ndarray] = None
    benchmark_daily: Optional[np.ndarray] = None  # [T-1]
    group_membership: Optional[np.ndarray] = None # [T, N] int8, G = ineligible
    formation_ms: float = 0.0
    notes: Dict[str, object] = field(default_factory=dict)
    ledger: Dict[str, object] = field(default_factory=dict)
    # Daily account and exposure curves aligned to ``dates``.  These are
    # derived from the share ledger after each real execution, so callers can
    # independently reconcile equity, cash and gross/net risk.
    cash_curve: Optional[np.ndarray] = None
    equity_curve: Optional[np.ndarray] = None
    gross_exposure_series: Optional[np.ndarray] = None
    net_exposure_series: Optional[np.ndarray] = None
    long_notional_series: Optional[np.ndarray] = None
    short_notional_series: Optional[np.ndarray] = None

    @property
    def n_groups(self) -> int:
        return len(self.group_curves)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _gpu_module():
    from . import gpu as gpu_mod      # lazy: torch import costs ~1.8s
    return gpu_mod


def _resolve_backend(backend: str):
    """-> (use_gpu: bool, mode: str, engine_or_None)"""
    gpu_mod = _gpu_module()
    if backend not in ("auto", "cpu", "gpu"):
        raise ValueError(f"unknown backend {backend!r}")
    if backend == "cpu" or not gpu_mod.cuda_available():
        if backend == "gpu":
            raise gpu_mod.GPUBackendUnavailable("backend='gpu' but CUDA is unavailable")
        return False, "cpu", None
    engine = gpu_mod.GPUTensorEngine()
    return True, engine.mode, engine


def _cumulative(daily: np.ndarray) -> np.ndarray:
    """Length T curve starting at 1.0; ``daily[t]`` is return t -> t+1."""
    d = np.asarray(daily, dtype=np.float64)
    curve = np.empty(d.shape[0] + 1, dtype=np.float64)
    curve[0] = 1.0
    # Missing observations are not zero returns.  Preserve them so a
    # published curve cannot silently manufacture performance through an
    # incomplete panel.  Callers may explicitly encode a terminal no-forward
    # period as 0.0 when the replay window ends.
    curve[1:] = np.nan
    running = 1.0
    for i, value in enumerate(d, start=1):
        if np.isfinite(value):
            running *= 1.0 + float(value)
            curve[i] = running
    return curve


def _underwater_from_curve(curve: np.ndarray) -> np.ndarray:
    """Drawdown series aligned with an equity curve (same length)."""
    c = np.asarray(curve, dtype=np.float64)
    peak = np.maximum.accumulate(c)
    return c / np.where(peak > 0, peak, 1.0) - 1.0


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def run_backtest(panel: Panel, config: Optional[BacktestConfig] = None,
                 *, backend: Backend = "auto") -> BacktestResult:
    """Pure compute. 2023-2024 full market: < 0.5s when CUDA is available."""
    t_start = time.perf_counter()
    cfg = config or BacktestConfig()

    # ---- window ---------------------------------------------------------- #
    start_idx = panel.date_index(cfg.start_date) if cfg.start_date else 0
    end_idx = panel.date_index(cfg.end_date) + 1 if cfg.end_date else panel.shape[0]
    if start_idx > 0 or end_idx < panel.shape[0]:
        panel = panel.window(start_idx, end_idx)

    T, N = panel.shape
    if T < 3 or N < 10:
        raise ValueError(f"panel too small: T={T}, N={N}")

    factor = np.asarray(panel.factor, dtype=np.float64)
    adj = np.asarray(panel.adj_close, dtype=np.float64)
    if not np.isfinite(adj).all() or (adj <= 0).any():
        adj = np.where(np.isfinite(adj) & (adj > 0), adj, np.nan)

    # ---- universe hygiene ------------------------------------------------ #
    suspended = U.suspension_mask(panel.vol, panel.amount)
    age_ok = U.age_ok_matrix(panel.list_date, panel.dates, cfg.min_list_days)
    limits_ok = U.has_limit_prices(panel.up_limit, panel.down_limit)
    st = np.asarray(panel.st, dtype=bool)
    pool_mask = (np.ones((T, N), dtype=bool) if panel.pool_membership is None
                 else np.asarray(panel.pool_membership, dtype=bool))
    cash_dividend = (np.zeros((T, N), dtype=np.float64)
                     if panel.cash_dividend is None
                     else np.asarray(panel.cash_dividend, dtype=np.float64))
    if cash_dividend.shape != (T, N) or not np.isfinite(cash_dividend).all():
        raise ValueError("authoritative cash_dividend must be finite [T, N]")
    shortable = (np.ones((T, N), dtype=bool)
                 if panel.shortable is None else np.asarray(panel.shortable, dtype=bool))
    if shortable.shape != (T, N):
        raise ValueError("authoritative shortable shape mismatch")
    borrow_rate = (None if panel.borrow_rate is None
                   else np.asarray(panel.borrow_rate, dtype=np.float64))
    if borrow_rate is not None and borrow_rate.shape != (T, N):
        raise ValueError("authoritative borrow_rate shape mismatch")
    eligible = (~st) & (~suspended) & age_ok & limits_ok & pool_mask      # [T, N]

    # ---- forward returns (next-day-open matching) ------------------------ #
    # adj_open: adjustment-consistent open price (same factor as adj_close)
    with np.errstate(invalid="ignore", divide="ignore"):
        adj_open = np.asarray(panel.open, dtype=np.float64) * adj \
            / np.asarray(panel.close, dtype=np.float64)
    adj_open[~np.isfinite(adj_open) | (adj_open <= 0)] = np.nan

    # signal at close t -> fill at open t+1 -> earn open t+1 -> open t+2
    fwd = np.full((T, N), np.nan, dtype=np.float64)
    if T >= 3:
        np.divide(adj_open[2:], adj_open[1:T - 1], out=fwd[:T - 2])
        fwd[:T - 2] -= 1.0
    fwd[~np.isfinite(fwd)] = np.nan
    # a return only exists when the name was tradable on the execution day
    fwd[:-1][~eligible[1:]] = np.nan

    # ---- one ranking pass -> quintile membership ------------------------- #
    # backend resolution (and the ~1.8 s cold torch import) happens first so
    # it is timed as ``backend_ms`` and does not pollute the kernel costs
    t_backend = time.perf_counter()
    use_gpu, mode, engine = _resolve_backend(backend)
    backend_ms = (time.perf_counter() - t_backend) * 1000.0
    notes: Dict[str, object] = {"backend_requested": backend}
    if bool(getattr(cfg, "neutralize", False)):
        if panel.industry_id is not None or panel.circ_mv is not None:
            factor = U.neutralize_matrix(factor, eligible, panel.industry_id,
                                         panel.circ_mv)
            notes["neutralize"] = "industry+size"
        else:
            # Requested but the panel carries no PIT industry / size exposure.
            notes["neutralize"] = "unavailable"
    else:
        notes["neutralize"] = False
    t_rank = time.perf_counter()
    masked_factor = np.where(eligible, factor, np.nan)
    if use_gpu and engine is not None:
        factor_rank = engine.rank_matrix(masked_factor)
    else:
        factor_rank = M.rank_rows(masked_factor)
    rnorm = M.rank_normalize(factor_rank)
    G = int(cfg.n_groups)
    with np.errstate(invalid="ignore"):
        gid = np.floor(np.where(np.isfinite(rnorm), rnorm, 1.0) * G).astype(np.int64)
    gid = np.clip(gid, 0, G - 1)
    gid[~np.isfinite(rnorm)] = G                       # channel G = ineligible
    rank_ms = (time.perf_counter() - t_rank) * 1000.0

    # ---- GPU-accelerated statistics (H8) --------------------------------- #
    # IC decay window starts at the NEXT open (factor known at close t):
    #   IC(lag L) = spearman(factor[t], adj_open[t+1+L]/adj_open[t+1] - 1)
    # realised as factor_input[s] = factor[s-1] vs prices[s+L]/prices[s]-1
    # with prices = adj_open (row 0 unusable -> NaN).
    decay_factor = np.empty_like(masked_factor)
    decay_factor[0] = np.nan
    decay_factor[1:] = masked_factor[:-1]
    decay_ranks = np.empty_like(factor_rank)
    decay_ranks[0] = np.nan
    decay_ranks[1:] = factor_rank[:-1]
    t_stat = time.perf_counter()
    if use_gpu and engine is not None:
        factor_f32 = np.ascontiguousarray(masked_factor, dtype=np.float32)
        fwd_f32 = np.ascontiguousarray(fwd, dtype=np.float32)
        ic_series = engine.batched_spearman_rank_ic(factor_f32, fwd_f32)
        ic_decay = engine.batched_ic_decay(
            np.ascontiguousarray(decay_factor, dtype=np.float32),
            np.ascontiguousarray(adj_open, dtype=np.float32),
            lags=cfg.lags)
        mode = engine.mode                       # -> "cuda" or "cpu-fallback"
        notes["gpu_uploads"] = list(engine.uploads)
        notes["gpu_device"] = engine.device_info()
    else:
        ic_series = M.spearman_ic_rows(masked_factor, fwd,
                                       factor_ranks=factor_rank)
        ic_decay = M.ic_decay_map(decay_factor, adj_open, lags=cfg.lags,
                                  factor_ranks=decay_ranks)
        mode = "cpu"
    ic_series = np.asarray(ic_series, dtype=np.float64)
    if ic_series.shape[0] == T - 1:               # align to dates
        ic_series = np.append(ic_series, np.nan)
    # Normal IC is the same dated next-open label paired with the raw factor
    # values.  Cross-sectional autocorrelation compares adjacent dated
    # exposures over their common eligible symbols; both remain NaN when the
    # real panel does not contain enough observations.
    normal_ic_series = M.normal_ic_series(masked_factor, fwd)
    cross_sectional_autocorrelation_series = (
        M.cross_sectional_autocorrelation(masked_factor))
    stats_ms = (time.perf_counter() - t_stat) * 1000.0

    # ---- group returns (gross, vectorised via bincount) ------------------ #
    t_grp = time.perf_counter()
    group_daily = np.full((G, T - 1), np.nan, dtype=np.float64)
    fwd_rows = fwd[:-1]
    fwd_valid = np.isfinite(fwd[:-1])
    for t in range(T - 1):
        # Missing next-open observations are excluded from both numerator
        # and denominator.  Counting them as zero would manufacture returns
        # on suspended/missing execution rows.
        valid = fwd_valid[t] & (gid[t] < G)
        g = gid[t, valid]
        # ``valid`` guarantees that the selected observations are finite;
        # passing the original values through keeps missing data from being
        # silently converted into a zero return.
        sums = np.bincount(g, weights=fwd_rows[t, valid], minlength=G + 1)
        cnts = np.bincount(g, minlength=G + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            valid_groups = cnts[:G] > 0
            group_daily[valid_groups, t] = (
                sums[:G][valid_groups] / cnts[:G][valid_groups])
        # The final execution has no next open inside the requested window;
        # encode that known terminal no-forward period explicitly.  Any other
        # empty group/day remains NaN and is never treated as a zero return.
        if t == T - 2:
            group_daily[:, t] = np.where(
                np.isfinite(group_daily[:, t]), group_daily[:, t], 0.0)
    group_ms = (time.perf_counter() - t_grp) * 1000.0

    # ---- friction state machine on the long-short book ------------------- #
    t_ls = time.perf_counter()
    fric_cfg = cfg.friction if cfg.costs_enabled else replace(cfg.friction, commission_rate=0.0,
                                                              impact_coeff=0.0)
    sm = TradingStateMachine(fric_cfg)
    top_c, bot_c = G - 1, 0
    ls_gross = np.zeros(T - 1, dtype=np.float64)
    ls_net = np.zeros(T - 1, dtype=np.float64)
    turnover = np.zeros(T, dtype=np.float64)
    cash_curve = np.full(T, np.nan, dtype=np.float64)
    equity_curve = np.full(T, np.nan, dtype=np.float64)
    gross_exposure = np.full(T, np.nan, dtype=np.float64)
    net_exposure = np.full(T, np.nan, dtype=np.float64)
    long_notional = np.full(T, np.nan, dtype=np.float64)
    short_notional = np.full(T, np.nan, dtype=np.float64)
    cash_curve[0] = float(fric_cfg.initial_cash)
    equity_curve[0] = float(fric_cfg.initial_cash)
    long_notional[0] = 0.0
    short_notional[0] = 0.0
    gross_exposure[0] = 0.0
    net_exposure[0] = 0.0

    # pre-compute 一字 locks for the whole window (real stk_limit columns;
    # indexed by EXECUTION day inside the loop below)
    up_lock = None
    down_lock = None
    if cfg.costs_enabled and cfg.friction.block_limit_moves:
        from .friction import one_word_limit_down, one_word_limit_up
        up_lock = one_word_limit_up(panel.open, panel.high, panel.low,
                                    panel.close, panel.up_limit)
        down_lock = one_word_limit_down(panel.open, panel.high, panel.low,
                                        panel.close, panel.down_limit)

    target_buf = np.zeros(N, dtype=np.float64)
    suspension_carry_days = 0
    suspension_carry_positions = 0
    for t in range(T - 1):
        # signal at close of day t -> order executes on day e = t+1 at the
        # next open; every trading rule (一字锁 / 停牌 / ST / 成交额 / 印花税
        # 政策日期) reads the bars of the EXECUTION day e.
        e = t + 1
        g_t = gid[t]
        top_n = int((g_t == top_c).sum())
        bot_n = int((g_t == bot_c).sum())
        target_buf[:] = 0.0
        if fric_cfg.allow_short:
            if top_n and bot_n:
                target_buf[g_t == top_c] = 1.0 / top_n
                target_buf[g_t == bot_c] = -1.0 / bot_n
        elif top_n:
            target_buf[g_t == top_c] = 1.0 / top_n
        res = sm.rebalance(
            date=int(panel.dates[e]), target=target_buf,
            close=panel.close[e], open_=panel.open[e], high=panel.high[e],
            low=panel.low[e], up_limit=panel.up_limit[e],
            down_limit=panel.down_limit[e], amount=panel.amount[e],
            suspended=suspended[e], tradable=eligible[e],
            up_lock_row=None if up_lock is None else up_lock[e],
            down_lock_row=None if down_lock is None else down_lock[e],
            cash_dividend=cash_dividend[e],
            shortable=shortable[e],
            borrow_rate=None if borrow_rate is None else borrow_rate[e],
        )
        turnover[t] = res.turnover
        held = sm.holdings
        # Reconcile account state from the share ledger at the actual fill
        # price.  ``holdings`` is a return-loop weight vector; the curves
        # below intentionally use signed notionals and post-fill equity.
        prices_e = TradingStateMachine._round_price(
            np.asarray(panel.open[e], dtype=np.float64), fric_cfg.tick_size)
        shares_e = np.asarray(sm.shares, dtype=np.float64)
        notionals_e = shares_e * np.where(np.isfinite(prices_e), prices_e, 0.0)
        equity_e = float(sm.equity)
        if equity_e <= 0 or not np.isfinite(equity_e):
            raise ValueError("canonical ledger produced non-positive equity")
        cash_curve[e] = float(sm.cash)
        equity_curve[e] = equity_e
        long_e = float(np.maximum(notionals_e, 0.0).sum())
        short_e = float(np.maximum(-notionals_e, 0.0).sum())
        long_notional[e] = long_e
        short_notional[e] = short_e
        gross_exposure[e] = (long_e + short_e) / equity_e
        net_exposure[e] = (long_e - short_e) / equity_e
        # Holdings filled at open e are marked at the next authoritative open.
        # A suspended next bar has no executable quote: keep the last valid
        # mark frozen and realise the move on the first valid re-open.  A
        # missing quote without suspension evidence is a data-integrity
        # failure, never a fabricated zero return.  The final execution has
        # no next open inside the requested window and is explicitly terminal.
        if t < T - 2:
            held_mask = np.abs(held) > 1e-12
            current_marks = np.asarray(sm.mark_prices, dtype=np.float64)
            next_prices = TradingStateMachine._round_price(
                np.asarray(panel.open[e + 1], dtype=np.float64),
                fric_cfg.tick_size)
            next_valid = np.isfinite(next_prices) & (next_prices > 0)
            # A suspension flag dominates any stale quote left in the row;
            # only a genuinely tradable/reopened bar may re-mark the book.
            next_valid &= ~suspended[e + 1]
            if np.any(held_mask & ~np.isfinite(current_marks)):
                bad = ",".join(str(panel.codes[i]) for i in np.flatnonzero(
                    held_mask & ~np.isfinite(current_marks))[:8])
                raise ValueError(
                    "missing authoritative mark for held position(s) on "
                    f"{int(panel.dates[t])}: {bad}")
            missing_next = held_mask & ~next_valid
            if missing_next.any():
                unsignalled = missing_next & ~suspended[e + 1]
                if unsignalled.any():
                    bad = ",".join(str(panel.codes[i]) for i in np.flatnonzero(
                        unsignalled)[:8])
                    raise ValueError(
                        "missing forward open without suspension evidence on "
                        f"{int(panel.dates[t])}: {bad}")
                suspension_carry_days += 1
                suspension_carry_positions += int(missing_next.sum())
            interval_ret = np.zeros(N, dtype=np.float64)
            marked = held_mask & next_valid
            interval_ret[marked] = next_prices[marked] / current_marks[marked] - 1.0
            gross = float(np.dot(held[held_mask], interval_ret[held_mask]))
        else:
            gross = 0.0
        cash_costs = (res.costs["stamp"] + res.costs["commission"]
                      + res.costs["slippage"] + res.costs["financing"]
                      + res.costs["borrow_fee"])
        cost_frac = cash_costs / max(float(res.equity_before), 1e-12)
        ls_gross[t] = gross
        ls_net[t] = gross - cost_frac
    ls_ms = (time.perf_counter() - t_ls) * 1000.0

    # ---- benchmark ------------------------------------------------------- #
    if panel.benchmark_ret is not None:
        # benchmark_ret[s] = close s -> close s+1; strategy row t spans
        # (open t+1 -> open t+2) => take row t+1 so both curves share the
        # same calendar window (documented close-to-close deviation).
        bench_daily = np.asarray(panel.benchmark_ret,
                                 dtype=np.float64)[1:T].copy()
        missing_benchmark = ~np.isfinite(bench_daily)
        if missing_benchmark[:-1].any():
            raise ValueError(
                "authoritative benchmark_ret has interior missing values; "
                "refusing a zero-return fallback")
        if missing_benchmark[-1]:
            # Only the final signal has no next benchmark close in-window.
            bench_daily[-1] = 0.0
        bench_note = str(panel.meta.get("benchmark", "panel.benchmark_ret"))
    else:
        raise ValueError(
            "authoritative benchmark_ret is required; refusing the "
            "equal-weight benchmark fallback")
    notes["benchmark"] = bench_note

    # ---- curves & metrics ------------------------------------------------- #
    group_curves = [_cumulative(group_daily[g]) for g in range(G)]
    ls_curve = _cumulative(ls_net)
    ls_curve_gross = _cumulative(ls_gross)
    benchmark_curve = _cumulative(bench_daily)
    underwater = _underwater_from_curve(ls_curve)
    metrics = M.summarize_metrics(
        ls_net, ic_series,
        normal_ic=normal_ic_series,
        cross_sectional_autocorrelation=cross_sectional_autocorrelation_series,
        n_trials=int(cfg.n_trials),
    )

    result = BacktestResult(
        dates=panel.dates, codes=panel.codes,
        group_curves=group_curves, ls_curve=ls_curve,
        benchmark_curve=benchmark_curve, ic_series=ic_series,
        ic_decay={int(k): float(v) for k, v in ic_decay.items()},
        underwater=underwater, turnover_series=turnover,
        friction_breakdown=sm.friction_breakdown(), metrics=metrics,
        acceleration_mode=mode, blocked_orders=sm.blocked_totals(),
        normal_ic_series=normal_ic_series,
        cross_sectional_autocorrelation_series=(
            cross_sectional_autocorrelation_series),
        ls_curve_gross=ls_curve_gross, group_daily=group_daily,
        ls_daily=ls_net, ls_daily_gross=ls_gross, benchmark_daily=bench_daily,
        group_membership=gid.astype(np.int8),
        formation_ms=(time.perf_counter() - t_start) * 1000.0,
        notes=notes,
        ledger=sm.audit_summary(),
        cash_curve=cash_curve, equity_curve=equity_curve,
        gross_exposure_series=gross_exposure,
        net_exposure_series=net_exposure,
        long_notional_series=long_notional,
        short_notional_series=short_notional,
    )
    result.notes.update({
        "matching": "next-day-open",
        "rank_ms": round(rank_ms, 2),
        "backend_ms": round(backend_ms, 2),
        "stats_ms": round(stats_ms, 2),
        "group_ms": round(group_ms, 2),
        "ls_ms": round(ls_ms, 2),
        "total_ms": round(result.formation_ms, 2),
        "eligible_ratio": round(float(eligible.mean()), 4),
        "group_counts_last": [int((gid[-1] == g).sum()) for g in range(G)],
        "trade_days": T - 1,
        "turnover_mean": round(float(turnover[:T - 1].mean()), 4),
        "cost_total": round(float(sum(sm.friction_breakdown().values())), 2),
        "ledger_rows": int(len(sm.ledger)),
        "final_cash": round(float(sm.cash), 6),
        "final_equity": round(float(sm.equity), 6),
        "suspension_mark_policy": "carry_last_valid_open_until_reopen",
        "suspension_carry_days": int(suspension_carry_days),
        "suspension_carry_positions": int(suspension_carry_positions),
    })
    return result
