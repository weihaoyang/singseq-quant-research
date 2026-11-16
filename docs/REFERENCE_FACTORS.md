# 完整参考因子

`factors_catalog.r441_registry` 提供 241 个登记因子的完整快照。数值公式来自冻结的 r441 oracle，不按因子名称重新猜测；三个历史交互因子使用显式 `interaction-v1` 合同。

## Python 入口

```python
from factors_catalog.r441_registry import (
    RegistryFactorInputs,
    compute_registry_factor_snapshot,
)

snapshot = compute_registry_factor_snapshot(
    RegistryFactorInputs(
        prices=closes,
        turnover=amounts,
        opens=opens,
        highs=highs,
        lows=lows,
        benchmark_prices=benchmark_closes,
        stock_flows=stock_flows,
        fundamental_factors=fundamentals,
        enable_numba_kernels=False,
    )
)
```

返回字典只包含当前窗口真实可计算的因子。缺失侧输入不会被零填充。

## 数据适配器

`factors_catalog.r441_data.load_reference_factor_context` 面向 Python DB-API/DuckDB 连接，读取以下真实表或由成员直接派生的点对点上下文：

- `index_daily` 的 `000300.SH` 基准；
- `moneyflow` 股票资金流；
- 时点有效的 `industry_membership`；
- 行业成员等权收益、宽度、领导者和落后者；
- `limit_list_d` 情绪和炸板；
- `top_list` 事件方向、强度、置信度和证券命中；
- `income`、`balancesheet`、`cashflow` 及补充 Parquet。

补充财务文件按证券、报告期和最新有效公告日去重。基本面映射只使用当时已经披露的数据。

## 事件输入

| 因子 | 必需输入 |
| --- | --- |
| `event_surprise_score_5` | 方向、强度、置信度 |
| `event_recency_decay_30m` | 事件年龄、影响 |
| `event_consistency_20d` | 影响 |
| `event_sector_diffusion_5d` | 证券命中、行业命中 |
| `regulatory_risk_penalty_20d` | 监管事件 |
| `shareholder_alignment_60d` | 股东事件 |
| `policy_alignment_score_10d` | 政策事件、行业收益 |
| `crowding_event_conflict_5d` | 情绪代理、热词 |

通用 `top_list` 支持前两项中的惊喜度和一致性。没有对应专属源的因子保持不可用。

## 30 分钟因子

当前公开分钟源截止 `20260410`。六个 30 分钟因子在该日期前可计算；请求窗口跨过该日期时，覆盖状态为 `PARTIAL`，并保留请求区间、发布截止日和未覆盖起点。部分窗口不得作为完整窗口成绩发布。

## Windows 与 Linux

数组入口在 Linux 使用按证券分区的进程池；Windows 没有 `fork`，因此按相同分区确定性内联执行。两条路径的因子名称和数值列必须一致。
