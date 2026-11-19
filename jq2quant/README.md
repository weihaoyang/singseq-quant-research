# jq2quant

**jq2quant** 是一个完全解耦的开源量化策略翻译器与迁移适配器（JoinQuant-to-Native Strategy Transpiler）。

它将聚宽（JoinQuant）平台特有的过程式函数与全局变量体系，一键无损翻译为现代、类型安全、纯面向对象（OOP）的量化策略代码，并可直接在本地零泄密量化研究套件或工业级事件驱动引擎上运行。

---

## 核心设计理念：彻底解耦

* **核心引擎不受污染**：现代量化框架（如 `quant_sdk`、Nautilus Trader 等）拥有自己纯净的事件驱动抽象与对象生命周期，不应硬编码或兼容任何第三方专有平台的历史包袱。
* **独立的迁移工具**：作为独立的开源生态组件，`jq2quant` 负责聚宽 AST 语义解析、模式替换与运行时桥接，随时可随聚宽语法更新独立迭代。

---

## 代码转换示例

### 原始聚宽代码 (`dual_ma_jq.py`)
```python
def initialize(context):
    set_benchmark('000300.SH')
    g.stock = '000001.SZ'
    run_daily(rebalance, time='open')

def rebalance(context):
    hist = attribute_history(g.stock, 25, fields=['close'])
    ma5 = hist['close'][-5:].mean()
    ma20 = hist['close'][-20:].mean()
    if ma5 > ma20:
        order_target_percent(g.stock, 0.95)
    elif ma5 < ma20:
        order_target(g.stock, 0)
```

### jq2quant 翻译后的现代策略代码 (`dual_ma_modern.py`)
```python
"""Transpiled from JoinQuant strategy by jq2quant (https://github.com/weihaoyang/jq2quant)."""
from __future__ import annotations
import quant_sdk as ts


class TranslatedStrategy(ts.Strategy):
    def on_init(self, context: ts.Context) -> None:
        context.set_benchmark('000300.SH')
        context.g.stock = '000001.SZ'
        ts.run_daily(self.rebalance, time='open')

    def rebalance(self, context: ts.Context):
        hist = ts.attribute_history(context.g.stock, 25, fields=['close'])
        ma5 = hist['close'][-5:].mean()
        ma20 = hist['close'][-20:].mean()
        if ma5 > ma20:
            self.order_target_percent(context.g.stock, 0.95)
        elif ma5 < ma20:
            self.order_target(context.g.stock, 0)


if __name__ == "__main__":
    result = ts.run_strategy(
        strategy=TranslatedStrategy(),
        start_date="20230101",
        end_date="20231231",
        initial_cash=1_000_000.0,
    )
    result.summary()
```

---

## CLI 命令行用法

```bash
# 1. 翻译策略文件
jq2quant transpile my_jq_strategy.py -o converted_strategy.py

# 2. 一键直接运行聚宽策略文件（运行时解耦适配）
jq2quant run my_jq_strategy.py --start 20230101 --end 20231231 --cash 1000000
```

---

## Python API 用法

```python
import jq2quant

# 方式一：代码字符串转换
modern_code = jq2quant.transpile(jq_code_str)

# 方式二：直接在内存中执行聚宽策略
result = jq2quant.run_jq_strategy(
    "my_jq_strategy.py",
    start_date="20230101",
    end_date="20231231",
    initial_cash=1_000_000,
)
result.summary()
```
