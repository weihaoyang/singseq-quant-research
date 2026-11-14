# Sources and research references

The Python modules in this repository are distributed under Apache-2.0.
Dependency packages use their respective licenses and are installed separately.

## Formula families

The Alpha158-style feature family refers to the publicly documented construction
in Microsoft Qlib, https://github.com/microsoft/qlib, distributed under MIT.
The numerical implementation in `factors_catalog/alpha158.py` uses the windows,
column names and behavior documented in that module.

Alpha101-style operators and selected formulas refer to Zura Kakushadze,
*101 Formulaic Alphas*, https://arxiv.org/abs/1601.00991.

Factor definitions in `evaluation/alpha_zoo.py` are implemented from formula
definitions. Related families include price momentum, reversal, volatility,
liquidity and volume-price relationships.

## Research models

The iTransformer, PatchTST and AlphaNet modules are local research implementations
of those model families. Architecture names identify research ideas. Their
performance depends on the input, configuration and validation design.

## Data

The deterministic data produced by `examples/quickstart.py` is generated for a
software demonstration. Market data used by researchers retains its provider's
terms and permissions.
