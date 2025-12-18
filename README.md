# gjr-garch-x

[![PyPI version](https://badge.fury.io/py/gjr-garch-x.svg)](https://badge.fury.io/py/gjr-garch-x)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17917922.svg)](https://doi.org/10.5281/zenodo.17917922)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**GJR-GARCH models with exogenous regressors in the variance equation.**

A pure Python implementation of Glosten-Jagannathan-Runkle (1993) GARCH models that properly supports exogenous variables in the conditional variance equation—a feature missing from standard econometrics packages.

## Why This Package?

Standard GARCH packages (including Python's `arch`) don't natively support exogenous regressors in the variance equation. This matters for:

- **Event studies**: Testing whether specific events (regulatory announcements, infrastructure failures) affect volatility
- **Sentiment analysis**: Including sentiment indicators as volatility drivers
- **Regime-dependent volatility**: Adding dummy variables for different market conditions

This package implements the full GJR-GARCH-X specification:

```
σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I(ε_{t-1}<0) + β·σ²_{t-1} + Σδⱼ·x_{j,t}
```

Where `x_{j,t}` are your exogenous variables with coefficients `δⱼ` estimated via maximum likelihood.

## Installation

```bash
pip install gjr-garch-x
```

## Quick Start

```python
import pandas as pd
from gjr_garch_x import estimate_gjr_garch_x

# Your returns data (log returns × 100)
returns = pd.Series(...)

# Exogenous variables for variance equation
exog_vars = pd.DataFrame({
    'D_event': event_dummy,           # Event indicator
    'sentiment': sentiment_score,      # Continuous variable
}, index=returns.index)

# Estimate model
results = estimate_gjr_garch_x(returns, exog_vars)

# Results
print(f"Converged: {results.converged}")
print(f"AIC: {results.aic:.2f}")
print(f"Event effect: {results.event_effects['D_event']:.4f}")
print(f"Leverage effect (γ): {results.leverage_effect:.4f}")

# Full summary
print(results.summary())
```

## Features

- **Student-t innovations**: Captures fat tails in financial returns
- **GJR-GARCH leverage effect**: Asymmetric response to positive/negative shocks
- **Robust standard errors**: Numerical Hessian computation with proper inference
- **Stationarity constraints**: Enforced during optimization
- **Pandas integration**: Works directly with Series/DataFrame objects
- **No dependencies on `arch`**: Standalone implementation
- **Type hints**: Full type annotations for IDE support

## Model Specification

### Variance Equation (GJR-GARCH-X)

```
σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I(ε_{t-1}<0) + β·σ²_{t-1} + Σδⱼ·x_{j,t}
```

**Parameters:**
- `ω` (omega): Intercept, baseline variance level
- `α` (alpha): ARCH effect, response to recent squared shocks
- `γ` (gamma): Leverage effect, *additional* response to negative shocks
- `β` (beta): GARCH effect, persistence of conditional variance
- `δⱼ`: Coefficients on exogenous variables
- `ν` (nu): Degrees of freedom for Student-t distribution

**Leverage Effect Interpretation:**
- Positive shocks: volatility impact = `α`
- Negative shocks: volatility impact = `α + γ`
- If `γ > 0`: bad news increases volatility more than good news

### Stationarity Condition

```
α + β + |γ|/2 < 1
```

Enforced automatically during estimation.

## API Reference

### `estimate_gjr_garch_x(returns, exog_vars, method='SLSQP', verbose=False)`

Main estimation function.

**Parameters:**
- `returns`: `pd.Series` of log returns (recommend × 100 for numerical stability)
- `exog_vars`: `pd.DataFrame` of exogenous variables, aligned with returns index
- `method`: Optimization method (`'SLSQP'`, `'L-BFGS-B'`, `'trust-constr'`)
- `verbose`: Print estimation progress

**Returns:** `GJRGARCHXResults` object

### `GJRGARCHXResults`

Results container with attributes:
- `converged`: `bool` — Did optimization converge?
- `params`: `Dict[str, float]` — All parameter estimates
- `std_errors`: `Dict[str, float]` — Standard errors
- `pvalues`: `Dict[str, float]` — Two-sided p-values
- `log_likelihood`: `float`
- `aic`, `bic`: `float` — Information criteria
- `volatility`: `pd.Series` — Conditional standard deviation σ_t
- `residuals`: `pd.Series` — Demeaned residuals ε_t
- `exog_effects`: `Dict[str, float]` — All exogenous variable coefficients
- `event_effects`: `Dict[str, float]` — Event-type exogenous coefficients
- `sentiment_effects`: `Dict[str, float]` — Sentiment-type coefficients
- `leverage_effect`: `float` — γ parameter
- `iterations`: `int` — Optimizer iterations
- `n_obs`: `int` — Number of observations

## Example: Cryptocurrency Event Study

```python
import pandas as pd
from gjr_garch_x import estimate_gjr_garch_x

# Load BTC returns
btc = pd.read_csv('btc_returns.csv', index_col='date', parse_dates=True)
returns = btc['log_return'] * 100  # Convert to percentage

# Create event dummies
exog = pd.DataFrame(index=returns.index)
exog['D_infrastructure'] = 0
exog['D_regulatory'] = 0

# Mark infrastructure events (e.g., exchange hacks)
infra_dates = ['2022-11-11', '2022-05-09']  # FTX, Terra
for date in infra_dates:
    # 7-day event window
    mask = (exog.index >= pd.Timestamp(date) - pd.Timedelta(days=3)) & \
           (exog.index <= pd.Timestamp(date) + pd.Timedelta(days=3))
    exog.loc[mask, 'D_infrastructure'] = 1

# Mark regulatory events
reg_dates = ['2024-01-10', '2021-09-24']  # ETF approval, China ban
for date in reg_dates:
    mask = (exog.index >= pd.Timestamp(date) - pd.Timedelta(days=3)) & \
           (exog.index <= pd.Timestamp(date) + pd.Timedelta(days=3))
    exog.loc[mask, 'D_regulatory'] = 1

# Estimate
results = estimate_gjr_garch_x(returns, exog, verbose=True)

# Compare effects
print(f"Infrastructure effect: {results.event_effects['D_infrastructure']:.4f}")
print(f"Regulatory effect: {results.event_effects['D_regulatory']:.4f}")
print(f"Ratio: {results.event_effects['D_infrastructure'] / results.event_effects['D_regulatory']:.2f}x")
```

## Backwards Compatibility

For users migrating from TARCH naming conventions, aliases are provided:

```python
from gjr_garch_x import estimate_tarch_x, TARCHXResults, TARCHXEstimator
```

These are identical to the GJR-prefixed versions.

## Citation

If you use this package in academic work, please cite:

```bibtex
@software{farzulla2025gjrgarchx,
  author = {Farzulla, Murad},
  title = {gjr-garch-x: GJR-GARCH Models with Exogenous Variance Regressors},
  year = {2025},
  publisher = {PyPI},
  url = {https://github.com/studiofarzulla/gjr-garch-x}
}
```

For the research paper that motivated this implementation:

```bibtex
@techreport{farzulla2025infrastructure,
  author = {Farzulla, Murad},
  title = {Market Reaction Asymmetry: Infrastructure Disruption Dominance
           Over Regulatory Uncertainty in Cryptocurrency Markets},
  year = {2025},
  type = {Working Paper},
  doi = {10.2139/ssrn.5788082}
}
```

## References

- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779-1801.
- Engle, R. F., & Ng, V. K. (1993). Measuring and testing the impact of news on volatility. *Journal of Finance*, 48(5), 1749-1778.
- Bollerslev, T., & Wooldridge, J. M. (1992). Quasi-maximum likelihood estimation and inference in dynamic models with time-varying covariances. *Econometric Reviews*, 11(2), 143-172.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome. Please open an issue first to discuss proposed changes.

## Author

**Murad Farzulla**
MSc Finance Analytics, King's College London
[ORCID: 0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)
