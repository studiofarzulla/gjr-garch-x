# gjr-garch-x

[![PyPI version](https://badge.fury.io/py/gjr-garch-x.svg)](https://badge.fury.io/py/gjr-garch-x)
[![DOI](https://zenodo.org/badge/1119107069.svg)](https://doi.org/10.5281/zenodo.17988193)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**GJR-GARCH models with exogenous regressors in the variance equation.**

A pure Python implementation of Glosten-Jagannathan-Runkle (1993) GARCH models that properly supports exogenous variables in the conditional variance equation—a feature missing from standard econometrics packages. It is the estimation engine behind [robust-eventstudy](https://github.com/studiofarzulla/robust-eventstudy), a dependence-robust inference toolkit for event studies.

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
- **Robust standard errors**: Bollerslev-Wooldridge (1992) QMLE sandwich covariance by
  default (`cov_type="robust"`), with the classical inverse-Hessian estimator available
  via `cov_type="hessian"`
- **Stationarity constraints**: Enforced during optimization
- **Configurable coefficient caps**: `alpha_max` / `beta_max` (relaxed defaults so the
  bounds do not silently bind on high-volatility daily series such as crypto)
- **Pandas integration**: Works directly with Series/DataFrame objects, with array-like
  coercion and informative alignment errors
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

### Coefficient bounds

The individual ARCH/GARCH bounds are exposed as keyword arguments and default to loose
values so that they do not silently bind:

| Argument | Default | Note |
|----------|---------|------|
| `alpha_max` | `0.99` | Upper bound on `α` (was a hard-coded `0.30` before v0.2.0) |
| `beta_max` | `0.999` | Upper bound on `β` (was a hard-coded `0.95` before v0.2.0) |

The economically meaningful restriction is the stationarity constraint
`α + β + |γ|/2 < 1`, which is always enforced regardless of these caps. Tighten the caps
explicitly (e.g. `alpha_max=0.30`) if you want the historical behaviour.

## Standard Errors and Inference

Coefficients are estimated by quasi-maximum likelihood (QMLE) with Student-t innovations.
Two covariance estimators are available via `cov_type`:

- **`cov_type="robust"` (default)** — the Bollerslev-Wooldridge (1992) QMLE sandwich
  covariance

  ```
  V = H⁻¹ · OPG · H⁻¹
  ```

  where `H` is the observed information (Hessian of the negative log-likelihood) and
  `OPG = Σ_t sₜ sₜᵀ` is the outer product of the per-observation score contributions.
  These standard errors remain valid when the Student-t likelihood is misspecified (the
  usual QMLE robustness), which is the relevant case for heavy-tailed asset returns.

- **`cov_type="hessian"`** — the classical inverse observed-information covariance `H⁻¹`,
  valid only under correct specification.

The Hessian is symmetrised and checked for positive definiteness; if it is not positive
definite (a near-degenerate fit), the covariance falls back to the Moore-Penrose
pseudo-inverse and a `RuntimeWarning` is emitted rather than returning silent `NaN`s.

```python
robust = estimate_gjr_garch_x(returns, exog_vars)                    # BW sandwich (default)
classical = estimate_gjr_garch_x(returns, exog_vars, cov_type="hessian")
print(robust.cov_type, classical.cov_type)  # 'robust' 'hessian'
```

## API Reference

### `estimate_gjr_garch_x(returns, exog_vars=None, method='SLSQP', max_iter=1000, verbose=False, cov_type='robust', alpha_max=0.99, beta_max=0.999)`

Main estimation function.

**Parameters:**
- `returns`: returns series (recommend log returns × 100). Accepts a `pd.Series`
  (index preserved), a single-column DataFrame, or any 1-D array-like; NaNs are dropped
- `exog_vars`: `pd.DataFrame` (index must cover the returns index) or a bare array matching
  the number of returns observations; exogenous variables for the variance equation
- `method`: Optimization method (`'SLSQP'`, `'L-BFGS-B'`, `'trust-constr'`)
- `max_iter`: Maximum optimizer iterations
- `verbose`: Print estimation progress
- `cov_type`: `'robust'` (Bollerslev-Wooldridge sandwich, default) or `'hessian'`
- `alpha_max`, `beta_max`: Upper bounds on `α` and `β` (see *Coefficient bounds*)

**Returns:** `GJRGARCHXResults` object

### `GJRGARCHXResults`

Results container with attributes:
- `converged`: `bool` — Did optimization converge?
- `params`: `Dict[str, float]` — All parameter estimates
- `std_errors`: `Dict[str, float]` — Standard errors (per `cov_type`)
- `pvalues`: `Dict[str, float]` — Two-sided p-values
- `cov_type`: `str` — Covariance estimator used (`'robust'` or `'hessian'`)
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
  publisher = {Zenodo},
  doi = {10.5281/zenodo.17988193},
  url = {https://github.com/studiofarzulla/gjr-garch-x}
}
```

For the research paper that motivated this implementation (under review at
*Digital Finance*):

```bibtex
@techreport{farzulla2026multimoment,
  author = {Farzulla, Murad},
  title = {Do Cryptocurrency Markets Differentiate Infrastructure from
           Regulatory Shocks? A Multi-Moment Event Study with
           Dependence-Robust Inference},
  year = {2026},
  type = {Preprint},
  doi = {10.21203/rs.3.rs-8323026/v1}
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
