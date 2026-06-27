# Changelog

All notable changes to `gjr-garch-x` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-27

### Added

- **Genuine robust standard errors.** A `cov_type` argument on
  `estimate_gjr_garch_x` / `GJRGARCHXEstimator.estimate`. The default
  `cov_type="robust"` now computes the Bollerslev-Wooldridge (1992) QMLE sandwich
  covariance `H⁻¹ · OPG · H⁻¹` (inverse Hessian × outer product of per-observation
  gradients × inverse Hessian). The previous inverse-Hessian estimator remains
  available as `cov_type="hessian"`. The `cov_type` used is recorded on
  `GJRGARCHXResults.cov_type` and shown in `summary()`.
- **Configurable coefficient caps.** `alpha_max` (default `0.99`) and `beta_max`
  (default `0.999`) keyword arguments. These replace the previous undocumented hard
  caps (`α ≤ 0.30`, `β ≤ 0.95`), which silently bound on high-volatility daily
  series such as cryptocurrency returns. The stationarity constraint
  `α + β + |γ|/2 < 1` is still always enforced.
- **Input validation.** Array-like `returns` (lists, numpy arrays, single-column
  DataFrames) are coerced to a `pd.Series`; misaligned or wrong-length `exog_vars`
  now raise informative `ValueError`s instead of obscure `KeyError`s.
- `CHANGELOG.md` (fixes the previously dead `[project.urls] Changelog` link).
- Tests for the robust covariance, the `hessian` covariance, the coefficient caps,
  and the input-validation paths (18 → 34 tests).
- **Near-unit-root half-life display (issue #8).** `summary()` now reports
  `∞ (near unit root)` for the half-life of shocks when persistence
  `α + β + |γ|/2 ≥ 0.9999`, instead of printing a misleadingly large finite value.
  Merged from the parallel `origin/master` issue-fix branch.

### Changed

- **Numerical guarding of the covariance.** The Hessian is symmetrised and checked
  for positive definiteness; a non-positive-definite or singular Hessian falls back
  to the Moore-Penrose pseudo-inverse with a `RuntimeWarning`, and the standard-error
  `sqrt` is guarded against negative variances (no more silent `invalid value`
  warnings).
- The per-observation Student-t log-likelihood is vectorised and the constant
  log-gamma normalising term is hoisted out of the per-`t` loop (the term depends on
  `ν` only). `scipy.special.gammaln` replaces `log(gamma(·))` for numerical
  stability. Estimation is materially faster; coefficient estimates are unchanged to
  numerical tolerance.
- README updated so the "robust standard errors / Bollerslev-Wooldridge" claim is now
  accurate, with a dedicated *Standard Errors and Inference* section and documented
  coefficient caps.

### Quality

- `requires-python` raised to `>=3.10`; classifiers, `black`, `ruff`, and `mypy`
  targets aligned to Python 3.10+.
- Legacy `typing.Dict` / `List` / `Optional` / `Tuple` replaced with builtin generics
  and `X | None`. `ruff check` and `mypy` (strict `disallow_untyped_defs`) pass clean.
- Reconciled with the parallel `origin/master` issue-fix branch (PRs #4/#6, issues
  #5/#7/#8). The documentation (#5, `max_iter` in the API reference), Hessian-warning
  (#7, the negative-variance `sqrt` guard), and mypy (#6) fixes from that branch are
  superseded by this release's larger rewrite; the unique near-unit-root half-life
  display (#8) is merged in (see *Added*). `ISSUES.md` from that branch is retained
  for provenance.

## [0.1.0] - 2025-12-19

### Added

- Initial release: Student-t GJR-GARCH-X estimator with exogenous regressors in the
  conditional variance equation, inverse-Hessian standard errors, stationarity
  constraints, information criteria, and a results summary. TARCH-X backwards-
  compatibility aliases.

[0.2.0]: https://github.com/studiofarzulla/gjr-garch-x/releases/tag/v0.2.0
[0.1.0]: https://github.com/studiofarzulla/gjr-garch-x/releases/tag/v0.1.0
