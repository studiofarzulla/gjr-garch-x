"""
gjr-garch-x: GJR-GARCH models with exogenous variance regressors.

This module implements GJR-GARCH models (Glosten, Jagannathan & Runkle, 1993) that
properly support exogenous variables in the conditional variance equation—a feature
missing from standard econometrics packages.

Model Specification (GJR-GARCH-X):
    σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I(ε_{t-1}<0) + β·σ²_{t-1} + Σδⱼ·x_{j,t}

Where:
    ω (omega): Intercept, baseline variance level
    α (alpha): ARCH effect, response to recent squared shocks
    γ (gamma): Leverage effect, ADDITIONAL response to negative shocks
    β (beta): GARCH effect, persistence of conditional variance
    δⱼ: Coefficients on exogenous variables x_{j,t}
    ν (nu): Degrees of freedom for Student-t distribution

Leverage Effect Interpretation:
    - For positive shocks (ε_{t-1} > 0): volatility impact = α
    - For negative shocks (ε_{t-1} < 0): volatility impact = α + γ
    - If γ > 0: negative returns increase volatility MORE than positive returns

References:
    Glosten, Jagannathan & Runkle (1993). On the relation between expected
        value and volatility of nominal excess return on stocks.
    Engle & Ng (1993). Measuring and testing the impact of news on volatility.
    Bollerslev & Wooldridge (1992). Quasi-maximum likelihood estimation and
        inference in dynamic models with time-varying covariances.

Author: Murad Farzulla <murad@farzulla.org>
License: MIT
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func
from scipy.stats import t as student_t

__version__ = "0.1.0"
__author__ = "Murad Farzulla"
__email__ = "murad@farzulla.org"

__all__ = [
    "estimate_gjr_garch_x",
    "GJRGARCHXResults",
    "GJRGARCHXEstimator",
    # Backwards compatibility aliases
    "estimate_tarch_x",
    "TARCHXResults",
    "TARCHXEstimator",
]


@dataclass
class GJRGARCHXResults:
    """
    Container for GJR-GARCH-X estimation results.

    Attributes
    ----------
    converged : bool
        Whether optimization converged successfully.
    params : Dict[str, float]
        All parameter estimates including GARCH and exogenous coefficients.
    std_errors : Dict[str, float]
        Standard errors computed from numerical Hessian.
    pvalues : Dict[str, float]
        Two-sided p-values for parameter significance.
    log_likelihood : float
        Maximized log-likelihood value.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    volatility : pd.Series
        Conditional standard deviation series σ_t.
    residuals : pd.Series
        Demeaned residuals ε_t.
    exog_effects : Dict[str, float]
        Coefficients on all exogenous variables.
    event_effects : Dict[str, float]
        Coefficients on event-type exogenous variables (detected by keywords).
    sentiment_effects : Dict[str, float]
        Coefficients on sentiment-type variables (detected by keywords).
    leverage_effect : float
        The γ parameter capturing asymmetric volatility response.
    iterations : int
        Number of optimizer iterations.
    n_obs : int
        Number of observations used in estimation.
    """

    converged: bool
    params: Dict[str, float]
    std_errors: Dict[str, float]
    pvalues: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    volatility: pd.Series
    residuals: pd.Series
    exog_effects: Dict[str, float]
    event_effects: Dict[str, float]
    sentiment_effects: Dict[str, float]
    leverage_effect: float
    iterations: int
    n_obs: int = 0

    def __post_init__(self):
        """Set n_obs from volatility length if not provided."""
        if self.n_obs == 0 and len(self.volatility) > 0:
            object.__setattr__(self, "n_obs", len(self.volatility))

    def summary(self) -> str:
        """
        Generate formatted summary of estimation results.

        Returns
        -------
        str
            Multi-line summary string suitable for printing.
        """
        lines = [
            "",
            "=" * 65,
            "GJR-GARCH-X Model Results",
            "=" * 65,
            f"Converged:       {self.converged}",
            f"Log-likelihood:  {self.log_likelihood:.4f}",
            f"AIC:             {self.aic:.4f}",
            f"BIC:             {self.bic:.4f}",
            f"Observations:    {self.n_obs}",
            "",
            "Variance Equation Parameters:",
            "-" * 45,
            f"{'Parameter':<12} {'Coef':>12} {'Std.Err':>12} {'P-value':>10}",
            "-" * 45,
        ]

        # Core GARCH parameters
        core_params = ["omega", "alpha", "gamma", "beta", "nu"]
        for param in core_params:
            if param in self.params:
                coef = self.params[param]
                se = self.std_errors.get(param, np.nan)
                pval = self.pvalues.get(param, np.nan)
                sig = _significance_stars(pval)
                lines.append(
                    f"{param:<12} {coef:>12.6f} {se:>12.6f} {pval:>10.4f}{sig}"
                )

        # Persistence metrics
        alpha = self.params.get("alpha", 0)
        beta = self.params.get("beta", 0)
        gamma = self.params.get("gamma", 0)
        persistence = alpha + beta + abs(gamma) / 2

        lines.append("")
        lines.append(f"Persistence (α + β + |γ|/2): {persistence:.4f}")

        if 0 < persistence < 1:
            half_life = -np.log(0.5) / np.log(persistence)
            lines.append(f"Half-life of shocks:         {half_life:.1f} periods")

        # Unconditional variance (if stationary)
        omega = self.params.get("omega", 0)
        if persistence < 1 and omega > 0:
            uncond_var = omega / (1 - persistence)
            uncond_vol = np.sqrt(uncond_var)
            lines.append(f"Unconditional volatility:    {uncond_vol:.4f}")

        # Exogenous effects
        if self.event_effects:
            lines.append("")
            lines.append("Exogenous Effects (Event-type):")
            lines.append("-" * 45)
            for name, coef in self.event_effects.items():
                se = self.std_errors.get(name, np.nan)
                pval = self.pvalues.get(name, np.nan)
                sig = _significance_stars(pval)
                lines.append(
                    f"{name:<25} {coef:>+10.6f} ({se:.4f}) [{pval:.4f}]{sig}"
                )

        if self.sentiment_effects:
            lines.append("")
            lines.append("Exogenous Effects (Sentiment-type):")
            lines.append("-" * 45)
            for name, coef in self.sentiment_effects.items():
                se = self.std_errors.get(name, np.nan)
                pval = self.pvalues.get(name, np.nan)
                sig = _significance_stars(pval)
                lines.append(
                    f"{name:<25} {coef:>+10.6f} ({se:.4f}) [{pval:.4f}]{sig}"
                )

        lines.append("")
        lines.append("=" * 65)
        lines.append("Signif. codes: *** p<0.01, ** p<0.05, * p<0.10")

        return "\n".join(lines)

    def __repr__(self) -> str:
        n_exog = len(self.event_effects) + len(self.sentiment_effects)
        return (
            f"GJRGARCHXResults(converged={self.converged}, "
            f"aic={self.aic:.2f}, bic={self.bic:.2f}, "
            f"n_obs={self.n_obs}, n_exog={n_exog})"
        )


# Backwards compatibility alias
TARCHXResults = GJRGARCHXResults


def _significance_stars(pval: float) -> str:
    """Return significance stars for p-value."""
    if np.isnan(pval):
        return ""
    if pval < 0.01:
        return " ***"
    if pval < 0.05:
        return " **"
    if pval < 0.10:
        return " *"
    return ""


class GJRGARCHXEstimator:
    """
    GJR-GARCH-X model estimator with exogenous variance regressors.

    Implements Student-t GJR-GARCH with exogenous variables in the variance
    equation via quasi-maximum likelihood estimation.

    Parameters
    ----------
    returns : pd.Series
        Series of returns (recommend log returns × 100 for numerical stability).
    exog_vars : pd.DataFrame, optional
        DataFrame of exogenous variables for the variance equation.
        Index must align with returns.

    Examples
    --------
    >>> estimator = GJRGARCHXEstimator(returns, exog_vars)
    >>> results = estimator.estimate()
    >>> print(results.summary())
    """

    def __init__(
        self,
        returns: pd.Series,
        exog_vars: Optional[pd.DataFrame] = None,
    ):
        self.returns = returns.dropna()

        if exog_vars is not None:
            self.exog_vars = exog_vars.loc[self.returns.index].fillna(0)
            self.has_exog = True
            self.n_exog = self.exog_vars.shape[1]
            self.exog_names = list(self.exog_vars.columns)
        else:
            self.exog_vars = None
            self.has_exog = False
            self.n_exog = 0
            self.exog_names = []

        self.n_obs = len(self.returns)
        self.param_names = ["omega", "alpha", "gamma", "beta", "nu"] + self.exog_names
        self.n_params = 5 + self.n_exog

    def _unpack_params(self, params: np.ndarray) -> Dict[str, float]:
        """Unpack parameter vector into named dictionary."""
        param_dict = {
            "omega": params[0],
            "alpha": params[1],
            "gamma": params[2],
            "beta": params[3],
            "nu": params[4],
        }
        for i, name in enumerate(self.exog_names):
            param_dict[name] = params[5 + i]
        return param_dict

    def _variance_recursion(
        self, params: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute conditional variance via GARCH recursion.

        Returns
        -------
        variance : np.ndarray
            Conditional variance series σ²_t.
        residuals : np.ndarray
            Demeaned residuals ε_t.
        """
        param_dict = self._unpack_params(params)
        omega = param_dict["omega"]
        alpha = param_dict["alpha"]
        gamma = param_dict["gamma"]
        beta = param_dict["beta"]

        variance = np.zeros(self.n_obs)
        mean_return = self.returns.mean()
        residuals = (self.returns - mean_return).values

        # Initialize with unconditional variance estimate
        variance[0] = np.var(self.returns)

        for t in range(1, self.n_obs):
            eps_sq_prev = residuals[t - 1] ** 2
            leverage_term = gamma * eps_sq_prev * (residuals[t - 1] < 0)

            variance[t] = (
                omega
                + alpha * eps_sq_prev
                + leverage_term
                + beta * variance[t - 1]
            )

            # Add exogenous effects
            if self.has_exog:
                for i, exog_name in enumerate(self.exog_names):
                    delta = param_dict[exog_name]
                    exog_value = self.exog_vars.iloc[t, i]
                    variance[t] += delta * exog_value

            # Ensure positive variance
            variance[t] = max(variance[t], 1e-8)

        return variance, residuals

    def _log_likelihood(self, params: np.ndarray) -> float:
        """Compute negative log-likelihood for Student-t GJR-GARCH-X."""
        try:
            param_dict = self._unpack_params(params)
            nu = param_dict["nu"]

            variance, residuals = self._variance_recursion(params)
            std_residuals = residuals / np.sqrt(variance)

            # Student-t log-likelihood
            log_lik = 0.0
            for t in range(self.n_obs):
                log_gamma_term = (
                    np.log(gamma_func((nu + 1) / 2))
                    - np.log(gamma_func(nu / 2))
                    - 0.5 * np.log(np.pi * (nu - 2))
                )
                log_var_term = -0.5 * np.log(variance[t])
                density_term = -((nu + 1) / 2) * np.log(
                    1 + std_residuals[t] ** 2 / (nu - 2)
                )
                log_lik += log_gamma_term + log_var_term + density_term

            return -log_lik

        except (ValueError, OverflowError, RuntimeWarning):
            return 1e8

    def _parameter_constraints(self) -> List[Dict]:
        """Define optimization constraints including stationarity."""
        return [
            {"type": "ineq", "fun": lambda x: x[0] - 1e-8},  # omega > 0
            {"type": "ineq", "fun": lambda x: x[1] - 1e-8},  # alpha > 0
            {"type": "ineq", "fun": lambda x: x[3] - 1e-8},  # beta > 0
            {"type": "ineq", "fun": lambda x: x[4] - 2.1},  # nu > 2
            {"type": "ineq", "fun": lambda x: 50 - x[4]},  # nu < 50
            # Stationarity: α + β + |γ|/2 < 1
            {
                "type": "ineq",
                "fun": lambda x: 0.999 - (x[1] + x[3] + abs(x[2]) / 2),
            },
        ]

    def _get_starting_values(self) -> np.ndarray:
        """Generate reasonable starting values."""
        sample_var = np.var(self.returns)
        start_vals = np.array([
            sample_var * 0.1,  # omega
            0.05,  # alpha
            0.05,  # gamma (leverage)
            0.85,  # beta
            5.0,  # nu
        ])
        if self.has_exog:
            start_vals = np.append(start_vals, np.zeros(self.n_exog))
        return start_vals

    def estimate(
        self,
        method: str = "SLSQP",
        max_iter: int = 1000,
        verbose: bool = False,
    ) -> GJRGARCHXResults:
        """
        Estimate GJR-GARCH-X model via maximum likelihood.

        Parameters
        ----------
        method : str, default "SLSQP"
            Optimization method. Options: "SLSQP", "L-BFGS-B", "trust-constr".
        max_iter : int, default 1000
            Maximum number of optimizer iterations.
        verbose : bool, default False
            Print estimation progress.

        Returns
        -------
        GJRGARCHXResults
            Estimation results container.
        """
        if verbose:
            print(f"Estimating GJR-GARCH-X with {self.n_exog} exogenous variables...")

        start_vals = self._get_starting_values()

        bounds = [
            (1e-8, None),  # omega > 0
            (1e-8, 0.3),  # alpha
            (-0.5, 0.5),  # gamma (leverage)
            (1e-8, 0.95),  # beta
            (2.1, 50),  # nu
        ]
        # Exogenous coefficients are unbounded
        for _ in range(self.n_exog):
            bounds.append((None, None))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                result = minimize(
                    fun=self._log_likelihood,
                    x0=start_vals,
                    method=method,
                    bounds=bounds,
                    constraints=self._parameter_constraints(),
                    options={"maxiter": max_iter, "disp": False},
                )

            converged = result.success and result.fun < 1e6
            optimal_params = result.x
            param_dict = self._unpack_params(optimal_params)

            variance, residuals = self._variance_recursion(optimal_params)
            volatility = pd.Series(np.sqrt(variance), index=self.returns.index)
            residuals_series = pd.Series(residuals, index=self.returns.index)

            std_errors, pvalues = self._compute_standard_errors(optimal_params)

            log_lik = -result.fun
            aic = 2 * self.n_params - 2 * log_lik
            bic = np.log(self.n_obs) * self.n_params - 2 * log_lik

            # Classify exogenous effects
            exog_effects = {}
            event_effects = {}
            sentiment_effects = {}

            sentiment_keywords = {"sentiment", "gdelt", "tone", "mood", "fear", "greed"}
            event_keywords = {"event", "dummy", "d_", "infra", "reg", "shock"}

            for name in self.exog_names:
                name_lower = name.lower()
                exog_effects[name] = param_dict[name]

                if any(kw in name_lower for kw in sentiment_keywords):
                    sentiment_effects[name] = param_dict[name]
                else:
                    event_effects[name] = param_dict[name]

            if verbose:
                status = "OK" if converged else "WARNING: Did not converge"
                print(f"  [{status}] Iterations: {result.nit}")
                print(f"  Log-likelihood: {log_lik:.2f}")
                print(f"  AIC: {aic:.2f}, BIC: {bic:.2f}")

            return GJRGARCHXResults(
                converged=converged,
                params=param_dict,
                std_errors=std_errors,
                pvalues=pvalues,
                log_likelihood=log_lik,
                aic=aic,
                bic=bic,
                volatility=volatility,
                residuals=residuals_series,
                exog_effects=exog_effects,
                event_effects=event_effects,
                sentiment_effects=sentiment_effects,
                leverage_effect=param_dict["gamma"],
                iterations=result.nit,
                n_obs=self.n_obs,
            )

        except Exception as e:
            if verbose:
                print(f"  [FAIL] Estimation failed: {e}")

            return GJRGARCHXResults(
                converged=False,
                params={},
                std_errors={},
                pvalues={},
                log_likelihood=np.nan,
                aic=np.nan,
                bic=np.nan,
                volatility=pd.Series(dtype=float),
                residuals=pd.Series(dtype=float),
                exog_effects={},
                event_effects={},
                sentiment_effects={},
                leverage_effect=np.nan,
                iterations=0,
                n_obs=0,
            )

    def _compute_standard_errors(
        self, params: np.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute standard errors via numerical Hessian."""
        try:
            hessian = self._numerical_hessian(params)
            cov_matrix = np.linalg.inv(hessian)
            std_errs = np.sqrt(np.diag(cov_matrix))

            dof = self.n_obs - self.n_params
            if dof <= 0:
                return (
                    {name: np.nan for name in self.param_names},
                    {name: np.nan for name in self.param_names},
                )

            t_stats = params / std_errs
            pvals = 2 * (1 - student_t.cdf(np.abs(t_stats), dof))

            return (
                dict(zip(self.param_names, std_errs)),
                dict(zip(self.param_names, pvals)),
            )

        except (np.linalg.LinAlgError, ValueError):
            return (
                {name: np.nan for name in self.param_names},
                {name: np.nan for name in self.param_names},
            )

    def _numerical_hessian(self, params: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """Compute numerical Hessian via central differences."""
        n = len(params)
        hessian = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    params_plus = params.copy()
                    params_minus = params.copy()
                    params_plus[i] += h
                    params_minus[i] -= h

                    f_plus = self._log_likelihood(params_plus)
                    f_minus = self._log_likelihood(params_minus)
                    f_center = self._log_likelihood(params)

                    hessian[i, j] = (f_plus - 2 * f_center + f_minus) / (h**2)
                else:
                    params_pp = params.copy()
                    params_pm = params.copy()
                    params_mp = params.copy()
                    params_mm = params.copy()

                    params_pp[i] += h
                    params_pp[j] += h
                    params_pm[i] += h
                    params_pm[j] -= h
                    params_mp[i] -= h
                    params_mp[j] += h
                    params_mm[i] -= h
                    params_mm[j] -= h

                    f_pp = self._log_likelihood(params_pp)
                    f_pm = self._log_likelihood(params_pm)
                    f_mp = self._log_likelihood(params_mp)
                    f_mm = self._log_likelihood(params_mm)

                    hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h**2)

        return hessian


# Backwards compatibility alias
TARCHXEstimator = GJRGARCHXEstimator


def estimate_gjr_garch_x(
    returns: pd.Series,
    exog_vars: Optional[pd.DataFrame] = None,
    method: str = "SLSQP",
    max_iter: int = 1000,
    verbose: bool = False,
) -> GJRGARCHXResults:
    """
    Estimate GJR-GARCH-X model with exogenous variance regressors.

    This is the main entry point for the package. Estimates a GJR-GARCH model
    with Student-t innovations and exogenous variables in the variance equation.

    Parameters
    ----------
    returns : pd.Series
        Series of returns. Recommend log returns × 100 for numerical stability.
    exog_vars : pd.DataFrame, optional
        DataFrame of exogenous variables for the variance equation.
        Index must align with returns. Columns become variance regressors.
    method : str, default "SLSQP"
        Optimization method. Recommended: "SLSQP" or "L-BFGS-B".
    max_iter : int, default 1000
        Maximum optimizer iterations.
    verbose : bool, default False
        Print estimation progress to stdout.

    Returns
    -------
    GJRGARCHXResults
        Estimation results including parameters, standard errors, p-values,
        conditional volatility series, and information criteria.

    Examples
    --------
    Basic usage without exogenous variables:

    >>> results = estimate_gjr_garch_x(returns)
    >>> print(f"Persistence: {results.params['alpha'] + results.params['beta']:.3f}")

    With event dummies:

    >>> exog = pd.DataFrame({'D_event': event_dummy}, index=returns.index)
    >>> results = estimate_gjr_garch_x(returns, exog)
    >>> print(f"Event effect: {results.event_effects['D_event']:.4f}")

    Notes
    -----
    The model specification is:

        σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I(ε_{t-1}<0) + β·σ²_{t-1} + Σδⱼ·x_{j,t}

    Stationarity (α + β + |γ|/2 < 1) is enforced during estimation.

    References
    ----------
    Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation
    between the expected value and the volatility of the nominal excess return
    on stocks. Journal of Finance, 48(5), 1779-1801.
    """
    estimator = GJRGARCHXEstimator(returns, exog_vars)
    return estimator.estimate(method=method, max_iter=max_iter, verbose=verbose)


# Backwards compatibility alias
estimate_tarch_x = estimate_gjr_garch_x
