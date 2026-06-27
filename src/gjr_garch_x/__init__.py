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

Inference:
    Coefficients are estimated by quasi-maximum likelihood (QMLE) with Student-t
    innovations. Two covariance estimators are provided:

    - ``cov_type="robust"`` (default): the Bollerslev-Wooldridge (1992) QMLE
      sandwich covariance H⁻¹ · OPG · H⁻¹, where H is the observed information
      (Hessian of the negative log-likelihood) and OPG is the outer product of
      the per-observation score contributions. These standard errors remain valid
      under distributional misspecification of the innovations.
    - ``cov_type="hessian"``: the classical inverse-Hessian (observed-information)
      covariance, valid only when the likelihood is correctly specified.

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

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t as student_t

__version__ = "0.2.0"
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
    params : dict[str, float]
        All parameter estimates including GARCH and exogenous coefficients.
    std_errors : dict[str, float]
        Standard errors for each parameter (see ``cov_type``).
    pvalues : dict[str, float]
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
    exog_effects : dict[str, float]
        Coefficients on all exogenous variables.
    event_effects : dict[str, float]
        Coefficients on event-type exogenous variables (detected by keywords).
    sentiment_effects : dict[str, float]
        Coefficients on sentiment-type variables (detected by keywords).
    leverage_effect : float
        The γ parameter capturing asymmetric volatility response.
    iterations : int
        Number of optimizer iterations.
    n_obs : int
        Number of observations used in estimation.
    cov_type : str
        Covariance estimator used for ``std_errors``: ``"robust"``
        (Bollerslev-Wooldridge QMLE sandwich) or ``"hessian"``
        (inverse observed information).
    """

    converged: bool
    params: dict[str, float]
    std_errors: dict[str, float]
    pvalues: dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    volatility: pd.Series
    residuals: pd.Series
    exog_effects: dict[str, float]
    event_effects: dict[str, float]
    sentiment_effects: dict[str, float]
    leverage_effect: float
    iterations: int
    n_obs: int = 0
    cov_type: str = "robust"

    def __post_init__(self) -> None:
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
        se_label = {
            "robust": "robust (Bollerslev-Wooldridge QMLE sandwich)",
            "hessian": "inverse Hessian (observed information)",
        }.get(self.cov_type, self.cov_type)

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
            f"Std. errors:     {se_label}",
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
                lines.append(f"{name:<25} {coef:>+10.6f} ({se:.4f}) [{pval:.4f}]{sig}")

        if self.sentiment_effects:
            lines.append("")
            lines.append("Exogenous Effects (Sentiment-type):")
            lines.append("-" * 45)
            for name, coef in self.sentiment_effects.items():
                se = self.std_errors.get(name, np.nan)
                pval = self.pvalues.get(name, np.nan)
                sig = _significance_stars(pval)
                lines.append(f"{name:<25} {coef:>+10.6f} ({se:.4f}) [{pval:.4f}]{sig}")

        lines.append("")
        lines.append("=" * 65)
        lines.append("Signif. codes: *** p<0.01, ** p<0.05, * p<0.10")

        return "\n".join(lines)

    def __repr__(self) -> str:
        n_exog = len(self.event_effects) + len(self.sentiment_effects)
        return (
            f"GJRGARCHXResults(converged={self.converged}, "
            f"aic={self.aic:.2f}, bic={self.bic:.2f}, "
            f"n_obs={self.n_obs}, n_exog={n_exog}, cov_type={self.cov_type!r})"
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


def _as_series(returns: object) -> pd.Series:
    """Coerce array-like returns into a float pd.Series with informative errors."""
    if isinstance(returns, pd.Series):
        series = returns.astype(float)
    elif isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            raise ValueError(
                "returns must be one-dimensional; got a DataFrame with "
                f"{returns.shape[1]} columns. Pass a single Series/column."
            )
        series = returns.iloc[:, 0].astype(float)
    else:
        arr = np.asarray(returns, dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                f"returns must be one-dimensional; got array with shape {arr.shape}."
            )
        series = pd.Series(arr)
    return series


class GJRGARCHXEstimator:
    """
    GJR-GARCH-X model estimator with exogenous variance regressors.

    Implements Student-t GJR-GARCH with exogenous variables in the variance
    equation via quasi-maximum likelihood estimation.

    Parameters
    ----------
    returns : array-like
        Returns series (recommend log returns × 100 for numerical stability).
        Accepts a ``pd.Series`` (index preserved), a single-column DataFrame, or
        any 1-D array-like (a default integer index is assigned). NaNs are dropped.
    exog_vars : pd.DataFrame or array-like, optional
        Exogenous variables for the variance equation. A DataFrame index must
        cover the returns index; a bare array must match the number of returns
        observations (before NaN-dropping) and is assigned generic column names.

    Examples
    --------
    >>> estimator = GJRGARCHXEstimator(returns, exog_vars)
    >>> results = estimator.estimate()
    >>> print(results.summary())
    """

    def __init__(
        self,
        returns: pd.Series,
        exog_vars: pd.DataFrame | None = None,
    ):
        returns = _as_series(returns)
        raw_index = returns.index
        self.returns: pd.Series = returns.dropna()

        if len(self.returns) == 0:
            raise ValueError("returns contains no non-NaN observations.")

        self.exog_vars: pd.DataFrame | None
        if exog_vars is not None:
            exog_df = self._coerce_exog(exog_vars, raw_index)
            missing = self.returns.index.difference(exog_df.index)
            if len(missing) > 0:
                raise ValueError(
                    "exog_vars index does not cover all returns observations: "
                    f"{len(missing)} returns timestamps are missing from exog_vars "
                    f"(e.g. {list(missing[:3])}). Align exog_vars to the returns index."
                )
            self.exog_vars = exog_df.loc[self.returns.index].fillna(0.0)
            self.has_exog = True
            self.n_exog = self.exog_vars.shape[1]
            self.exog_names = list(self.exog_vars.columns)
            self._exog_array = self.exog_vars.to_numpy(dtype=float)
        else:
            self.exog_vars = None
            self.has_exog = False
            self.n_exog = 0
            self.exog_names = []
            self._exog_array = np.empty((len(self.returns), 0), dtype=float)

        self.n_obs = len(self.returns)
        self.param_names = ["omega", "alpha", "gamma", "beta", "nu"] + self.exog_names
        self.n_params = 5 + self.n_exog

    @staticmethod
    def _coerce_exog(exog_vars: object, raw_index: pd.Index) -> pd.DataFrame:
        """Coerce exogenous regressors into an index-aligned float DataFrame."""
        if isinstance(exog_vars, pd.DataFrame):
            df: pd.DataFrame = exog_vars.astype(float)
            return df
        if isinstance(exog_vars, pd.Series):
            frame: pd.DataFrame = exog_vars.astype(float).to_frame()
            return frame

        arr = np.asarray(exog_vars, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(
                f"exog_vars must be 1-D or 2-D; got array with shape {arr.shape}."
            )
        if arr.shape[0] != len(raw_index):
            raise ValueError(
                f"exog_vars has {arr.shape[0]} rows but returns has {len(raw_index)} "
                "observations; row counts must match for a bare array. Pass a "
                "DataFrame with an aligned index if they differ."
            )
        cols = [f"x{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, index=raw_index, columns=cols)

    def _unpack_params(self, params: np.ndarray) -> dict[str, float]:
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

    def _variance_recursion(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute conditional variance via GARCH recursion.

        Returns
        -------
        variance : np.ndarray
            Conditional variance series σ²_t.
        residuals : np.ndarray
            Demeaned residuals ε_t.
        """
        omega = params[0]
        alpha = params[1]
        gamma = params[2]
        beta = params[3]
        deltas = params[5:]

        variance = np.zeros(self.n_obs)
        returns_arr = self.returns.to_numpy(dtype=float)
        mean_return = float(returns_arr.mean())
        residuals = returns_arr - mean_return

        # Initialize with unconditional variance estimate
        variance[0] = float(np.var(returns_arr))

        for t in range(1, self.n_obs):
            eps_sq_prev = residuals[t - 1] ** 2
            leverage_term = gamma * eps_sq_prev * (residuals[t - 1] < 0)

            v = omega + alpha * eps_sq_prev + leverage_term + beta * variance[t - 1]

            if self.has_exog:
                v += float(self._exog_array[t] @ deltas)

            # Ensure positive variance
            variance[t] = v if v > 1e-8 else 1e-8

        return variance, residuals

    def _loglik_contributions(self, params: np.ndarray) -> np.ndarray:
        """
        Per-observation log-likelihood contributions (Student-t GJR-GARCH-X).

        Returns one log-density value per observation. The constant
        (parameter-only) log-gamma normalising term is computed once and
        broadcast, rather than recomputed inside a per-observation loop.
        """
        nu = params[4]

        variance, residuals = self._variance_recursion(params)
        std_residuals = residuals / np.sqrt(variance)

        # Constant Student-t normalising term (depends on nu only, not on t).
        log_const = (
            gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
        )

        contributions = (
            log_const
            - 0.5 * np.log(variance)
            - ((nu + 1) / 2) * np.log(1.0 + std_residuals**2 / (nu - 2))
        )
        return contributions

    def _log_likelihood(self, params: np.ndarray) -> float:
        """Compute negative log-likelihood for Student-t GJR-GARCH-X."""
        try:
            total = float(np.sum(self._loglik_contributions(params)))
            if not np.isfinite(total):
                return 1e8
            return -total
        except (ValueError, OverflowError, RuntimeWarning, FloatingPointError):
            return 1e8

    def _parameter_constraints(self) -> list[dict]:
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
        start_vals = np.array(
            [
                sample_var * 0.1,  # omega
                0.05,  # alpha
                0.05,  # gamma (leverage)
                0.85,  # beta
                5.0,  # nu
            ]
        )
        if self.has_exog:
            start_vals = np.append(start_vals, np.zeros(self.n_exog))
        return start_vals

    def estimate(
        self,
        method: str = "SLSQP",
        max_iter: int = 1000,
        verbose: bool = False,
        cov_type: str = "robust",
        alpha_max: float = 0.99,
        beta_max: float = 0.999,
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
        cov_type : {"robust", "hessian"}, default "robust"
            Standard-error estimator. ``"robust"`` uses the Bollerslev-Wooldridge
            (1992) QMLE sandwich H⁻¹·OPG·H⁻¹; ``"hessian"`` uses the classical
            inverse observed information.
        alpha_max : float, default 0.99
            Upper bound on the ARCH coefficient α. Relaxed from the historical
            hard cap of 0.3, which silently bound on high-volatility daily series
            (e.g. crypto). The economically meaningful restriction is the
            stationarity constraint α + β + |γ|/2 < 1, which is always enforced.
        beta_max : float, default 0.999
            Upper bound on the GARCH coefficient β. Relaxed from the historical
            hard cap of 0.95 for the same reason as ``alpha_max``.

        Returns
        -------
        GJRGARCHXResults
            Estimation results container.
        """
        if cov_type not in ("robust", "hessian"):
            raise ValueError(
                f"cov_type must be 'robust' or 'hessian', got {cov_type!r}."
            )
        if not 0 < alpha_max <= 1:
            raise ValueError(f"alpha_max must be in (0, 1], got {alpha_max}.")
        if not 0 < beta_max < 1:
            raise ValueError(f"beta_max must be in (0, 1), got {beta_max}.")

        if verbose:
            print(f"Estimating GJR-GARCH-X with {self.n_exog} exogenous variables...")

        start_vals = self._get_starting_values()

        bounds: list[tuple[float | None, float | None]] = [
            (1e-8, None),  # omega > 0
            (1e-8, alpha_max),  # alpha
            (-0.5, 0.5),  # gamma (leverage)
            (1e-8, beta_max),  # beta
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

            std_errors, pvalues = self._compute_standard_errors(
                optimal_params, cov_type=cov_type
            )

            log_lik = -result.fun
            aic = 2 * self.n_params - 2 * log_lik
            bic = np.log(self.n_obs) * self.n_params - 2 * log_lik

            # Classify exogenous effects
            exog_effects = {}
            event_effects = {}
            sentiment_effects = {}

            sentiment_keywords = {"sentiment", "gdelt", "tone", "mood", "fear", "greed"}

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
                print(f"  Std. errors: {cov_type}")

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
                cov_type=cov_type,
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
                cov_type=cov_type,
            )

    def _compute_standard_errors(
        self, params: np.ndarray, cov_type: str = "robust"
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Compute standard errors and two-sided p-values.

        Parameters
        ----------
        params : np.ndarray
            Estimated parameter vector.
        cov_type : {"robust", "hessian"}
            ``"robust"`` returns Bollerslev-Wooldridge QMLE sandwich SEs;
            ``"hessian"`` returns inverse-observed-information SEs.
        """
        nan_dict = dict.fromkeys(self.param_names, np.nan)
        try:
            cov_matrix = self._covariance_matrix(params, cov_type=cov_type)

            dof = self.n_obs - self.n_params
            if dof <= 0:
                return nan_dict, dict(nan_dict)

            # Guard the sqrt against non-positive variances (e.g. when a
            # non-positive-definite Hessian leaks through the pseudo-inverse).
            var_params = np.diag(cov_matrix)
            std_errs = np.full(self.n_params, np.nan)
            positive = var_params > 0
            std_errs[positive] = np.sqrt(var_params[positive])

            with np.errstate(divide="ignore", invalid="ignore"):
                t_stats = params / std_errs
            pvals = 2 * (1 - student_t.cdf(np.abs(t_stats), dof))

            return (
                dict(zip(self.param_names, std_errs, strict=True)),
                dict(zip(self.param_names, pvals, strict=True)),
            )

        except (np.linalg.LinAlgError, ValueError):
            return nan_dict, dict(nan_dict)

    def _covariance_matrix(
        self, params: np.ndarray, cov_type: str = "robust"
    ) -> np.ndarray:
        """
        Parameter covariance matrix.

        Builds the observed information A = Hessian of the negative log-likelihood
        and inverts it (the inverse-Hessian / "hessian" covariance). For the
        Bollerslev-Wooldridge robust covariance, it additionally forms the outer
        product of gradients B = Σ_t s_t s_tᵀ from the per-observation scores and
        returns the QMLE sandwich A⁻¹ B A⁻¹.

        The Hessian is symmetrised and tested for positive definiteness; if it is
        not positive definite (or is singular), a pseudo-inverse is used and a
        ``RuntimeWarning`` is emitted, so a degenerate fit degrades gracefully
        rather than raising.
        """
        hessian = self._numerical_hessian(params)
        # Symmetrise: finite-difference Hessians are only symmetric up to noise.
        hessian = 0.5 * (hessian + hessian.T)

        min_eig = float(np.linalg.eigvalsh(hessian).min())
        if min_eig <= 0:
            warnings.warn(
                "Observed-information matrix is not positive definite "
                f"(min eigenvalue {min_eig:.3e}); falling back to the "
                "Moore-Penrose pseudo-inverse for the covariance. Standard "
                "errors for near-degenerate parameters may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
            a_inv = np.linalg.pinv(hessian)
        else:
            a_inv = np.linalg.inv(hessian)

        if cov_type == "hessian":
            return a_inv

        # Robust (Bollerslev-Wooldridge) sandwich: A^-1 B A^-1.
        scores = self._score_contributions(params)
        opg = scores.T @ scores
        sandwich = a_inv @ opg @ a_inv
        if not np.all(np.isfinite(sandwich)):
            warnings.warn(
                "Robust (sandwich) covariance contains non-finite entries; "
                "falling back to inverse-Hessian standard errors.",
                RuntimeWarning,
                stacklevel=2,
            )
            return a_inv
        return sandwich

    def _score_contributions(self, params: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Per-observation score matrix G (n_obs × n_params).

        Column k is the central-difference derivative of the per-observation
        log-likelihood contributions with respect to parameter k. The outer
        product Gᵀ·G is the OPG (information-matrix) estimator used by the
        Bollerslev-Wooldridge sandwich.
        """
        n = len(params)
        scores = np.zeros((self.n_obs, n))
        steps = h * np.maximum(np.abs(params), 1.0)

        for k in range(n):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[k] += steps[k]
            params_minus[k] -= steps[k]

            contrib_plus = self._loglik_contributions(params_plus)
            contrib_minus = self._loglik_contributions(params_minus)
            scores[:, k] = (contrib_plus - contrib_minus) / (2 * steps[k])

        # Guard against the rare non-finite contribution at a perturbed point.
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    def _numerical_hessian(self, params: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """Compute numerical Hessian of the negative log-likelihood (central diff)."""
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
    exog_vars: pd.DataFrame | None = None,
    method: str = "SLSQP",
    max_iter: int = 1000,
    verbose: bool = False,
    cov_type: str = "robust",
    alpha_max: float = 0.99,
    beta_max: float = 0.999,
) -> GJRGARCHXResults:
    """
    Estimate GJR-GARCH-X model with exogenous variance regressors.

    This is the main entry point for the package. Estimates a GJR-GARCH model
    with Student-t innovations and exogenous variables in the variance equation.

    Parameters
    ----------
    returns : array-like
        Returns series. Recommend log returns × 100 for numerical stability.
        Accepts a ``pd.Series``, a single-column DataFrame, or any 1-D array-like.
    exog_vars : pd.DataFrame or array-like, optional
        Exogenous variables for the variance equation. A DataFrame index must
        cover the returns index; a bare array must match the number of returns.
    method : str, default "SLSQP"
        Optimization method. Recommended: "SLSQP" or "L-BFGS-B".
    max_iter : int, default 1000
        Maximum optimizer iterations.
    verbose : bool, default False
        Print estimation progress to stdout.
    cov_type : {"robust", "hessian"}, default "robust"
        Standard-error estimator. ``"robust"`` returns Bollerslev-Wooldridge
        (1992) QMLE sandwich standard errors; ``"hessian"`` returns the classical
        inverse-observed-information standard errors.
    alpha_max : float, default 0.99
        Upper bound on the ARCH coefficient α (see :meth:`GJRGARCHXEstimator.estimate`).
    beta_max : float, default 0.999
        Upper bound on the GARCH coefficient β.

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

    Classical inverse-Hessian standard errors instead of the robust default:

    >>> results = estimate_gjr_garch_x(returns, cov_type="hessian")

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

    Bollerslev, T., & Wooldridge, J. M. (1992). Quasi-maximum likelihood
    estimation and inference in dynamic models with time-varying covariances.
    Econometric Reviews, 11(2), 143-172.
    """
    estimator = GJRGARCHXEstimator(returns, exog_vars)
    return estimator.estimate(
        method=method,
        max_iter=max_iter,
        verbose=verbose,
        cov_type=cov_type,
        alpha_max=alpha_max,
        beta_max=beta_max,
    )


# Backwards compatibility alias
estimate_tarch_x = estimate_gjr_garch_x
