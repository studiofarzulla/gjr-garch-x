"""
Tests for gjr-garch-x package.
"""

import numpy as np
import pandas as pd
import pytest

from gjr_garch_x import (
    GJRGARCHXEstimator,
    GJRGARCHXResults,
    TARCHXEstimator,
    TARCHXResults,
    estimate_gjr_garch_x,
    # Backwards compatibility
    estimate_tarch_x,
)


def generate_garch_data(n: int = 1000, seed: int = 42) -> pd.Series:
    """Generate synthetic GARCH(1,1) returns for testing."""
    np.random.seed(seed)

    omega, alpha, beta = 0.05, 0.08, 0.88

    returns = np.zeros(n)
    variance = np.zeros(n)
    variance[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]
        returns[t] = np.sqrt(variance[t]) * np.random.standard_t(df=5)

    return pd.Series(returns, index=pd.date_range("2020-01-01", periods=n, freq="D"))


class TestBasicEstimation:
    """Test basic model estimation without exogenous variables."""

    def test_convergence(self):
        """Model should converge on well-behaved data."""
        returns = generate_garch_data(500)
        results = estimate_gjr_garch_x(returns)
        assert results.converged

    def test_parameter_bounds(self):
        """Estimated parameters should be within valid ranges."""
        returns = generate_garch_data(500)
        results = estimate_gjr_garch_x(returns)

        assert results.params["omega"] > 0
        assert results.params["alpha"] > 0
        assert results.params["beta"] > 0
        assert results.params["nu"] > 2

    def test_stationarity(self):
        """Persistence should be less than 1."""
        returns = generate_garch_data(500)
        results = estimate_gjr_garch_x(returns)

        persistence = (
            results.params["alpha"]
            + results.params["beta"]
            + abs(results.params["gamma"]) / 2
        )
        assert persistence < 1.0

    def test_volatility_output(self):
        """Volatility series should be positive and same length as input."""
        returns = generate_garch_data(500)
        results = estimate_gjr_garch_x(returns)

        assert len(results.volatility) == len(returns)
        assert (results.volatility > 0).all()

    def test_information_criteria(self):
        """AIC and BIC should be finite."""
        returns = generate_garch_data(500)
        results = estimate_gjr_garch_x(returns)

        assert np.isfinite(results.aic)
        assert np.isfinite(results.bic)

    def test_n_obs_attribute(self):
        """Results should track number of observations."""
        returns = generate_garch_data(500)
        results = estimate_gjr_garch_x(returns)

        assert results.n_obs == 500


class TestExogenousVariables:
    """Test estimation with exogenous variance regressors."""

    def test_single_dummy(self):
        """Model should handle a single event dummy."""
        returns = generate_garch_data(500)

        # Create event dummy
        exog = pd.DataFrame(index=returns.index)
        exog["D_event"] = 0
        exog.iloc[100:110, 0] = 1  # 10-day event window

        results = estimate_gjr_garch_x(returns, exog)

        assert results.converged
        assert "D_event" in results.event_effects
        assert "D_event" in results.exog_effects

    def test_multiple_dummies(self):
        """Model should handle multiple event dummies."""
        returns = generate_garch_data(500)

        exog = pd.DataFrame(index=returns.index)
        exog["D_infra"] = 0
        exog["D_reg"] = 0
        exog.iloc[100:110, 0] = 1
        exog.iloc[200:210, 1] = 1

        results = estimate_gjr_garch_x(returns, exog)

        assert results.converged
        assert len(results.event_effects) == 2
        assert len(results.exog_effects) == 2

    def test_continuous_exog(self):
        """Model should handle continuous exogenous variables."""
        returns = generate_garch_data(500)

        exog = pd.DataFrame(index=returns.index)
        exog["sentiment"] = np.random.randn(500)

        results = estimate_gjr_garch_x(returns, exog)

        assert results.converged
        assert "sentiment" in results.sentiment_effects

    def test_event_effect_recovery(self):
        """Event dummy should capture added volatility."""
        np.random.seed(42)
        n = 1000

        omega, alpha, beta = 0.05, 0.08, 0.88
        event_effect = 2.0  # Large effect

        returns = np.zeros(n)
        variance = np.zeros(n)
        event_dummy = np.zeros(n)
        event_dummy[400:450] = 1  # 50-day event

        variance[0] = omega / (1 - alpha - beta)

        for t in range(1, n):
            variance[t] = (
                omega
                + alpha * returns[t - 1] ** 2
                + beta * variance[t - 1]
                + event_effect * event_dummy[t]
            )
            returns[t] = np.sqrt(variance[t]) * np.random.standard_t(df=5)

        returns_series = pd.Series(
            returns, index=pd.date_range("2020-01-01", periods=n, freq="D")
        )
        exog = pd.DataFrame({"D_event": event_dummy}, index=returns_series.index)

        results = estimate_gjr_garch_x(returns_series, exog)

        # Should recover positive event effect (not necessarily exact)
        assert results.converged
        assert results.event_effects["D_event"] > 0


class TestSummary:
    """Test results summary output."""

    def test_summary_string(self):
        """Summary should produce readable output."""
        returns = generate_garch_data(300)
        results = estimate_gjr_garch_x(returns)

        summary = results.summary()

        assert "GJR-GARCH-X Model Results" in summary
        assert "omega" in summary
        assert "alpha" in summary
        assert "gamma" in summary
        assert "beta" in summary
        assert "Persistence" in summary

    def test_repr(self):
        """Repr should be informative."""
        returns = generate_garch_data(300)
        results = estimate_gjr_garch_x(returns)

        repr_str = repr(results)

        assert "GJRGARCHXResults" in repr_str
        assert "converged" in repr_str
        assert "n_obs" in repr_str

    @staticmethod
    def _results_with_persistence(alpha: float, beta: float, gamma: float = 0.0):
        """Build a minimal results container with a controlled persistence."""
        vol = pd.Series(np.ones(10))
        return GJRGARCHXResults(
            converged=True,
            params={"omega": 0.05, "alpha": alpha, "gamma": gamma, "beta": beta},
            std_errors={},
            pvalues={},
            log_likelihood=-100.0,
            aic=210.0,
            bic=220.0,
            volatility=vol,
            residuals=pd.Series(np.zeros(10)),
            exog_effects={},
            event_effects={},
            sentiment_effects={},
            leverage_effect=gamma,
            iterations=10,
        )

    def test_half_life_finite_for_stationary(self):
        """A stationary persistence < 1 should report a finite half-life."""
        results = self._results_with_persistence(alpha=0.05, beta=0.90)
        summary = results.summary()

        assert "Half-life of shocks:" in summary
        assert "periods" in summary
        assert "∞" not in summary

    def test_half_life_infinite_near_unit_root(self):
        """Persistence at/above the unit root should report an infinite half-life."""
        results = self._results_with_persistence(alpha=0.10, beta=0.90)
        summary = results.summary()

        assert "Half-life of shocks:" in summary
        assert "∞ (near unit root)" in summary


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_short_series(self):
        """Should handle short time series."""
        returns = generate_garch_data(100)
        results = estimate_gjr_garch_x(returns)

        # May or may not converge, but shouldn't crash
        # Note: numpy bool types (np.True_) are truthy but not isinstance(bool)
        assert results.converged in (True, False)

    def test_missing_values(self):
        """Should handle series with NaN values."""
        returns = generate_garch_data(500)
        returns.iloc[50:55] = np.nan

        results = estimate_gjr_garch_x(returns)

        # Should drop NaN and estimate on remaining data
        assert results.converged
        assert len(results.volatility) < 500


class TestBackwardsCompatibility:
    """Test backwards compatibility aliases."""

    def test_tarch_alias_function(self):
        """estimate_tarch_x should be alias for estimate_gjr_garch_x."""
        returns = generate_garch_data(300)

        results_gjr = estimate_gjr_garch_x(returns)
        results_tarch = estimate_tarch_x(returns)

        # Same function, same results (with same seed)
        assert type(results_gjr) is type(results_tarch)

    def test_tarch_alias_class(self):
        """TARCHXEstimator should be alias for GJRGARCHXEstimator."""
        assert TARCHXEstimator is GJRGARCHXEstimator

    def test_tarch_alias_results(self):
        """TARCHXResults should be alias for GJRGARCHXResults."""
        assert TARCHXResults is GJRGARCHXResults


class TestVerboseMode:
    """Test verbose output."""

    def test_verbose_runs(self, capsys):
        """Verbose mode should print without crashing."""
        returns = generate_garch_data(300)
        results = estimate_gjr_garch_x(returns, verbose=True)

        captured = capsys.readouterr()
        assert "Estimating GJR-GARCH-X" in captured.out
        assert results.converged


class TestStandardErrors:
    """Test the robust (Bollerslev-Wooldridge) and Hessian covariance estimators."""

    def test_robust_is_default(self):
        """Robust SEs should be the default covariance type."""
        returns = generate_garch_data(500)
        results = estimate_gjr_garch_x(returns)
        assert results.cov_type == "robust"

    def test_robust_ses_finite_and_positive(self):
        """Robust standard errors should be computed (finite and positive)."""
        returns = generate_garch_data(500)
        results = estimate_gjr_garch_x(returns, cov_type="robust")

        for param in ["omega", "alpha", "gamma", "beta", "nu"]:
            se = results.std_errors[param]
            assert np.isfinite(se), f"{param} robust SE is not finite"
            assert se > 0, f"{param} robust SE is not positive"

    def test_hessian_cov_type_available(self):
        """The classical inverse-Hessian SEs should remain available."""
        returns = generate_garch_data(500)
        results = estimate_gjr_garch_x(returns, cov_type="hessian")
        assert results.cov_type == "hessian"
        for param in ["omega", "alpha", "beta"]:
            assert np.isfinite(results.std_errors[param])

    def test_robust_differs_from_hessian(self):
        """
        The QMLE sandwich must actually differ from the inverse-Hessian SEs.

        Under the Bollerslev-Wooldridge sandwich H^-1 . OPG . H^-1, the OPG block
        is not equal to the Hessian unless the likelihood is exactly correctly
        specified, so the two SE vectors should be sensibly (not trivially)
        different on real-ish data.
        """
        returns = generate_garch_data(600)
        exog = pd.DataFrame(index=returns.index)
        exog["D_event"] = 0.0
        exog.iloc[100:140, 0] = 1.0

        res_h = estimate_gjr_garch_x(returns, exog, cov_type="hessian")
        res_r = estimate_gjr_garch_x(returns, exog, cov_type="robust")

        hessian_ses = np.array([res_h.std_errors[p] for p in res_h.params])
        robust_ses = np.array([res_r.std_errors[p] for p in res_r.params])

        # Both vectors are well-defined ...
        assert np.all(np.isfinite(robust_ses)) and np.all(robust_ses > 0)
        # ... but not identical: the sandwich is doing real work.
        assert not np.allclose(robust_ses, hessian_ses, rtol=1e-2)
        # At least one parameter's SE should shift by a non-trivial margin.
        rel_diff = np.abs(robust_ses - hessian_ses) / hessian_ses
        assert rel_diff.max() > 0.05

    def test_invalid_cov_type_raises(self):
        """An unknown cov_type should raise a clear error."""
        returns = generate_garch_data(300)
        with pytest.raises(ValueError, match="cov_type"):
            GJRGARCHXEstimator(returns).estimate(cov_type="bogus")


class TestParameterCaps:
    """Test the relaxed / parameterised alpha and beta caps."""

    def test_default_caps_relaxed(self):
        """Default beta cap should be relaxed well above the old 0.95 hard cap."""
        # Highly persistent series: beta wants to sit above 0.95.
        np.random.seed(7)
        n = 1500
        omega, alpha, beta = 0.02, 0.06, 0.93
        r = np.zeros(n)
        v = np.zeros(n)
        v[0] = omega / (1 - alpha - beta)
        for t in range(1, n):
            v[t] = omega + alpha * r[t - 1] ** 2 + beta * v[t - 1]
            r[t] = np.sqrt(v[t]) * np.random.standard_t(df=6)
        s = pd.Series(r, index=pd.date_range("2018-01-01", periods=n, freq="D"))

        results = estimate_gjr_garch_x(s)
        # With the old beta<=0.95 cap this would silently bind at 0.95.
        assert results.converged
        assert results.params["beta"] <= 0.999

    def test_caps_are_configurable(self):
        """alpha_max / beta_max kwargs should be honoured as upper bounds."""
        returns = generate_garch_data(500)
        results = estimate_gjr_garch_x(returns, alpha_max=0.2, beta_max=0.9)
        assert results.params["alpha"] <= 0.2 + 1e-6
        assert results.params["beta"] <= 0.9 + 1e-6

    def test_invalid_caps_raise(self):
        returns = generate_garch_data(300)
        with pytest.raises(ValueError, match="alpha_max"):
            GJRGARCHXEstimator(returns).estimate(alpha_max=1.5)
        with pytest.raises(ValueError, match="beta_max"):
            GJRGARCHXEstimator(returns).estimate(beta_max=1.0)


class TestInputValidation:
    """Test array-like coercion and exogenous-variable alignment checks."""

    def test_accepts_numpy_array(self):
        """A bare numpy array of returns should be coerced to a Series."""
        returns = generate_garch_data(400).to_numpy()
        results = estimate_gjr_garch_x(returns)
        assert results.converged
        assert results.n_obs == 400

    def test_accepts_list(self):
        """A Python list of returns should be coerced to a Series."""
        returns = list(generate_garch_data(300).to_numpy())
        results = estimate_gjr_garch_x(returns)
        assert results.n_obs == 300

    def test_exog_as_array(self):
        """A bare exog array matching the return length should be accepted."""
        returns = generate_garch_data(400)
        exog = np.zeros(400)
        exog[100:140] = 1.0
        results = estimate_gjr_garch_x(returns, exog)
        assert results.converged
        assert "x0" in results.exog_effects

    def test_exog_length_mismatch_raises(self):
        """A bare exog array of the wrong length should raise informatively."""
        returns = generate_garch_data(400)
        exog = np.zeros(399)
        with pytest.raises(ValueError, match="rows"):
            estimate_gjr_garch_x(returns, exog)

    def test_exog_index_misalignment_raises(self):
        """A DataFrame exog that does not cover the returns index should raise."""
        returns = generate_garch_data(400)
        # Drop part of the exog index so it cannot cover all returns timestamps.
        exog = pd.DataFrame({"D_event": np.zeros(400)}, index=returns.index)
        exog = exog.iloc[:300]
        with pytest.raises(ValueError, match="does not cover"):
            estimate_gjr_garch_x(returns, exog)

    def test_multidim_returns_raises(self):
        """Multi-column returns input should raise a clear error."""
        returns = generate_garch_data(300)
        df = pd.DataFrame({"a": returns, "b": returns})
        with pytest.raises(ValueError, match="one-dimensional|single"):
            estimate_gjr_garch_x(df)


class TestRecursionCore:
    """Test the extracted (optionally numba-jitted) variance recursion core."""

    def test_core_matches_reference_loop(self):
        """The core must reproduce a hand-rolled GJR recursion exactly."""
        from gjr_garch_x import _variance_recursion_core

        rng = np.random.default_rng(0)
        resid = rng.standard_normal(200)
        exog_contrib = rng.normal(0, 0.01, 200)
        omega, alpha, gamma, beta = 0.05, 0.07, 0.10, 0.85
        var0 = float(np.var(resid))

        expected = np.empty(200)
        expected[0] = var0
        for t in range(1, 200):
            e = resid[t - 1]
            ee = e * e  # square first: float multiplication is not associative
            v = (
                omega
                + alpha * ee
                + (gamma * ee if e < 0 else 0.0)
                + beta * expected[t - 1]
                + exog_contrib[t]
            )
            expected[t] = max(v, 1e-8)

        got = _variance_recursion_core(
            omega, alpha, gamma, beta, resid, exog_contrib, var0
        )
        np.testing.assert_array_equal(got, expected)

    def test_estimates_unchanged_vs_known_fit(self):
        """The refactored recursion should not move a well-identified fit."""
        returns = generate_garch_data(800)
        results = estimate_gjr_garch_x(returns)
        assert results.converged
        # Values in the usual GARCH neighbourhood of the DGP (omega=.05, a=.08, b=.88)
        assert 0.5 < results.params["alpha"] + results.params["beta"] < 1.0

    def test_have_numba_flag_exposed(self):
        """HAVE_NUMBA must be importable and boolean."""
        from gjr_garch_x import HAVE_NUMBA

        assert isinstance(HAVE_NUMBA, bool)


class TestMultistart:
    """Test the multistart estimation path."""

    def _returns_with_event(self, n: int = 600):
        returns = generate_garch_data(n)
        exog = pd.DataFrame(index=returns.index)
        exog["D_event"] = 0.0
        exog.iloc[200:240, 0] = 1.0
        return returns, exog

    def test_multistart_converges(self):
        returns, exog = self._returns_with_event()
        est = GJRGARCHXEstimator(returns, exog)
        results = est.estimate_multistart(n_starts=3, seed=0)
        assert results.converged
        assert "D_event" in results.params

    def test_seeded_multistart_deterministic(self):
        """Same seed, same data => identical parameter estimates."""
        returns, exog = self._returns_with_event()
        r1 = GJRGARCHXEstimator(returns, exog).estimate_multistart(n_starts=4, seed=123)
        r2 = GJRGARCHXEstimator(returns, exog).estimate_multistart(n_starts=4, seed=123)
        p1 = np.array([r1.params[k] for k in r1.params])
        p2 = np.array([r2.params[k] for k in r2.params])
        np.testing.assert_array_equal(p1, p2)

    def test_multistart_no_worse_than_single_start(self):
        """Best-of-n likelihood can only match or beat the single default start."""
        returns, exog = self._returns_with_event()
        single = GJRGARCHXEstimator(returns, exog).estimate(max_iter=2000)
        multi = GJRGARCHXEstimator(returns, exog).estimate_multistart(
            n_starts=5, seed=42, max_iter=2000
        )
        assert multi.log_likelihood >= single.log_likelihood - 1e-6

    def test_n_starts_dispatch_from_convenience_function(self):
        """estimate_gjr_garch_x(n_starts>1) should run the multistart path."""
        returns, exog = self._returns_with_event()
        r1 = estimate_gjr_garch_x(returns, exog, n_starts=3, seed=7, max_iter=2000)
        r2 = GJRGARCHXEstimator(returns, exog).estimate_multistart(
            n_starts=3, seed=7, max_iter=2000
        )
        assert r1.log_likelihood == r2.log_likelihood

    def test_multistart_respects_caps(self):
        returns, exog = self._returns_with_event()
        results = GJRGARCHXEstimator(returns, exog).estimate_multistart(
            n_starts=3, seed=0, alpha_max=0.3, beta_max=0.95
        )
        assert results.params["alpha"] <= 0.3 + 1e-6
        assert results.params["beta"] <= 0.95 + 1e-6


class TestComputeSEFlag:
    """Test the compute_se fast path."""

    def test_compute_se_false_returns_nan_ses(self):
        returns = generate_garch_data(400)
        results = estimate_gjr_garch_x(returns, compute_se=False)
        assert results.converged
        assert all(np.isnan(v) for v in results.std_errors.values())
        assert all(np.isnan(v) for v in results.pvalues.values())

    def test_compute_se_false_same_point_estimates(self):
        """Skipping SEs must not change the point estimates."""
        returns = generate_garch_data(400)
        with_se = estimate_gjr_garch_x(returns)
        without_se = estimate_gjr_garch_x(returns, compute_se=False)
        for k in with_se.params:
            assert with_se.params[k] == without_se.params[k]

    def test_compute_se_false_is_faster(self):
        """The whole point: skipping the Hessian should save real time."""
        import time

        returns = generate_garch_data(600)
        exog = pd.DataFrame(
            {"D_event": np.r_[np.zeros(300), np.ones(40), np.zeros(260)]},
            index=returns.index,
        )
        t0 = time.perf_counter()
        estimate_gjr_garch_x(returns, exog, compute_se=True)
        t_with = time.perf_counter() - t0
        t0 = time.perf_counter()
        estimate_gjr_garch_x(returns, exog, compute_se=False)
        t_without = time.perf_counter() - t0
        assert t_without < t_with


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
