"""
Tests for gjr-garch-x package.
"""

import numpy as np
import pandas as pd
import pytest

from gjr_garch_x import (
    estimate_gjr_garch_x,
    GJRGARCHXEstimator,
    GJRGARCHXResults,
    # Backwards compatibility
    estimate_tarch_x,
    TARCHXEstimator,
    TARCHXResults,
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
        assert type(results_gjr) == type(results_tarch)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
