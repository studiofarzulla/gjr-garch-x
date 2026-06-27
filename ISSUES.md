# Review Issues

The following issues were identified during code review. Each section can be used to create a GitHub issue.

---

## Issue 1: Add missing `max_iter` parameter to API Reference

**Labels:** `documentation`

### Description

The README API Reference section is missing the `max_iter` parameter in the function signature.

**README currently shows:**
```python
estimate_gjr_garch_x(returns, exog_vars, method='SLSQP', verbose=False)
```

**Actual implementation:**
```python
estimate_gjr_garch_x(returns, exog_vars, method='SLSQP', max_iter=1000, verbose=False)
```

### Location
- `README.md` line 103

### Suggested Fix
Update the API Reference section to include the `max_iter` parameter:

```python
estimate_gjr_garch_x(returns, exog_vars, method='SLSQP', max_iter=1000, verbose=False)
```

And add to the Parameters list:
- `max_iter`: Maximum number of optimizer iterations (default: 1000)

### Priority
Low - Documentation only, does not affect functionality.

---

## Issue 2: Resolve mypy strict mode type annotation errors

**Labels:** `bug`

### Description

Running `mypy --strict` on the source code reveals 6 type annotation errors. While the code is functional, these should be fixed to fully meet the "Full type annotations for IDE support" claim.

### Errors Found

```
src/gjr_garch_x/__init__.py:117: error: Function is missing a return type annotation  [no-untyped-def]
src/gjr_garch_x/__init__.py:292: error: Incompatible return value type (got "dict[str, ndarray[Any, dtype[Any]]]", expected "dict[str, float]")  [return-value]
src/gjr_garch_x/__init__.py:341: error: Incompatible return value type (got "tuple[Any, Union[Any, ExtensionArray]]", expected "tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]")  [return-value]
src/gjr_garch_x/__init__.py:398: error: Returning Any from function declared to return "ndarray[Any, dtype[Any]]"  [no-any-return]
src/gjr_garch_x/__init__.py:437: error: Argument 1 to "append" of "list" has incompatible type "tuple[None, None]"; expected "tuple[float, Optional[float]]"  [arg-type]
src/gjr_garch_x/__init__.py:599: error: Returning Any from function declared to return "ndarray[Any, dtype[Any]]"  [no-any-return]
```

### Suggested Fixes

1. **Line 117** (`__post_init__`): Add `-> None` return type annotation
2. **Line 292** (`_unpack_params`): Fix return type to `Dict[str, Union[float, np.floating]]` or cast values
3. **Line 341** (`_variance_recursion`): Ensure numpy array types are explicit
4. **Line 398** (`_get_starting_values`): Add explicit return type cast
5. **Line 437** (bounds append): Fix tuple type annotation for bounds list
6. **Line 599** (`_numerical_hessian`): Add explicit return type cast

### Priority
Low - Does not affect functionality, but improves developer experience and IDE support.

---

## Issue 3: Handle ill-conditioned Hessian gracefully without RuntimeWarning

**Labels:** `enhancement`

### Description

During estimation, when the Hessian matrix is ill-conditioned (e.g., with short time series or edge cases), a RuntimeWarning is emitted:

```
RuntimeWarning: invalid value encountered in sqrt
  std_errs = np.sqrt(np.diag(cov_matrix))
```

This occurs in `_compute_standard_errors()` at line 536.

### Current Behavior
- Warning is printed to stderr
- Code handles it gracefully by returning NaN for affected standard errors
- Tests pass but with warnings

### Suggested Fix

Suppress the specific warning or add explicit checks before the sqrt operation:

```python
def _compute_standard_errors(self, params: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
    try:
        hessian = self._numerical_hessian(params)
        cov_matrix = np.linalg.inv(hessian)
        diag = np.diag(cov_matrix)

        # Check for negative variances before sqrt
        if np.any(diag < 0):
            return (
                {name: np.nan for name in self.param_names},
                {name: np.nan for name in self.param_names},
            )

        std_errs = np.sqrt(diag)
        # ... rest of method
```

Or use `np.errstate(invalid='ignore')` context manager around the sqrt call.

### Priority
Low - Cosmetic issue, does not affect results.

---

## Issue 4: Handle half-life display when persistence is near 1.0

**Labels:** `enhancement`

### Description

When the persistence parameter (α + β + |γ|/2) is very close to 1.0, the half-life calculation in `summary()` produces mathematically questionable results.

### Example Output

```
Persistence (α + β + |γ|/2): 0.9990
Half-life of shocks:         -692.8 periods
```

A negative half-life is not meaningful in this context.

### Root Cause

The half-life formula is:
```python
half_life = -np.log(0.5) / np.log(persistence)
```

When `persistence ≈ 1.0`:
- `np.log(persistence)` approaches 0 from below (negative)
- Division by a very small negative number produces a large negative result

### Suggested Fix

Add a check for near-unit-root persistence in the summary method:

```python
if 0 < persistence < 0.9999:
    half_life = -np.log(0.5) / np.log(persistence)
    lines.append(f"Half-life of shocks:         {half_life:.1f} periods")
elif persistence >= 0.9999:
    lines.append("Half-life of shocks:         ∞ (near unit root)")
```

### Location
- `src/gjr_garch_x/__init__.py` lines 169-171

### Priority
Low - Display/cosmetic issue only.
