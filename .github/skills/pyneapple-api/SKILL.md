---
name: pyneapple-api
description: 'Detailed Pyneapple solver/model API reference for plugin development. Use when: implementing a new solver, debugging fit() contract issues, checking param_/diagnostics_ shapes, understanding model interfaces, or verifying entry_point wiring.'
---

# Pyneapple Plugin API Reference

## When to use me

Use this skill when:
- Implementing `fit()` and need to know exact input/output shapes
- Debugging why `get_params()` or `get_diagnostics()` raises `RuntimeError`
- Checking how parametric vs distribution models differ
- Verifying TOML config ↔ entry_point wiring
- Understanding `pixel_fixed_params` per-pixel fixed parameter maps

## BaseSolver — full contract

Source: `pyneapple.solvers.base.BaseSolver`

### Constructor

```python
def __init__(self, model, max_iter=250, tol=1e-8, verbose=False, **solver_kwargs):
```

Sets `self.model`, `self.max_iter`, `self.tol`, `self.verbose`.
Initializes `self.params_` and `self.diagnostics_` as empty dicts.

### fit() — abstract

```python
@abstractmethod
def fit(self, *args, **kwargs) -> "BaseSolver":
```

Must:
1. Call `self._reset_state()` — clears `params_` and `diagnostics_`
2. Perform optimization
3. Populate `self.params_` with fitted parameter values
4. Populate `self.diagnostics_` with solver metadata
5. Return `self`

### get_params() / get_diagnostics()

Both raise `RuntimeError` if the corresponding dict is empty (i.e. `fit()` was not called or stored nothing). Both return `.copy()` of the dict.

### _reset_state()

Sets `self.diagnostics_ = {}` and `self.params_ = {}`. Call at the start of every `fit()`.

## CurveFitSolver — parametric reference implementation

### Constructor signature

```python
def __init__(self, model, max_iter, tol,
             p0: dict[str, float] | None = None,
             bounds: dict[str, tuple[float, float]] | None = None,
             verbose=False, method="trf", multi_threading=False,
             use_jacobian=True, **solver_kwargs):
```

- `p0` keys must match `model.param_names`
- `bounds` keys must match `model.param_names`, values are `(lower, upper)` tuples
- `n_pools` extracted from `solver_kwargs` for joblib parallelism

### fit() signature

```python
def fit(self, xdata, ydata, p0=None, bounds=None,
        pixel_fixed_params=None, **fit_kwargs) -> "CurveFitSolver":
```

- `xdata`: `np.ndarray` shape `(n_measurements,)` — e.g. b-values
- `ydata`: `np.ndarray` shape `(n_pixels, n_measurements)` or `(n_measurements,)` for single pixel
- `pixel_fixed_params`: `dict[str, np.ndarray]` where each value is shape `(n_pixels,)` — per-pixel fixed parameter maps (e.g. T1 from a prior fit)

### params_ output format

```python
self.params_ = {
    "f1": np.ndarray shape (n_pixels,),   # or float for single pixel
    "D1": np.ndarray shape (n_pixels,),
    "D2": np.ndarray shape (n_pixels,),
    # ... one entry per free param in model.param_names
}
```

### diagnostics_ output format

```python
self.diagnostics_ = {
    "pcov": np.ndarray shape (n_pixels, n_params, n_params),  # covariance matrices
    "n_pixels": int,
}
```

## NNLSSolver — distribution reference implementation

### Constructor signature

```python
def __init__(self, model: DistributionModel, reg_order=0, mu=0.02,
             max_iter=250, tol=1e-8, verbose=False,
             multi_threading=False, **solver_kwargs):
```

- `model` must be a `DistributionModel` providing `bins`, `n_bins`, `get_basis(xdata)`
- `reg_order`: 0=none, 1=first difference, 2=second difference, 3=extended

### fit() signature

```python
def fit(self, xdata, signal, pixel_fixed_params=None) -> "NNLSSolver":
```

- `xdata`: `np.ndarray` shape `(n_measurements,)`
- `signal`: `np.ndarray` shape `(n_pixels, n_measurements)` or `(n_measurements,)`
- `pixel_fixed_params`: accepted for API compatibility, ignored by NNLS

### params_ output format

```python
self.params_ = {
    "coefficients": np.ndarray shape (n_pixels, n_bins),  # spectrum per pixel
}
```

### diagnostics_ output format

```python
self.diagnostics_ = {
    "residual": np.ndarray shape (n_pixels,),  # NNLS residual per pixel
}
```

## ParametricModel interface

Models the solver receives as `self.model`:

| Attribute / Method | Type / Signature | Description |
|---|---|---|
| `param_names` | `list[str]` | Free parameter names (excludes fixed) |
| `_all_param_names` | `list[str]` | All parameter names including fixed |
| `n_params` | `int` | Number of free parameters |
| `fixed_params` | `dict[str, float]` | Model-level fixed parameter values |
| `forward(xdata, *params)` | `np.ndarray → np.ndarray` | Signal equation |
| `jacobian(xdata, *params)` | `np.ndarray → np.ndarray \| None` | Analytical Jacobian |
| `forward_with_fixed(xdata, fixed_dict, *free)` | `np.ndarray` | Forward with injected fixed params |
| `jacobian_with_fixed(xdata, fixed_dict, *free)` | `np.ndarray \| None` | Jacobian with fixed param columns sliced |

Concrete models: `MonoExpModel`, `BiExpModel`, `TriExpModel`

## DistributionModel interface

| Attribute / Method | Type / Signature | Description |
|---|---|---|
| `bins` | `np.ndarray` shape `(n_bins,)` | Discrete parameter grid |
| `n_bins` | `int` | Number of bins |
| `get_basis(xdata)` | `np.ndarray` shape `(n_measurements, n_bins)` | Basis/dictionary matrix |
| `forward(xdata, *spectrum)` | `np.ndarray` | `basis @ spectrum` reconstruction |

Concrete models: `NNLSModel`

## TOML config ↔ entry_point mapping

A user selects a solver in their TOML config:

```toml
[Fitting.solver]
type = "gpufit_curvefit"
max_iter = 500
```

Pyneapple's `load_config()` looks up `type` in `_SOLVER_REGISTRY`. Plugins are discovered via:

```python
from importlib.metadata import entry_points

for ep in entry_points(group="pyneapple.solvers"):
    _SOLVER_REGISTRY[ep.name] = ep  # lazy — ep.load() called on first use
```

The entry_point name (`gpufit_curvefit`) must match the TOML `type` value.

All remaining keys under `[Fitting.solver]` (except reserved: `type`, `max_iter`, `tol`, `p0`, `bounds`, `fraction_constraint`) are forwarded as `**solver_kwargs` to the constructor.

## Error conventions

| Situation | Exception | Example |
|---|---|---|
| Invalid parameter names | `ValueError` | `"Missing required parameters: {'D1'}"` |
| Shapes mismatch | `ValueError` | `"xdata length 10 != ydata columns 8"` |
| fit() not called yet | `RuntimeError` | `"No parameters available..."` |
| CUDA not available | `RuntimeError` | `"CUDA is not available on this machine..."` |
| Optional dep missing | `ImportError` | `"pygpufit is required for GPU fitting"` |
| Pixel fit failure | `logger.warning()` + return zeros | `"Pixel 42 fit failed: {e}"` |
