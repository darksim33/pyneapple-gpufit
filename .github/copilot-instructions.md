# Copilot Instructions — pyneapple-gpufit

This is a **Pyneapple plugin** that provides GPU-accelerated solvers via pygpufit.
It registers into Pyneapple's entry_point-based plugin system.

## What this package does

- Provides `GpuCurveFitSolver` and (optionally) `GpuNNLSSolver`
- Each solver inherits from `pyneapple.solvers.base.BaseSolver`
- Registers via `[project.entry-points."pyneapple.solvers"]` in `pyproject.toml`
- CUDA/pygpufit is only imported when a solver is instantiated, never at discovery time

## Pyneapple solver contract

Every solver **must**:

1. Inherit from `pyneapple.solvers.base.BaseSolver`
2. Call `super().__init__(model=model, max_iter=max_iter, tol=tol, verbose=verbose, **solver_kwargs)`
3. Implement `fit(self, xdata, ydata, **kwargs) -> self`
4. Inside `fit()`:
   - Call `self._reset_state()` first
   - Store fitted parameters in `self.params_` (a `dict[str, Any]`)
   - Store diagnostics in `self.diagnostics_` (a `dict[str, Any]`)
   - Return `self`

### BaseSolver interface

```python
class BaseSolver(ABC):
    def __init__(self, model, max_iter=250, tol=1e-8, verbose=False, **solver_kwargs):
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.diagnostics_: dict[str, Any] = {}
        self.params_: dict[str, Any] = {}

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseSolver": ...

    def get_diagnostics(self) -> dict[str, Any]: ...  # raises RuntimeError if empty
    def get_params(self) -> dict[str, Any]: ...        # raises RuntimeError if empty
    def _reset_state(self): ...                        # clears params_ and diagnostics_
```

### Parametric solver pattern (CurveFitSolver-style)

For fitting parametric models (MonoExp, BiExp, TriExp):

```python
class GpuCurveFitSolver(BaseSolver):
    def __init__(self, model, max_iter, tol,
                 p0: dict[str, float] | None = None,
                 bounds: dict[str, tuple[float, float]] | None = None,
                 verbose=False, multi_threading=False, **solver_kwargs):
        super().__init__(model=model, max_iter=max_iter, tol=tol, verbose=verbose)
        # Validate p0 keys match model.param_names
        # Validate bounds keys match model.param_names
        # Store p0, bounds, multi_threading, n_pools, etc.

    def fit(self, xdata, ydata, p0=None, bounds=None,
            pixel_fixed_params=None, **fit_kwargs) -> "GpuCurveFitSolver":
        self._reset_state()
        # xdata: 1D np.ndarray (e.g. b-values)
        # ydata: 2D np.ndarray shape (n_pixels, n_xdata)
        # Store results:
        #   self.params_ = {param_name: np.ndarray} for each free param
        #   self.diagnostics_ = {"pcov": ..., "n_pixels": int}
        return self
```

### Distribution solver pattern (NNLSSolver-style)

For fitting distribution models (NNLS):

```python
class GpuNNLSSolver(BaseSolver):
    def __init__(self, model: DistributionModel, reg_order=0, mu=0.02,
                 max_iter=250, tol=1e-8, verbose=False,
                 multi_threading=False, **solver_kwargs):
        super().__init__(model, max_iter, tol, verbose, **solver_kwargs)
        # model provides: model.bins, model.n_bins, model.get_basis(xdata)

    def fit(self, xdata, signal, pixel_fixed_params=None) -> "GpuNNLSSolver":
        self._reset_state()
        # Store results:
        #   self.params_["coefficients"] = np.ndarray shape (n_pixels, n_bins)
        #   self.diagnostics_["residual"] = np.ndarray shape (n_pixels,)
        return self
```

## Pyneapple model interface (read-only — do not reimplement)

Solvers receive a model instance. Key attributes/methods:

### ParametricModel
- `model.param_names -> list[str]` — free parameter names (excludes fixed)
- `model.n_params -> int` — number of free parameters
- `model.forward(xdata, *params) -> np.ndarray` — evaluate the signal equation
- `model.jacobian(xdata, *params) -> np.ndarray | None` — analytical Jacobian (optional)
- `model.fixed_params -> dict[str, float]` — model-level fixed parameters
- `model.forward_with_fixed(xdata, fixed_dict, *free_params)` — forward with injected fixed values
- `model.jacobian_with_fixed(xdata, fixed_dict, *free_params)` — Jacobian with fixed values sliced

### DistributionModel
- `model.bins -> np.ndarray` — shape (n_bins,), the discrete parameter grid
- `model.n_bins -> int` — number of bins
- `model.get_basis(xdata) -> np.ndarray` — shape (n_measurements, n_bins), the basis matrix
- `model.forward(xdata, *spectrum) -> np.ndarray` — reconstructs signal from spectrum coefficients

## Coding conventions

- **Naming:** `PascalCase` classes with layer suffix (`GpuCurveFitSolver`), `snake_case` methods
- **Fitted state:** trailing underscore: `params_`, `diagnostics_`
- **Logging:** use `loguru` via `from loguru import logger`; gate `logger.info()` behind `if self.verbose:`; never `logger.error()` then `raise` — just `raise`
- **Imports:** `from __future__ import annotations` in every module
- **Type hints:** `dict[str, float | np.ndarray]` for params, `dict[str, tuple[float, float]] | None` for bounds
- **Error handling:** `ValueError` for bad inputs, `RuntimeError` for missing prerequisites, `ImportError` for optional deps
- **Tests:** pytest, class-based grouping, `pytest-mock` for mocking, docstrings on every test

## Entry_point registration

In `pyproject.toml`:

```toml
[project.entry-points."pyneapple.solvers"]
gpufit_curvefit = "pyneapple_gpufit:GpuCurveFitSolver"
gpufit_nnls = "pyneapple_gpufit:GpuNNLSSolver"
```

The key (e.g. `gpufit_curvefit`) is what users write in their TOML config:

```toml
[Fitting.solver]
type = "gpufit_curvefit"
```
