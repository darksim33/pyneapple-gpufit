# API Reference

> **TL;DR** — Complete reference for `GpuCurveFitSolver` and `GpuNNLSSolver`, the two solvers provided by `pyneapple-gpufit`. Both inherit from `pyneapple.solvers.base.BaseSolver` and register as Pyneapple entry-points.

---

## GpuCurveFitSolver

GPU-accelerated Levenberg-Marquardt solver. Wraps the Gpufit CUDA library to fit all pixels in a single batched call.

### Constructor

```python
GpuCurveFitSolver(
    model,
    max_iter=250,
    tol=1e-4,
    p0,
    bounds,
    verbose=False,
    **solver_kwargs,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `ParametricModel` | — | A pyneapple model instance (`MonoExpModel`, `BiExpModel`, `TriExpModel`). |
| `max_iter` | `int` | `250` | Maximum LM iterations per pixel. |
| `tol` | `float` | `1e-4` | Convergence tolerance. |
| `p0` | `dict[str, float]` | — | Initial guess for each free parameter. Keys must match `model.param_names`. |
| `bounds` | `dict[str, tuple[float, float]]` | — | Lower/upper bounds per parameter. Keys must match `model.param_names`. |
| `verbose` | `bool` | `False` | Log convergence statistics per batch via `loguru`. |

**Raises:**

- `ImportError` — if `Gpufit.dll` / `libGpufit.so` cannot be loaded (missing or incompatible driver).
- `RuntimeError` — if no CUDA-capable GPU is detected.
- `ValueError` — if `p0` or `bounds` keys don't match `model.param_names`, or if the model configuration is unsupported.
- `NotImplementedError` — if `p0=None` or `bounds=None` (auto-derivation from model not yet implemented).

---

### `fit()`

```python
solver.fit(xdata, ydata, p0=None, bounds=None, pixel_fixed_params=None)
```

| Parameter | Type | Description |
|---|---|---|
| `xdata` | `np.ndarray` | 1D array of b-values, shape `(n_points,)`. |
| `ydata` | `np.ndarray` | Signal data — shape `(n_pixels, n_points)` or `(n_points,)` for a single voxel. |
| `p0` | `dict[str, float] \| None` | Per-call initial guess; overrides the constructor `p0` when provided. |
| `bounds` | `dict[str, tuple[float, float]] \| None` | Per-call bounds; overrides the constructor `bounds` when provided. |
| `pixel_fixed_params` | `dict \| None` | Not supported in v0.1 — raises `NotImplementedError`. |

Returns `self` to allow chaining.

**Raises:** `NotImplementedError` if `pixel_fixed_params` is provided; `ValueError` if `xdata` and `ydata` sizes are inconsistent.

---

### `get_params()`

Returns fitted parameters as `dict[str, float | np.ndarray]`.

- **Single pixel** — values are `float`.
- **Batch** — values are `np.ndarray` of shape `(n_pixels,)`, dtype `float64`.

**Raises:** `RuntimeError` if called before `fit()`.

---

### `get_diagnostics()`

Returns a `dict` populated after `fit()`:

| Key | Type | Description |
|---|---|---|
| `states` | `np.ndarray[int32]` | Per-pixel convergence state (`0` = converged, `1` = max iterations reached, `2` = singular Hessian). |
| `chi_squares` | `np.ndarray[float32]` | Per-pixel weighted sum of squared residuals at convergence. |
| `number_iterations` | `np.ndarray[int32]` | LM iterations used per pixel. |
| `execution_time` | `float` | Total GPU wall-clock time in seconds. |
| `n_pixels` | `int` | Number of pixels submitted to the GPU. |

**Raises:** `RuntimeError` if called before `fit()`.

---

## GpuNNLSSolver

Placeholder solver registered under the `gpufit_nnls` entry-point key. `fit()` always raises `NotImplementedError` — NNLS on GPU is not available in the current Gpufit library version. The entry-point slot is reserved for a future implementation.

---

## Supported model configurations

| Model | Flag | GPU kernel | Kernel ID | `param_names` |
|---|---|---|---|---|
| `MonoExpModel` | — | `MONOEXP` | 100 | `["S0", "D"]` |
| `BiExpModel` | `fit_reduced=True` | `BIEXP_RED` | 210 | `["f1", "D1", "D2"]` |
| `BiExpModel` | `fit_reduced=False` | `BIEXP` | 200 | `["f1", "D1", "f2", "D2"]` |
| `TriExpModel` | `fit_reduced=True` | `TRIEXP_RED` | 310 | `["f1", "D1", "f2", "D2", "D3"]` |
| `TriExpModel` | `fit_reduced=False` | `TRIEXP` | 300 | `["f1", "D1", "f2", "D2", "f3", "D3"]` |

### Unsupported configurations

`ValueError` is raised for:

- Any model with `model.fixed_params` set — use CPU `CurveFitSolver` instead.
- `BiExpModel` or `TriExpModel` with `fit_t1=True` or `fit_s0=True`.

---

## GPU library

The CUDA kernels are provided by [darksim33/GPUfit](https://github.com/darksim33/GPUfit/tree/main),
a fork of [gpufit/Gpufit](https://github.com/gpufit/Gpufit) that adds diffusion MRI models
(`BIEXP_RED`, `TRIEXP_RED`, `MONOEXP_RED`) and T1/S0 correction variants.
The library is vendored at `pyneapple_gpufit/_vendor/pygpufit/` and bundled inside the wheel.

### Citation

If you use this plugin in published work, cite the Gpufit paper:

> Przybylski, A., Throm, B., Kaderali, L. & Grüll, H.\
> **Gpufit: An open-source toolkit for GPU-accelerated curve fitting.**\
> *Scientific Reports* **7**, 15722 (2017).\
> <https://doi.org/10.1038/s41598-017-15313-9>
