# Changelog

All notable changes to `pyneapple-gpufit` are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/).

---

## [0.3.0] — 2026-05-12

### Added

- `pixel_results_` attribute populated on every `fit()` call, including the single-pixel
  (`ydata.ndim == 1`) path. Each element is a `_PixelFitResult(params, state, chi_sq,
  n_iterations, covariance=None)` instance, satisfying the full `BaseSolver` plugin contract.
- `_GPUFIT_STATE_MESSAGES` module-level constant mapping Gpufit state codes (0–4) to
  human-readable strings; used when `verbose=True` to log per-state convergence counts.
- 16 new mocked unit tests in `TestGpuCurveFitSolverPixelResults` covering length,
  dtypes, state codes, chi-sq, iteration counts, covariance, single-pixel path, and
  multi-pixel heterogeneous states.

### Removed

- `__version__` string from `pyneapple_gpufit.__init__`. It was stale (`"0.1.0"` while
  the package was already at `0.2.0`). Use
  `importlib.metadata.version("pyneapple-gpufit")` to retrieve the version at runtime.

---

## [0.2.0] — 2026-04-28

### Added

- `fit_s0=True` support for `BiExpModel` and `TriExpModel` via the `BIEXP_S0` and `TRIEXP_S0`
  GPU kernels. The GPU kernels place `S0` as the last parameter (`["f1", "D1", "D2", "S0"]`
  and `["f1", "D1", "f2", "D2", "D3", "S0"]`); pyneapple v2.0 may order `S0` differently —
  see the [API reference](docs/api-reference.md) for details.
- Per-pixel `bounds` support in `fit()` via `tuple[np.ndarray, np.ndarray]`. Each array may
  be shaped `(n_pixels, n_params)` or `(n_params, n_pixels)` (transposed automatically).
- Per-pixel `p0` support in `fit()` via `np.ndarray` shaped `(n_pixels, n_params)` or
  `(n_params, n_pixels)`.
- Minimum pyneapple version bumped to `>=2.0.0`.

### Fixed

- Constraint array shape for per-pixel tuple bounds was `(2·n_params, n_pixels)` instead of
  the gpufit-expected `(n_pixels, 2·n_params)`.
- Per-pixel `p0` and `bounds` arrays in `(n_params, n_pixels)` layout were incorrectly
  reshaped (`reshape`) instead of transposed (`.T`), scrambling per-pixel values.
- `_bounds_type` did not validate the second element of a `(lower, upper)` tuple, allowing
  non-ndarray values to pass silently.

---

## [0.1.0] — 2026-03-30

### Added

- `GpuCurveFitSolver` — GPU-accelerated Levenberg-Marquardt solver wrapping the Gpufit CUDA library.
  - Supports `MonoExpModel`, `BiExpModel` (reduced and full), and `TriExpModel` (reduced and full).
  - All pixels submitted in a single batched `fit_constrained()` call for maximum GPU utilization.
  - Constructor validates `p0` and `bounds` key sets against `model.param_names`.
  - `verbose=True` logs per-batch convergence statistics via `loguru`.
- `GpuNNLSSolver` — entry-point stub for a future GPU NNLS implementation; raises `NotImplementedError` on `fit()`.
- Vendored `pygpufit` ctypes wrapper with `Gpufit.dll` (Windows) and `libGpufit.so` (Linux) bundled inside the wheel.
- Entry-points registered in Pyneapple's plugin system: `gpufit_curvefit` → `GpuCurveFitSolver`, `gpufit_nnls` → `GpuNNLSSolver`.
- Full pytest suite: 29 tests covering constructor validation, model mapping, fit contract (mocked GPU), and live GPU integration.

### Fixed

- Updated `ModelID` enum in the vendored `pygpufit` wrapper to match the reorganized `constants.h`
  in the rebuilt Gpufit DLL.  
  The `*_RED` model IDs shifted to a decade-based scheme to make room for new T1/S0 correction variants:

  | Model | Old ID | New ID |
  |---|---|---|
  | `BIEXP_RED` | 201 | 210 |
  | `MONOEXP_RED` | 101 | 110 |
  | `TRIEXP_RED` | 301 | 310 |

  Without this fix, every `BIEXP_RED` call silently dispatched to the `BIEXP_T1` kernel,
  producing `state=2` (singular Hessian) even at exact ground-truth parameters.

- Corrected `dy/da` Jacobian sign in `biexp_red.cuh` in the [GPUfit fork](https://github.com/darksim33/GPUfit):
  `exp(-p[2]·x) − exp(-p[1]·x)` → `exp(-p[1]·x) − exp(-p[2]·x)`.
  The previous sign caused the fraction parameter `f1` to converge to `1 − f1` instead of the correct value.
  The rebuilt DLL incorporating the fix is vendored in this release.
