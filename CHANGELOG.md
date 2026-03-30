# Changelog

All notable changes to `pyneapple-gpufit` are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/).

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
