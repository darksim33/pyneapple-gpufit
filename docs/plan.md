# Pyneapple-Gpufit Implementation Plan

> **TL;DR** — A separate Python package providing GPU-accelerated solvers for Pyneapple via pygpufit. Registers through entry_points. CUDA is never imported on machines without NVIDIA GPUs. Covers repo structure, solver implementation, entry_point wiring, testing strategy, and CI.

---

## Prerequisites

The Pyneapple main repo must have entry_point-based plugin discovery implemented first. See the [plugin-discovery plan](../../Pyneapple/docs/plans/plugin-discovery.md) in the main repo.

## Repo structure

```
pyneapple-gpufit/
├── .github/
│   ├── copilot-instructions.md
│   └── skills/
│       ├── pyneapple-api/SKILL.md
│       ├── testing-guidelines/SKILL.md
│       └── docs-style-guide/SKILL.md
├── docs/
│   └── plan.md                       # this file
├── src/
│   └── pyneapple_gpufit/
│       ├── __init__.py
│       ├── _cuda.py                  # CUDA availability guard
│       ├── curvefit_solver.py        # GpuCurveFitSolver
│       └── nnls_solver.py           # GpuNNLSSolver (or stub)
├── tests/
│   ├── conftest.py                  # requires_cuda marker, shared fixtures
│   ├── test_curvefit_solver.py
│   └── test_nnls_solver.py
├── pyproject.toml
├── README.md
└── LICENSE
```

## Implementation steps

### Step 1 — `pyproject.toml`

```toml
[project]
name = "pyneapple-gpufit"
version = "0.1.0"
description = "GPU-accelerated solvers for Pyneapple using pygpufit."
requires-python = ">=3.12"
license = { text = "GPL-3.0-or-later" }
dependencies = [
    "pyneapple>=1.7.0",
    "pygpufit",
    "numpy>=2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/pyneapple_gpufit"]

[project.entry-points."pyneapple.solvers"]
gpufit_curvefit = "pyneapple_gpufit:GpuCurveFitSolver"
gpufit_nnls = "pyneapple_gpufit:GpuNNLSSolver"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.14",
    "pytest-cov>=6.0",
    "ruff>=0.15",
]
```

### Step 2 — `_cuda.py`

Single function `require_cuda()` that imports `pygpufit.gpufit` and checks `cuda_available()`. Raises `RuntimeError` with a clear message if CUDA is missing. This isolates the only `pygpufit` import to solver instantiation time.

### Step 3 — `GpuCurveFitSolver`

Inherits `pyneapple.solvers.base.BaseSolver`. Constructor mirrors `CurveFitSolver` signature:

- Calls `require_cuda()` in `__init__`
- Accepts `model`, `max_iter`, `tol`, `p0`, `bounds`, `multi_threading`
- Validates `p0` / `bounds` keys against `model.param_names`

`fit()`:

- Calls `self._reset_state()`
- Maps model to gpufit estimator ID or custom model
- Flattens `p0` and `bounds` dicts to contiguous arrays for gpufit
- Calls `gpufit.fit_constrained()` or `gpufit.fit()`
- Reshapes output into `self.params_` dict (same format as `CurveFitSolver`)
- Stores `states`, `chi_squares`, `number_iterations` in `self.diagnostics_`

### Step 4 — `GpuNNLSSolver`

Same approach for distribution-based fitting if pygpufit supports it. Otherwise, a stub that raises `NotImplementedError` — omit from entry_points until ready.

### Step 5 — `__init__.py`

```python
from .curvefit_solver import GpuCurveFitSolver
from .nnls_solver import GpuNNLSSolver

__all__ = ["GpuCurveFitSolver", "GpuNNLSSolver"]
```

### Step 6 — Tests

`conftest.py` defines:

- `_cuda_available()` helper (try/except around `pygpufit` import)
- `requires_cuda` skip marker
- Shared fixtures for synthetic biexp data

Unit tests (always run, mock gpufit):

- Solver init validates p0/bounds
- `require_cuda()` raises `RuntimeError` when CUDA unavailable
- `fit()` calls `_reset_state()` and populates `params_`/`diagnostics_`

Integration tests (`@requires_cuda`):

- Fit synthetic biexp data, verify parameter recovery within tolerance
- Multi-pixel batch fitting, verify output shapes
- Convergence diagnostics populated correctly

### Step 7 — End-to-end verification

1. `pip install -e .` into a Pyneapple development env
2. Create a TOML config with `type = "gpufit_curvefit"`
3. Run `pyneapple-pixelwise --config config.toml ...`
4. Verify the plugin is discovered and fitting completes
