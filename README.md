# pyneapple-gpufit

> **TL;DR** — A [Pyneapple](https://github.com/darksim33/Pyneapple) plugin that provides GPU-accelerated curve fitting via a custom build of [Gpufit](https://github.com/darksim33/GPUfit). It registers two solvers (`gpufit_curvefit`, `gpufit_nnls`) into Pyneapple's entry-point system and bundles the native CUDA library inside the wheel — no separate toolkit installation required.

---

## Requirements

- Python ≥ 3.12
- An NVIDIA GPU with a CUDA-compatible driver
- [Pyneapple](https://github.com/darksim33/Pyneapple)

## Installation

```bash
# pip
pip install git+https://github.com/darksim33/pyneapple-gpufit.git

# uv
uv add git+https://github.com/darksim33/pyneapple-gpufit
```

The wheel includes `Gpufit.dll` (Windows) and `libGpufit.so` (Linux). Only an up-to-date NVIDIA driver is needed — no CUDA toolkit. Currently the supplied libraries are only compiled for x86 systems. For other systems custom compiles are needed. For detailed instructions on how to compile [see](https://github.com/darksim33/GPUfit).

## Quick start

```python
from pyneapple.models.biexp import BiExpModel
from pyneapple_gpufit import GpuCurveFitSolver
import numpy as np

b_values = np.array([0, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1200])

model = BiExpModel(fit_reduced=True)   # 3-parameter IVIM: f1, D1, D2

solver = GpuCurveFitSolver(
    model=model,
    p0={"f1": 0.2, "D1": 0.01, "D2": 0.001},
    bounds={"f1": (0.0, 1.0), "D1": (1e-4, 0.1), "D2": (1e-5, 0.01)},
)

# ydata shape: (n_pixels, n_b_values) or (n_b_values,) for a single voxel
solver.fit(b_values, ydata)

params = solver.get_params()        # {"f1": ..., "D1": ..., "D2": ...}
diag   = solver.get_diagnostics()  # {"states": ..., "chi_squares": ..., ...}
```

## TOML configuration

```toml
[Fitting]
model = "BiExp"
fit_reduced = true

[Fitting.solver]
type     = "gpufit_curvefit"
max_iter = 500
tol      = 1e-4

[Fitting.solver.p0]
f1 = 0.2
D1 = 0.010
D2 = 0.001

[Fitting.solver.bounds]
f1 = [0.0,  1.0 ]
D1 = [1e-4, 0.1 ]
D2 = [1e-5, 0.01]
```

## Supported models

| Pyneapple model | `fit_reduced` | GPU kernel | Parameters |
|---|---|---|---|
| `MonoExpModel` | — | `MONOEXP` | `S0`, `D` |
| `BiExpModel` | `True` (default) | `BIEXP_RED` | `f1`, `D1`, `D2` |
| `BiExpModel` | `False` | `BIEXP` | `f1`, `D1`, `f2`, `D2` |
| `TriExpModel` | `True` (default) | `TRIEXP_RED` | `f1`, `D1`, `f2`, `D2`, `D3` |
| `TriExpModel` | `False` | `TRIEXP` | `f1`, `D1`, `f2`, `D2`, `f3`, `D3` |

Models with T1 correction (`fit_t1=True`) or explicit S0 fitting (`fit_s0=True`) are not currently supported — use Pyneapple's CPU `CurveFitSolver` instead.

## Citing

The GPU fitting engine is based on Gpufit. If you use `pyneapple-gpufit` in published work, cite:

> Przybylski, A., Throm, B., Kaderali, L. & Grüll, H.\
> **Gpufit: An open-source toolkit for GPU-accelerated curve fitting.**\
> *Scientific Reports* **7**, 15722 (2017).\
> <https://doi.org/10.1038/s41598-017-15313-9>

The CUDA kernels used by this plugin are adapted from [darksim33/GPUfit](https://github.com/darksim33/GPUfit/tree/main), a fork of the [upstream Gpufit library](https://github.com/gpufit/Gpufit) that extends it with diffusion MRI models (`BIEXP_RED`, `TRIEXP_RED`, `MONOEXP_RED`, and their T1/S0 correction variants).

## Development

```bash
# clone and install in editable mode (requires uv)
git clone https://github.com/darksim33/pyneapple-gpufit
cd pyneapple-gpufit
uv sync --dev
uv run pytest tests
```

See [docs/](docs/) for the full API reference and implementation notes.

## License

GPL-3.0-or-later — see [`LICENSE`](LICENSE) for details.
