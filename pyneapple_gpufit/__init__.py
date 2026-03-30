"""pyneapple-gpufit — GPU-accelerated solvers for Pyneapple."""

from __future__ import annotations

from .curvefit_solver import GpuCurveFitSolver
from .nnls_solver import GpuNNLSSolver

__version__ = "0.1.0"

__all__ = ["GpuCurveFitSolver", "GpuNNLSSolver", "__version__"]
