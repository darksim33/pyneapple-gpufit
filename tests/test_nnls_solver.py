"""Tests for GpuNNLSSolver stub."""

from __future__ import annotations

import numpy as np
import pytest


class TestGpuNNLSSolver:
    """Tests for the GpuNNLSSolver not-implemented stub."""

    @pytest.fixture
    def nnls_model(self):
        """NNLSModel for constructing GpuNNLSSolver."""
        from pyneapple.models.nnls import NNLSModel

        return NNLSModel(d_range=(1e-4, 0.1), n_bins=50)

    @pytest.fixture
    def solver(self, nnls_model):
        """GpuNNLSSolver instance."""
        from pyneapple_gpufit import GpuNNLSSolver

        return GpuNNLSSolver(model=nnls_model)

    def test_fit_raises_not_implemented(self, solver):
        """fit() always raises NotImplementedError."""
        xdata = np.array([0, 100, 200, 400, 800], dtype=np.float64)
        ydata = np.ones((5, 5), dtype=np.float64)
        with pytest.raises(NotImplementedError, match="pygpufit"):
            solver.fit(xdata, ydata)

    def test_solver_is_base_solver_subclass(self, solver):
        """GpuNNLSSolver is a subclass of BaseSolver."""
        from pyneapple.solvers.base import BaseSolver

        assert isinstance(solver, BaseSolver)

    def test_get_params_raises_before_fit(self, solver):
        """get_params() raises RuntimeError because fit() was never called."""
        with pytest.raises(RuntimeError):
            solver.get_params()

    def test_get_diagnostics_raises_before_fit(self, solver):
        """get_diagnostics() raises RuntimeError because fit() was never called."""
        with pytest.raises(RuntimeError):
            solver.get_diagnostics()
