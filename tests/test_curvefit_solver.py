"""Tests for GpuCurveFitSolver."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_cuda


class TestGpuCurveFitSolverInit:
    """Tests for GpuCurveFitSolver constructor validation."""

    def test_raises_import_error_when_cuda_unavailable(
        self, biexp_reduced_model, biexp_reduced_p0, biexp_reduced_bounds, mocker
    ):
        """Constructor raises RuntimeError when CUDA is not available."""
        mocker.patch(
            "pyneapple_gpufit.curvefit_solver.require_cuda",
            side_effect=RuntimeError("CUDA is not available"),
        )
        from pyneapple_gpufit import GpuCurveFitSolver

        with pytest.raises(RuntimeError, match="CUDA is not available"):
            GpuCurveFitSolver(
                model=biexp_reduced_model,
                max_iter=100,
                tol=1e-4,
                p0=biexp_reduced_p0,
                bounds=biexp_reduced_bounds,
            )

    def test_raises_value_error_for_missing_p0_keys(
        self, biexp_reduced_model, biexp_reduced_bounds, mocker
    ):
        """Constructor raises ValueError when p0 is missing required parameter keys."""
        mocker.patch("pyneapple_gpufit.curvefit_solver.require_cuda")
        from pyneapple_gpufit import GpuCurveFitSolver

        incomplete_p0 = {"f1": 0.2}  # missing D1, D2
        with pytest.raises(ValueError, match="missing required parameter"):
            GpuCurveFitSolver(
                model=biexp_reduced_model,
                max_iter=100,
                tol=1e-4,
                p0=incomplete_p0,
                bounds=biexp_reduced_bounds,
            )

    def test_raises_value_error_for_extra_p0_keys(
        self, biexp_reduced_model, biexp_reduced_bounds, mocker
    ):
        """Constructor raises ValueError when p0 contains unknown parameter keys."""
        mocker.patch("pyneapple_gpufit.curvefit_solver.require_cuda")
        from pyneapple_gpufit import GpuCurveFitSolver

        bad_p0 = {"f1": 0.2, "D1": 0.01, "D2": 0.001, "S0": 1.0}  # extra key
        with pytest.raises(ValueError, match="unknown parameter"):
            GpuCurveFitSolver(
                model=biexp_reduced_model,
                max_iter=100,
                tol=1e-4,
                p0=bad_p0,
                bounds=biexp_reduced_bounds,
            )

    def test_raises_value_error_for_missing_bounds_keys(
        self, biexp_reduced_model, biexp_reduced_p0, mocker
    ):
        """Constructor raises ValueError when bounds is missing required parameter keys."""
        mocker.patch("pyneapple_gpufit.curvefit_solver.require_cuda")
        from pyneapple_gpufit import GpuCurveFitSolver

        incomplete_bounds = {"f1": (0.0, 1.0)}  # missing D1, D2
        with pytest.raises(ValueError, match="missing required parameter"):
            GpuCurveFitSolver(
                model=biexp_reduced_model,
                max_iter=100,
                tol=1e-4,
                p0=biexp_reduced_p0,
                bounds=incomplete_bounds,
            )

    def test_raises_not_implemented_when_p0_is_none(
        self, biexp_reduced_model, biexp_reduced_bounds, mocker
    ):
        """Constructor raises NotImplementedError when p0 is not provided."""
        mocker.patch("pyneapple_gpufit.curvefit_solver.require_cuda")
        from pyneapple_gpufit import GpuCurveFitSolver

        with pytest.raises(NotImplementedError):
            GpuCurveFitSolver(
                model=biexp_reduced_model,
                max_iter=100,
                tol=1e-4,
                p0=None,
                bounds=biexp_reduced_bounds,
            )

    def test_raises_not_implemented_when_bounds_is_none(
        self, biexp_reduced_model, biexp_reduced_p0, mocker
    ):
        """Constructor raises NotImplementedError when bounds is not provided."""
        mocker.patch("pyneapple_gpufit.curvefit_solver.require_cuda")
        from pyneapple_gpufit import GpuCurveFitSolver

        with pytest.raises(NotImplementedError):
            GpuCurveFitSolver(
                model=biexp_reduced_model,
                max_iter=100,
                tol=1e-4,
                p0=biexp_reduced_p0,
                bounds=None,
            )

    def test_stores_config_on_valid_init(
        self, biexp_reduced_model, biexp_reduced_p0, biexp_reduced_bounds, mocker
    ):
        """Constructor stores model, p0, bounds, and hyperparams correctly."""
        mocker.patch("pyneapple_gpufit.curvefit_solver.require_cuda")
        from pyneapple_gpufit import GpuCurveFitSolver

        solver = GpuCurveFitSolver(
            model=biexp_reduced_model,
            max_iter=150,
            tol=1e-5,
            p0=biexp_reduced_p0,
            bounds=biexp_reduced_bounds,
        )
        assert solver.max_iter == 150
        assert solver.tol == 1e-5
        assert solver.p0 is biexp_reduced_p0
        assert solver.bounds is biexp_reduced_bounds


class TestModelMapping:
    """Tests for the resolve_model_id helper and unsupported model rejection."""

    def test_raises_for_biexp_fit_s0(self, mocker):
        """resolve_model_id raises ValueError for BiExpModel with fit_s0=True."""
        from pyneapple.models.biexp import BiExpModel

        from pyneapple_gpufit._model_mapping import resolve_model_id

        model = BiExpModel(fit_reduced=True, fit_s0=True)
        with pytest.raises(ValueError, match="fit_s0=True"):
            resolve_model_id(model)

    def test_raises_for_triexp_fit_s0(self):
        """resolve_model_id raises ValueError for TriExpModel with fit_s0=True."""
        from pyneapple.models.triexp import TriExpModel

        from pyneapple_gpufit._model_mapping import resolve_model_id

        model = TriExpModel(fit_reduced=True, fit_s0=True)
        with pytest.raises(ValueError, match="fit_s0=True"):
            resolve_model_id(model)

    def test_raises_for_unsupported_model_class(self):
        """resolve_model_id raises ValueError for an unsupported model class."""
        from pyneapple_gpufit._model_mapping import resolve_model_id

        class CustomModel:
            param_names = ["x"]
            fixed_params = {}

        with pytest.raises(ValueError, match="not supported"):
            resolve_model_id(CustomModel())

    def test_biexp_reduced_maps_to_biexp_red(self):
        """BiExpModel(fit_reduced=True) maps to BIEXP_RED with correct param order."""
        from pyneapple.models.biexp import BiExpModel

        from pyneapple_gpufit._model_mapping import resolve_model_id
        from pyneapple_gpufit._vendor.pygpufit.gpufit import ModelID

        model = BiExpModel(fit_reduced=True)
        model_id, params = resolve_model_id(model)
        assert model_id == ModelID.BIEXP_RED
        assert params == ["f1", "D1", "D2"]

    def test_biexp_full_maps_to_biexp(self):
        """BiExpModel(fit_reduced=False) maps to BIEXP with correct param order."""
        from pyneapple.models.biexp import BiExpModel

        from pyneapple_gpufit._model_mapping import resolve_model_id
        from pyneapple_gpufit._vendor.pygpufit.gpufit import ModelID

        model = BiExpModel(fit_reduced=False)
        model_id, params = resolve_model_id(model)
        assert model_id == ModelID.BIEXP
        assert params == ["f1", "D1", "f2", "D2"]

    def test_monoexp_maps_to_monoexp(self):
        """MonoExpModel maps to MONOEXP with [S0, D]."""
        from pyneapple.models.monoexp import MonoExpModel

        from pyneapple_gpufit._model_mapping import resolve_model_id
        from pyneapple_gpufit._vendor.pygpufit.gpufit import ModelID

        model = MonoExpModel()
        model_id, params = resolve_model_id(model)
        assert model_id == ModelID.MONOEXP
        assert params == ["S0", "D"]


class TestGpuCurveFitSolverFit:
    """Tests for GpuCurveFitSolver.fit() method."""

    @pytest.fixture
    def solver(
        self, biexp_reduced_model, biexp_reduced_p0, biexp_reduced_bounds, mocker
    ):
        """GpuCurveFitSolver with mocked CUDA check."""
        mocker.patch("pyneapple_gpufit.curvefit_solver.require_cuda")
        from pyneapple_gpufit import GpuCurveFitSolver

        return GpuCurveFitSolver(
            model=biexp_reduced_model,
            max_iter=100,
            tol=1e-4,
            p0=biexp_reduced_p0,
            bounds=biexp_reduced_bounds,
        )

    def test_raises_not_implemented_for_pixel_fixed_params(
        self, solver, b_values, synthetic_biexp_signal, mocker
    ):
        """fit() raises NotImplementedError when pixel_fixed_params is provided."""
        signal, _ = synthetic_biexp_signal
        mocker.patch(
            "pyneapple_gpufit.curvefit_solver.fit_constrained",
        )
        with pytest.raises(NotImplementedError, match="pixel_fixed_params"):
            solver.fit(
                b_values,
                signal,
                pixel_fixed_params={"T1": np.array([800.0])},
            )

    def test_fit_calls_reset_state(
        self, solver, b_values, synthetic_biexp_signal, mocker
    ):
        """fit() calls _reset_state() before fitting."""
        signal, _ = synthetic_biexp_signal

        n_params = 3
        fake_parameters = np.array([[0.3, 0.012, 0.0012]], dtype=np.float32)
        fake_states = np.array([0], dtype=np.int32)
        fake_chi = np.array([0.001], dtype=np.float32)
        fake_iters = np.array([15], dtype=np.int32)

        mocker.patch(
            "pyneapple_gpufit.curvefit_solver.fit_constrained",
            return_value=(fake_parameters, fake_states, fake_chi, fake_iters, 0.01),
        )

        spy = mocker.spy(solver, "_reset_state")
        solver.fit(b_values, signal)
        spy.assert_called_once()

    def test_fit_populates_params_dict(
        self, solver, b_values, synthetic_biexp_signal, mocker
    ):
        """fit() populates params_ with correct keys after a successful call."""
        signal, _ = synthetic_biexp_signal

        fake_parameters = np.array([[0.3, 0.012, 0.0012]], dtype=np.float32)
        fake_states = np.array([0], dtype=np.int32)
        fake_chi = np.array([0.001], dtype=np.float32)
        fake_iters = np.array([15], dtype=np.int32)

        mocker.patch(
            "pyneapple_gpufit.curvefit_solver.fit_constrained",
            return_value=(fake_parameters, fake_states, fake_chi, fake_iters, 0.01),
        )

        solver.fit(b_values, signal)
        params = solver.get_params()
        assert set(params.keys()) == {"f1", "D1", "D2"}

    def test_fit_returns_float_for_single_pixel(
        self, solver, b_values, synthetic_biexp_signal, mocker
    ):
        """fit() stores float values in params_ when fitting a single pixel."""
        signal, _ = synthetic_biexp_signal

        fake_parameters = np.array([[0.3, 0.012, 0.0012]], dtype=np.float32)
        mocker.patch(
            "pyneapple_gpufit.curvefit_solver.fit_constrained",
            return_value=(
                fake_parameters,
                np.array([0], dtype=np.int32),
                np.array([0.001], dtype=np.float32),
                np.array([15], dtype=np.int32),
                0.01,
            ),
        )

        solver.fit(b_values, signal)
        params = solver.get_params()
        for val in params.values():
            assert isinstance(val, float), f"Expected float, got {type(val)}"

    def test_fit_returns_arrays_for_multi_pixel(
        self, solver, b_values, synthetic_biexp_signal_batch, mocker
    ):
        """fit() stores np.ndarray values in params_ when fitting multiple pixels."""
        signals, _ = synthetic_biexp_signal_batch
        n_pixels = signals.shape[0]

        fake_parameters = np.random.rand(n_pixels, 3).astype(np.float32)
        mocker.patch(
            "pyneapple_gpufit.curvefit_solver.fit_constrained",
            return_value=(
                fake_parameters,
                np.zeros(n_pixels, dtype=np.int32),
                np.zeros(n_pixels, dtype=np.float32),
                np.full(n_pixels, 15, dtype=np.int32),
                0.05,
            ),
        )

        solver.fit(b_values, signals)
        params = solver.get_params()
        for name, val in params.items():
            assert isinstance(val, np.ndarray), f"{name}: expected ndarray"
            assert val.shape == (n_pixels,), f"{name}: expected shape ({n_pixels},)"

    def test_fit_populates_diagnostics(
        self, solver, b_values, synthetic_biexp_signal, mocker
    ):
        """fit() populates diagnostics_ with states, chi_squares, number_iterations."""
        signal, _ = synthetic_biexp_signal

        mocker.patch(
            "pyneapple_gpufit.curvefit_solver.fit_constrained",
            return_value=(
                np.array([[0.3, 0.012, 0.0012]], dtype=np.float32),
                np.array([0], dtype=np.int32),
                np.array([0.001], dtype=np.float32),
                np.array([15], dtype=np.int32),
                0.01,
            ),
        )

        solver.fit(b_values, signal)
        diag = solver.get_diagnostics()
        assert "states" in diag
        assert "chi_squares" in diag
        assert "number_iterations" in diag
        assert "execution_time" in diag
        assert "n_pixels" in diag
        assert diag["n_pixels"] == 1

    def test_get_params_raises_before_fit(self, solver):
        """get_params() raises RuntimeError when fit() has not been called."""
        with pytest.raises(RuntimeError):
            solver.get_params()

    def test_get_diagnostics_raises_before_fit(self, solver):
        """get_diagnostics() raises RuntimeError when fit() has not been called."""
        with pytest.raises(RuntimeError):
            solver.get_diagnostics()

    def test_fit_raises_for_shape_mismatch(self, solver, b_values, mocker):
        """fit() raises ValueError when xdata and ydata lengths do not match."""
        mocker.patch("pyneapple_gpufit.curvefit_solver.fit_constrained")
        bad_ydata = np.ones(len(b_values) + 3)  # wrong size
        with pytest.raises(ValueError, match="does not match"):
            solver.fit(b_values, bad_ydata)


# ── GPU integration tests ─────────────────────────────────────────────────────


@requires_cuda
class TestGpuCurveFitSolverIntegration:
    """Integration tests that require an actual NVIDIA GPU."""

    @pytest.fixture
    def solver(self, biexp_reduced_model, biexp_reduced_p0, biexp_reduced_bounds):
        """GpuCurveFitSolver with real CUDA."""
        from pyneapple_gpufit import GpuCurveFitSolver

        return GpuCurveFitSolver(
            model=biexp_reduced_model,
            max_iter=250,
            tol=1e-4,
            p0=biexp_reduced_p0,
            bounds=biexp_reduced_bounds,
        )

    def test_recovers_biexp_parameters_single_pixel(
        self, solver, b_values, synthetic_biexp_signal
    ):
        """Solver recovers biexp parameters within 10% for a noise-free single pixel."""
        signal, true_params = synthetic_biexp_signal
        solver.fit(b_values, signal)
        params = solver.get_params()
        for name, true_val in true_params.items():
            fitted_val = params[name]
            assert (
                abs(fitted_val - true_val) / true_val < 0.10
            ), f"{name}: fitted={fitted_val:.6f}, true={true_val:.6f}"

    def test_recovers_biexp_parameters_batch(
        self, solver, b_values, synthetic_biexp_signal_batch
    ):
        """Solver recovers biexp parameters within 10% for a 10-pixel batch."""
        signals, true_params = synthetic_biexp_signal_batch
        solver.fit(b_values, signals)
        params = solver.get_params()
        n_pixels = signals.shape[0]
        for name in ["f1", "D1", "D2"]:
            assert params[name].shape == (
                n_pixels,
            ), f"{name}: expected shape ({n_pixels},), got {params[name].shape}"
        rel_err_f1 = np.abs(params["f1"] - true_params["f1"]) / true_params["f1"]
        assert (
            np.mean(rel_err_f1) < 0.10
        ), f"Mean relative error for f1 exceeds 10%: {np.mean(rel_err_f1):.3f}"

    def test_diagnostics_populated_after_fit(
        self, solver, b_values, synthetic_biexp_signal
    ):
        """diagnostics_ contains states and chi_squares after a successful GPU fit."""
        signal, _ = synthetic_biexp_signal
        solver.fit(b_values, signal)
        diag = solver.get_diagnostics()
        assert (
            diag["states"][0] == 0
        ), f"Expected convergence (state=0), got state={diag['states'][0]}"
        assert (
            diag["chi_squares"][0] < 1e-3
        ), f"chi_square unexpectedly large: {diag['chi_squares'][0]}"
