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


# ── _bounds_type helper ───────────────────────────────────────────────────────


class TestBoundsType:
    """Tests for the _bounds_type classification helper."""

    @pytest.fixture
    def solver(self, biexp_reduced_model, biexp_reduced_p0, biexp_reduced_bounds, mocker):
        """GpuCurveFitSolver with mocked CUDA check."""
        mocker.patch("pyneapple_gpufit.curvefit_solver.require_cuda")
        from pyneapple_gpufit import GpuCurveFitSolver

        return GpuCurveFitSolver(
            model=biexp_reduced_model,
            p0=biexp_reduced_p0,
            bounds=biexp_reduced_bounds,
        )

    def test_returns_scalar_for_dict_with_float_tuples(
        self, solver, biexp_reduced_bounds
    ):
        """_bounds_type returns 'scalar' for a dict with (float, float) values."""
        assert solver._bounds_type(biexp_reduced_bounds) == "scalar"

    def test_returns_ndarray_for_tuple_of_ndarrays(self, solver):
        """_bounds_type returns 'ndarray' for a (np.ndarray, np.ndarray) tuple."""
        lower = np.array([0.0, 0.0001, 0.00001], dtype=np.float32)
        upper = np.array([1.0, 0.1, 0.01], dtype=np.float32)
        assert solver._bounds_type((lower, upper)) == "ndarray"

    def test_raises_type_error_for_dict_with_non_string_key(self, solver):
        """_bounds_type raises TypeError when a dict key is not a string."""
        with pytest.raises(TypeError):
            solver._bounds_type({1: (0.0, 1.0)})

    def test_raises_type_error_for_dict_with_non_tuple_value(self, solver):
        """_bounds_type raises TypeError when a dict value is not a tuple."""
        with pytest.raises(TypeError):
            solver._bounds_type({"f1": 0.5})

    def test_raises_type_error_for_tuple_first_element_not_ndarray(self, solver):
        """_bounds_type raises TypeError when bounds[0] is not a numpy array."""
        with pytest.raises(TypeError):
            solver._bounds_type((0.5, np.array([1.0])))

    def test_raises_type_error_for_tuple_second_element_not_ndarray(self, solver):
        """_bounds_type raises TypeError when bounds[1] is not a numpy array."""
        with pytest.raises(TypeError):
            solver._bounds_type((np.array([0.0]), 0.5))

    def test_raises_type_error_for_non_dict_non_tuple(self, solver):
        """_bounds_type raises TypeError for inputs that are not dict or tuple."""
        with pytest.raises(TypeError):
            solver._bounds_type("bad_input")


# ── _validate_p0_and_bounds internals ────────────────────────────────────────


class TestValidateP0AndBounds:
    """Tests for _validate_p0_and_bounds — shape handling and constraint interleaving."""

    PARAM_NAMES = ["f1", "D1", "D2"]

    @pytest.fixture
    def solver(self, biexp_reduced_model, biexp_reduced_p0, biexp_reduced_bounds, mocker):
        """GpuCurveFitSolver with mocked CUDA check."""
        mocker.patch("pyneapple_gpufit.curvefit_solver.require_cuda")
        from pyneapple_gpufit import GpuCurveFitSolver

        return GpuCurveFitSolver(
            model=biexp_reduced_model,
            p0=biexp_reduced_p0,
            bounds=biexp_reduced_bounds,
        )

    # ── scalar dict paths ─────────────────────────────────────────────────────

    def test_scalar_p0_and_bounds_output_shapes(
        self, solver, biexp_reduced_p0, biexp_reduced_bounds
    ):
        """Scalar dict p0 and bounds produce (n_pixels, n_params) and (n_pixels, 2*n_params) arrays."""
        n_pixels = 5
        initial_parameters, constraints = solver._validate_p0_and_bounds(
            biexp_reduced_p0, biexp_reduced_bounds, self.PARAM_NAMES, n_pixels
        )
        assert initial_parameters.shape == (n_pixels, 3), (
            f"Expected ({n_pixels}, 3), got {initial_parameters.shape}"
        )
        assert constraints.shape == (n_pixels, 6), (
            f"Expected ({n_pixels}, 6), got {constraints.shape}"
        )

    def test_scalar_bounds_constraints_interleaved_correctly(
        self, solver, biexp_reduced_p0, biexp_reduced_bounds
    ):
        """Scalar bounds produce constraints interleaved as [lo_0, hi_0, lo_1, hi_1, ...]."""
        _, constraints = solver._validate_p0_and_bounds(
            biexp_reduced_p0, biexp_reduced_bounds, self.PARAM_NAMES, 1
        )
        # biexp_reduced_bounds: f1=(0.0, 1.0), D1=(0.0001, 0.1), D2=(0.00001, 0.01)
        assert constraints[0, 0] == pytest.approx(0.0)       # lo_f1
        assert constraints[0, 1] == pytest.approx(1.0)       # hi_f1
        assert constraints[0, 2] == pytest.approx(0.0001)    # lo_D1
        assert constraints[0, 3] == pytest.approx(0.1)       # hi_D1
        assert constraints[0, 4] == pytest.approx(0.00001, rel=1e-4)  # lo_D2
        assert constraints[0, 5] == pytest.approx(0.01)      # hi_D2

    # ── ndarray p0 paths ──────────────────────────────────────────────────────

    def test_ndarray_p0_n_pixels_n_params_shape_accepted(
        self, solver, biexp_reduced_bounds
    ):
        """p0 array shaped (n_pixels, n_params) is accepted without transposition."""
        n_pixels = 5
        p0 = np.tile([0.2, 0.01, 0.001], (n_pixels, 1)).astype(np.float32)
        initial_parameters, _ = solver._validate_p0_and_bounds(
            p0, biexp_reduced_bounds, self.PARAM_NAMES, n_pixels
        )
        assert initial_parameters.shape == (n_pixels, 3)

    def test_ndarray_p0_n_params_n_pixels_shape_transposed(
        self, solver, biexp_reduced_bounds
    ):
        """p0 array shaped (n_params, n_pixels) is transposed to (n_pixels, n_params)."""
        n_pixels, n_params = 4, 3
        # p0[param_idx, pixel_idx]: each column is one pixel's starting values
        p0 = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],        # f1 for pixels 0–3
                [0.01, 0.02, 0.03, 0.04],    # D1 for pixels 0–3
                [0.001, 0.002, 0.003, 0.004],  # D2 for pixels 0–3
            ],
            dtype=np.float32,
        )  # shape (3, 4) = (n_params, n_pixels)
        initial_parameters, _ = solver._validate_p0_and_bounds(
            p0, biexp_reduced_bounds, self.PARAM_NAMES, n_pixels
        )
        assert initial_parameters.shape == (n_pixels, n_params)
        # pixel 2: f1=0.3, D1=0.03, D2=0.003 — verifies transpose, not reshape
        np.testing.assert_allclose(
            initial_parameters[2],
            [0.3, 0.03, 0.003],
            rtol=1e-5,
            err_msg="p0 not correctly transposed from (n_params, n_pixels) to (n_pixels, n_params)",
        )

    def test_ndarray_p0_wrong_shape_raises_value_error(
        self, solver, biexp_reduced_bounds
    ):
        """p0 array with incompatible shape raises ValueError."""
        p0 = np.ones((7, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="p0 shape"):
            solver._validate_p0_and_bounds(
                p0, biexp_reduced_bounds, self.PARAM_NAMES, 5
            )

    # ── tuple ndarray bounds paths ────────────────────────────────────────────

    def test_tuple_bounds_n_pixels_n_params_shape_accepted(
        self, solver, biexp_reduced_p0
    ):
        """Tuple bounds shaped (n_pixels, n_params) are accepted without transposition."""
        n_pixels, n_params = 5, 3
        lower = np.tile([0.0, 0.0001, 0.00001], (n_pixels, 1)).astype(np.float32)
        upper = np.tile([1.0, 0.1, 0.01], (n_pixels, 1)).astype(np.float32)
        _, constraints = solver._validate_p0_and_bounds(
            biexp_reduced_p0, (lower, upper), self.PARAM_NAMES, n_pixels
        )
        assert constraints.shape == (n_pixels, 2 * n_params), (
            f"Expected ({n_pixels}, {2 * n_params}), got {constraints.shape}"
        )

    def test_tuple_bounds_n_params_n_pixels_shape_transposed(
        self, solver, biexp_reduced_p0
    ):
        """Tuple bounds shaped (n_params, n_pixels) are transposed to (n_pixels, n_params)."""
        n_pixels, n_params = 4, 3
        lower = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],              # lo_f1 pixels 0–3
                [0.001, 0.002, 0.003, 0.004],       # lo_D1
                [0.0001, 0.0002, 0.0003, 0.0004],   # lo_D2
            ],
            dtype=np.float32,
        )  # shape (3, 4) = (n_params, n_pixels)
        upper = np.array(
            [
                [1.0, 0.9, 0.8, 0.7],
                [0.1, 0.09, 0.08, 0.07],
                [0.01, 0.009, 0.008, 0.007],
            ],
            dtype=np.float32,
        )
        _, constraints = solver._validate_p0_and_bounds(
            biexp_reduced_p0, (lower, upper), self.PARAM_NAMES, n_pixels
        )
        assert constraints.shape == (n_pixels, 2 * n_params)
        # pixel 2: lo_f1=0.0, hi_f1=0.8 — verifies transpose, not reshape
        assert constraints[2, 0] == pytest.approx(0.0)   # lo_f1 pixel 2
        assert constraints[2, 1] == pytest.approx(0.8)   # hi_f1 pixel 2
        assert constraints[2, 2] == pytest.approx(0.003) # lo_D1 pixel 2

    def test_tuple_bounds_wrong_shape_raises_value_error(
        self, solver, biexp_reduced_p0
    ):
        """Tuple bounds with incompatible shape raise ValueError."""
        lower = np.ones((7, 4), dtype=np.float32)
        upper = np.ones((7, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Bounds shape"):
            solver._validate_p0_and_bounds(
                biexp_reduced_p0, (lower, upper), self.PARAM_NAMES, 5
            )

    def test_tuple_bounds_constraints_correctly_interleaved(
        self, solver, biexp_reduced_p0
    ):
        """Per-pixel tuple bounds produce constraints interleaved as [lo_0, hi_0, ...] per row."""
        n_pixels = 2
        lower = np.array(
            [[0.0, 0.0001, 0.00001], [0.05, 0.0002, 0.00002]], dtype=np.float32
        )
        upper = np.array(
            [[1.0, 0.1, 0.01], [0.9, 0.08, 0.009]], dtype=np.float32
        )
        _, constraints = solver._validate_p0_and_bounds(
            biexp_reduced_p0, (lower, upper), self.PARAM_NAMES, n_pixels
        )
        assert constraints.shape == (2, 6)
        # pixel 0
        assert constraints[0, 0] == pytest.approx(lower[0, 0])  # lo_f1
        assert constraints[0, 1] == pytest.approx(upper[0, 0])  # hi_f1
        assert constraints[0, 2] == pytest.approx(lower[0, 1])  # lo_D1
        assert constraints[0, 3] == pytest.approx(upper[0, 1])  # hi_D1
        # pixel 1
        assert constraints[1, 0] == pytest.approx(lower[1, 0])  # lo_f1
        assert constraints[1, 1] == pytest.approx(upper[1, 0])  # hi_f1


# ── fit() with tuple bounds and ndarray p0 ────────────────────────────────────


class TestGpuCurveFitSolverFitNdarrayInputs:
    """Tests for fit() accepting per-pixel ndarray p0 and tuple bounds."""

    @pytest.fixture
    def solver(self, biexp_reduced_model, biexp_reduced_p0, biexp_reduced_bounds, mocker):
        """GpuCurveFitSolver with mocked CUDA check."""
        mocker.patch("pyneapple_gpufit.curvefit_solver.require_cuda")
        from pyneapple_gpufit import GpuCurveFitSolver

        return GpuCurveFitSolver(
            model=biexp_reduced_model,
            p0=biexp_reduced_p0,
            bounds=biexp_reduced_bounds,
        )

    def _fake_fit_result(self, n_pixels: int):
        return (
            np.random.default_rng(0).random((n_pixels, 3)).astype(np.float32),
            np.zeros(n_pixels, dtype=np.int32),
            np.zeros(n_pixels, dtype=np.float32),
            np.full(n_pixels, 10, dtype=np.int32),
            0.05,
        )

    def test_fit_with_tuple_ndarray_bounds_multi_pixel(
        self,
        solver,
        b_values,
        synthetic_biexp_signal_batch,
        biexp_reduced_tuple_bounds_batch,
        mocker,
    ):
        """fit() accepts per-pixel (lower, upper) tuple bounds at call time."""
        signals, _ = synthetic_biexp_signal_batch
        n_pixels = signals.shape[0]
        mocker.patch(
            "pyneapple_gpufit.curvefit_solver.fit_constrained",
            return_value=self._fake_fit_result(n_pixels),
        )
        solver.fit(b_values, signals, bounds=biexp_reduced_tuple_bounds_batch)
        params = solver.get_params()
        assert set(params.keys()) == {"f1", "D1", "D2"}

    def test_fit_ndarray_p0_n_pixels_n_params_multi_pixel(
        self,
        solver,
        b_values,
        synthetic_biexp_signal_batch,
        biexp_reduced_ndarray_p0_batch,
        mocker,
    ):
        """fit() accepts p0 as a (n_pixels, n_params) ndarray at call time."""
        signals, _ = synthetic_biexp_signal_batch
        n_pixels = signals.shape[0]
        mocker.patch(
            "pyneapple_gpufit.curvefit_solver.fit_constrained",
            return_value=self._fake_fit_result(n_pixels),
        )
        solver.fit(b_values, signals, p0=biexp_reduced_ndarray_p0_batch)
        params = solver.get_params()
        assert set(params.keys()) == {"f1", "D1", "D2"}

    def test_fit_p0_dict_override_at_call_time(
        self, solver, b_values, synthetic_biexp_signal, mocker
    ):
        """fit() passes the call-time p0 dict to _validate_p0_and_bounds, not the constructor default."""
        signal, _ = synthetic_biexp_signal
        override_p0 = {"f1": 0.4, "D1": 0.015, "D2": 0.0015}
        mocker.patch(
            "pyneapple_gpufit.curvefit_solver.fit_constrained",
            return_value=self._fake_fit_result(1),
        )
        spy = mocker.spy(solver, "_validate_p0_and_bounds")
        solver.fit(b_values, signal, p0=override_p0)
        called_p0 = spy.call_args[0][0]
        assert called_p0 == override_p0, (
            f"Expected override p0 {override_p0} to be passed, got {called_p0}"
        )

    def test_fit_bounds_dict_override_at_call_time(
        self, solver, b_values, synthetic_biexp_signal, mocker
    ):
        """fit() passes the call-time bounds dict to _validate_p0_and_bounds, not the constructor default."""
        signal, _ = synthetic_biexp_signal
        override_bounds = {"f1": (0.1, 0.9), "D1": (0.001, 0.05), "D2": (0.0001, 0.005)}
        mocker.patch(
            "pyneapple_gpufit.curvefit_solver.fit_constrained",
            return_value=self._fake_fit_result(1),
        )
        spy = mocker.spy(solver, "_validate_p0_and_bounds")
        solver.fit(b_values, signal, bounds=override_bounds)
        called_bounds = spy.call_args[0][1]
        assert called_bounds == override_bounds, (
            f"Expected override bounds {override_bounds} to be passed, got {called_bounds}"
        )

    def test_fit_tuple_bounds_override_at_call_time(
        self,
        solver,
        b_values,
        synthetic_biexp_signal_batch,
        biexp_reduced_tuple_bounds_batch,
        mocker,
    ):
        """fit() passes call-time tuple bounds to _validate_p0_and_bounds, not the constructor default."""
        signals, _ = synthetic_biexp_signal_batch
        n_pixels = signals.shape[0]
        mocker.patch(
            "pyneapple_gpufit.curvefit_solver.fit_constrained",
            return_value=self._fake_fit_result(n_pixels),
        )
        spy = mocker.spy(solver, "_validate_p0_and_bounds")
        solver.fit(b_values, signals, bounds=biexp_reduced_tuple_bounds_batch)
        called_bounds = spy.call_args[0][1]
        assert called_bounds is biexp_reduced_tuple_bounds_batch, (
            "Expected tuple bounds to be forwarded unchanged to _validate_p0_and_bounds"
        )


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
            assert abs(fitted_val - true_val) / true_val < 0.10, (
                f"{name}: fitted={fitted_val:.6f}, true={true_val:.6f}"
            )

    def test_recovers_biexp_parameters_batch(
        self, solver, b_values, synthetic_biexp_signal_batch
    ):
        """Solver recovers biexp parameters within 10% for a 10-pixel batch."""
        signals, true_params = synthetic_biexp_signal_batch
        solver.fit(b_values, signals)
        params = solver.get_params()
        n_pixels = signals.shape[0]
        for name in ["f1", "D1", "D2"]:
            assert params[name].shape == (n_pixels,), (
                f"{name}: expected shape ({n_pixels},), got {params[name].shape}"
            )
        rel_err_f1 = np.abs(params["f1"] - true_params["f1"]) / true_params["f1"]
        assert np.mean(rel_err_f1) < 0.10, (
            f"Mean relative error for f1 exceeds 10%: {np.mean(rel_err_f1):.3f}"
        )

    def test_diagnostics_populated_after_fit(
        self, solver, b_values, synthetic_biexp_signal
    ):
        """diagnostics_ contains states and chi_squares after a successful GPU fit."""
        signal, _ = synthetic_biexp_signal
        solver.fit(b_values, signal)
        diag = solver.get_diagnostics()
        assert diag["states"][0] == 0, (
            f"Expected convergence (state=0), got state={diag['states'][0]}"
        )
        assert diag["chi_squares"][0] < 1e-3, (
            f"chi_square unexpectedly large: {diag['chi_squares'][0]}"
        )
