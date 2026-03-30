"""GPU-accelerated curve-fitting solver via pygpufit."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from pyneapple.solvers.base import BaseSolver

from ._cuda import require_cuda
from ._model_mapping import resolve_model_id
from ._vendor.pygpufit.gpufit import ConstraintType, EstimatorID, fit_constrained


class GpuCurveFitSolver(BaseSolver):
    """GPU-accelerated Levenberg-Marquardt solver using pygpufit.

    Wraps the Gpufit CUDA library to fit exponential models across all pixels
    in a single batched GPU call.  Supports MonoExpModel, BiExpModel, and
    TriExpModel (without T1 correction and without fit_s0 for bi/triexp).

    Args:
        model: A pyneapple ParametricModel instance.
        max_iter: Maximum number of LM iterations per pixel (default 250).
        tol: Convergence tolerance (default 1e-4).
        p0: Initial parameter guesses as ``{param_name: float}``.
        bounds: Parameter bounds as ``{param_name: (lower, upper)}``.
        verbose: Log fitting progress (default False).
        **solver_kwargs: Ignored; accepted for API compatibility.

    Raises:
        ImportError: If the Gpufit native library cannot be loaded.
        RuntimeError: If CUDA is not available on this machine.
        ValueError: If *p0* or *bounds* keys do not match ``model.param_names``.
        ValueError: If the model configuration is not supported by pygpufit.
    """

    def __init__(
        self,
        model: Any,
        max_iter: int = 250,
        tol: float = 1e-4,
        p0: dict[str, float] | None = None,
        bounds: dict[str, tuple[float, float]] | None = None,
        verbose: bool = False,
        **solver_kwargs,
    ):
        super().__init__(model=model, max_iter=max_iter, tol=tol, verbose=verbose)
        require_cuda()
        # Validate model is a supported configuration before storing anything
        resolve_model_id(model)

        if p0 is None:
            raise NotImplementedError(
                "Default p0 from model is not yet implemented. Provide p0 explicitly."
            )
        _validate_param_keys(p0, model.param_names, "p0")
        self.p0 = p0

        if bounds is None:
            raise NotImplementedError(
                "Default bounds from model is not yet implemented. "
                "Provide bounds explicitly."
            )
        _validate_param_keys(bounds, model.param_names, "bounds")
        self.bounds = bounds

    def fit(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: dict[str, float] | None = None,
        bounds: dict[str, tuple[float, float]] | None = None,
        pixel_fixed_params: dict[str, np.ndarray] | None = None,
        **fit_kwargs,
    ) -> "GpuCurveFitSolver":
        """Fit the model to data on GPU using pygpufit.

        Args:
            xdata: 1D array of b-values, shape ``(n_points,)``.
            ydata: Signal data, shape ``(n_pixels, n_points)`` or ``(n_points,)``
                for a single pixel.
            p0: Per-call initial guesses (overrides constructor defaults).
            bounds: Per-call bounds (overrides constructor defaults).
            pixel_fixed_params: Not supported in v0.1 — raises
                ``NotImplementedError`` if provided.
            **fit_kwargs: Accepted for API compatibility; ignored.

        Returns:
            self

        Raises:
            NotImplementedError: If *pixel_fixed_params* is provided.
        """
        self._reset_state()

        if pixel_fixed_params is not None:
            raise NotImplementedError(
                "pixel_fixed_params are not supported by GpuCurveFitSolver "
                "in v0.1. Use CurveFitSolver for T1-map-corrected fitting."
            )

        # ── normalise ydata to 2D ─────────────────────────────────────────────
        single_pixel = ydata.ndim == 1
        if single_pixel:
            ydata = ydata[np.newaxis, :]
        n_pixels, n_points = ydata.shape

        if xdata.shape[0] != n_points:
            raise ValueError(
                f"xdata length {xdata.shape[0]} does not match "
                f"ydata second dimension {n_points}."
            )

        # ── resolve model → ModelID + ordered param names ─────────────────────
        model_id, gpufit_param_names = resolve_model_id(self.model)
        n_params = len(gpufit_param_names)

        # ── effective p0 / bounds (per-call override or constructor default) ───
        effective_p0 = p0 if p0 is not None else self.p0
        effective_bounds = bounds if bounds is not None else self.bounds

        # ── build initial_parameters [n_pixels, n_params] float32 ─────────────
        p0_vec = np.array(
            [effective_p0[name] for name in gpufit_param_names], dtype=np.float32
        )
        initial_parameters = np.tile(p0_vec, (n_pixels, 1)).astype(
            np.float32, copy=False
        )

        # ── build constraints [n_pixels, 2*n_params] float32 ──────────────────
        lower = np.array(
            [effective_bounds[name][0] for name in gpufit_param_names],
            dtype=np.float32,
        )
        upper = np.array(
            [effective_bounds[name][1] for name in gpufit_param_names],
            dtype=np.float32,
        )
        # interleave: [lo_0, hi_0, lo_1, hi_1, ...] shape (2*n_params,)
        constraint_row = np.empty(2 * n_params, dtype=np.float32)
        constraint_row[0::2] = lower
        constraint_row[1::2] = upper
        constraints = np.tile(constraint_row, (n_pixels, 1)).astype(
            np.float32, copy=False
        )  # shape (n_pixels, 2*n_params)
        constraint_types = np.full(
            n_params, ConstraintType.LOWER_UPPER, dtype=np.int32
        )  # shape (n_params,)

        # ── data array ────────────────────────────────────────────────────────
        data = np.ascontiguousarray(
            ydata, dtype=np.float32
        )  # shape (n_pixels, n_points)

        # ── user_info: b-values as float32 byte buffer ───────────────────────
        user_info = np.ascontiguousarray(xdata, dtype=np.float32)  # shape (n_points,)

        if self.verbose:
            logger.info(
                f"GpuCurveFitSolver: fitting {n_pixels} pixel(s) "
                f"with model_id={model_id}, n_params={n_params}, "
                f"max_iter={self.max_iter}, tol={self.tol}"
            )

        # ── GPU fit ───────────────────────────────────────────────────────────
        parameters, states, chi_squares, number_iterations, execution_time = (
            fit_constrained(
                data=data,
                weights=None,
                model_id=model_id,
                initial_parameters=initial_parameters,
                constraints=constraints,
                constraint_types=constraint_types,
                tolerance=float(self.tol),
                max_number_iterations=int(self.max_iter),
                parameters_to_fit=None,
                estimator_id=EstimatorID.LSE,
                user_info=user_info,
            )
        )

        if self.verbose:
            converged = int(np.sum(states == 0))
            logger.info(
                f"GPU fit complete in {execution_time:.3f}s — "
                f"{converged}/{n_pixels} pixels converged."
            )

        # ── store results ─────────────────────────────────────────────────────
        # parameters: [n_pixels, n_params] float32 — reindex to pyneapple order
        # gpufit_param_names == model.param_names for all supported models, so
        # the column index matches directly.
        if single_pixel:
            self.params_ = {
                name: float(parameters[0, i])
                for i, name in enumerate(gpufit_param_names)
            }
        else:
            self.params_ = {
                name: parameters[:, i].astype(np.float64)
                for i, name in enumerate(gpufit_param_names)
            }

        self.diagnostics_ = {
            "states": states,
            "chi_squares": chi_squares,
            "number_iterations": number_iterations,
            "execution_time": execution_time,
            "n_pixels": n_pixels,
        }

        return self


# ── helpers ───────────────────────────────────────────────────────────────────


def _validate_param_keys(
    params: dict,
    param_names: list[str],
    label: str,
) -> None:
    """Raise ValueError if *params* keys do not match *param_names*."""
    missing = set(param_names) - set(params.keys())
    if missing:
        raise ValueError(
            f"{label} is missing required parameter(s): {sorted(missing)}. "
            f"Expected keys: {param_names}"
        )
    extra = set(params.keys()) - set(param_names)
    if extra:
        raise ValueError(
            f"{label} contains unknown parameter(s): {sorted(extra)}. "
            f"Expected keys: {param_names}"
        )
