"""GPU-accelerated curve-fitting solver via pygpufit."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from numpy._typing import NDArray
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
        p0: dict[str, float] | np.ndarray | None = None,
        bounds: dict[str, tuple[float, float]]
        | tuple[np.ndarray, np.ndarray]
        | None = None,
        pixel_fixed_params: dict[str, np.ndarray] | None = None,
        **fit_kwargs,
    ) -> "GpuCurveFitSolver":
        """Fit the model to data on GPU using pygpufit.

        Args:
            xdata: 1D array of b-values, shape ``(n_points,)``.
            ydata: Signal data, shape ``(n_pixels, n_points)`` or ``(n_points,)``
                for a single pixel.
            p0: Per-call initial guesses (overrides constructor defaults).
                Accepts ``dict[str, float]`` or ``np.ndarray`` shaped
                ``(n_pixels, n_params)`` or ``(n_params, n_pixels)``.
            bounds: Per-call bounds (overrides constructor defaults).
                Accepts ``dict[str, tuple[float, float]]`` for shared bounds or
                ``tuple[np.ndarray, np.ndarray]`` for per-pixel bounds, each
                array shaped ``(n_pixels, n_params)`` or ``(n_params, n_pixels)``.
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

        # ── validate p0 and bounds ────────────────────────────────────────────
        _p0 = p0 if p0 is not None else self.p0
        _bounds = bounds if bounds is not None else self.bounds
        if _p0 is not None and _bounds is not None:
            initial_parameters, constraints = self._validate_p0_and_bounds(
                _p0, _bounds, gpufit_param_names, n_pixels
            )
        else:
            raise NotImplementedError("No p0 or bounds provided.")

        # Convert constraints and initial parameters to contiguous float32 array
        initial_parameters = np.ascontiguousarray(initial_parameters, dtype=np.float32)
        constraints = np.ascontiguousarray(constraints, dtype=np.float32)

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

    def _validate_p0_and_bounds(
        self,
        p0: dict[str, float | int] | np.ndarray,
        bounds: dict[str, tuple[Any, Any]] | tuple[np.ndarray, np.ndarray],
        gpufit_param_names,
        n_pixels,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and convert *p0* and *bounds* into gpufit-ready arrays.

        Handles two input formats for each argument:

        - *p0*: ``dict[str, float]`` (one value per parameter, tiled to all
          pixels) or ``np.ndarray`` shaped ``(n_pixels, n_params)`` or
          ``(n_params, n_pixels)`` (the latter is transposed).
        - *bounds*: ``dict[str, tuple[float, float]]`` (shared bounds tiled to
          all pixels) or ``tuple[np.ndarray, np.ndarray]`` for per-pixel bounds,
          each array shaped ``(n_pixels, n_params)`` or ``(n_params, n_pixels)``
          (the latter is transposed).

        Args:
            p0: Initial parameter guesses.
            bounds: Parameter bounds as lower/upper pairs.
            gpufit_param_names: Ordered parameter names expected by the GPU kernel.
            n_pixels: Number of pixels to fit.

        Returns:
            ``(initial_parameters, constraints)`` where *initial_parameters* has
            shape ``(n_pixels, n_params)`` float32, and *constraints* has shape
            ``(n_pixels, 2 * n_params)`` float32 with values interleaved as
            ``[lo_0, hi_0, lo_1, hi_1, ...]`` per row.

        Raises:
            TypeError: If *p0* or *bounds* have an unsupported type.
            ValueError: If an ndarray *p0* or *bounds* has an incompatible shape.
        """

        n_params = len(gpufit_param_names)

        if self._p0_type(p0) == "scalar":
            # ── build initial_parameters [n_pixels, n_params] float32 ─────────
            p0_vec = np.array(
                [p0[name] for name in gpufit_param_names], dtype=np.float32
            )
            initial_parameters = np.tile(p0_vec, (n_pixels, 1)).astype(
                np.float32, copy=False
            )
        elif isinstance(p0, np.ndarray):
            # in contrast to the CurveFitSolver, the p0 needs to be transposed to [n_pixels, n_params]
            # from [n_params, n_pixels]

            # verify shape is either [n_params, n_pixels] or [n_pixels, n_params]
            if p0.shape == (n_pixels, n_params):
                initial_parameters = p0.astype(np.float32, copy=False)
            elif p0.shape == (n_params, n_pixels):
                initial_parameters = p0.T.astype(np.float32)
            else:
                raise ValueError(
                    "p0 shape must be either [n_params, n_pixels] or [n_pixels, n_params]"
                )
        else:
            raise TypeError(f" {p0} needs to be a dictionary or numpy array.")

        if self._bounds_type(bounds) == "scalar":
            # ── build constraints [n_pixels, 2*n_params] float32 ───────────────
            lower = np.array(
                [bounds[name][0] for name in gpufit_param_names],
                dtype=np.float32,
            )
            upper = np.array(
                [bounds[name][1] for name in gpufit_param_names],
                dtype=np.float32,
            )
            # interleave: [lo_0, hi_0, lo_1, hi_1, ...] shape (2*n_params,)
            constraint_row = np.empty(2 * n_params, dtype=np.float32)
            constraint_row[0::2] = lower
            constraint_row[1::2] = upper
            constraints = np.tile(constraint_row, (n_pixels, 1)).astype(
                np.float32, copy=False
            )  # shape (n_pixels, 2*n_params)
        elif isinstance(bounds, tuple) and self._bounds_type(bounds) == "ndarray":
            # in contrast to the CurveFitSolver, the bounds need to be transposed to [n_pixels, n_params]
            # from [n_params, n_pixels]
            lower, upper = bounds
            if lower.shape == (n_pixels, n_params):
                lower = lower.astype(np.float32, copy=False)
                upper = upper.astype(np.float32, copy=False)
            elif lower.shape == (n_params, n_pixels):
                lower = lower.T.astype(np.float32)
                upper = upper.T.astype(np.float32)
            else:
                raise ValueError(
                    f"Bounds shape {lower.shape} does not match expected shape (n_pixels, n_params) or vise versa."
                )
            constraints = np.empty((n_pixels, 2 * n_params), dtype=np.float32)
            constraints[:, 0::2] = lower
            constraints[:, 1::2] = upper
        else:
            raise TypeError(
                f" {bounds} needs to be a dictionary or tuple of numpy arrays."
            )

        return initial_parameters, constraints

    def _p0_type(self, p0: dict[str, Any] | np.ndarray) -> str | None:
        """Classify the type of *p0* and validate its contents.

        Args:
            p0: Either a ``dict[str, float | int]`` mapping parameter names to
                scalar initial guesses, or a ``np.ndarray`` (shape is validated
                separately in :meth:`_validate_p0_and_bounds`).

        Returns:
            ``"scalar"`` if *p0* is a dict with numeric values; ``None`` if *p0*
            is a ``np.ndarray``.

        Raises:
            TypeError: If *p0* is not a dict or ndarray, if dict keys are not
                strings, or if dict values are not ``int`` or ``float``.
        """
        if not isinstance(p0, (dict, np.ndarray)):
            raise TypeError(f" {p0} needs to be a dictionary.")
        _type = None
        if isinstance(p0, dict):
            for key, value in p0.items():
                if not isinstance(key, str):
                    raise TypeError(f" {key} needs to be a string.")
                if not isinstance(value, (int, float)):
                    raise TypeError(f" {value} needs to be an int or float.")
                _type = "scalar"
        return _type

    def _bounds_type(
        self, bounds: dict[str, Any] | tuple[np.ndarray, np.ndarray]
    ) -> str | None:
        """Classify the type of *bounds* and validate its contents.

        Args:
            bounds: Either a ``dict[str, tuple]`` mapping parameter names to
                ``(lower, upper)`` float pairs, or a
                ``tuple[np.ndarray, np.ndarray]`` of per-pixel bound arrays.

        Returns:
            ``"scalar"`` if *bounds* is a dict with ``(float, float)`` tuples;
            ``"ndarray"`` if it is a ``tuple[np.ndarray, np.ndarray]`` or a dict
            with ``(np.ndarray, ...)`` tuples.

        Raises:
            TypeError: If *bounds* is not a dict or tuple, or if the contents
                have unexpected types.
        """
        _type = None
        if isinstance(bounds, dict):
            for key, value in bounds.items():
                if not isinstance(key, str):
                    raise TypeError(f" {key} needs to be a string.")
                if not isinstance(value, tuple):
                    raise TypeError(f" {value} needs to be a tuple.")
                if not isinstance(value[0], (int, float, np.ndarray)):
                    raise TypeError(
                        f" {value} needs to be an int, float or numpy array."
                    )
                if isinstance(value[0], np.ndarray):
                    _type = "ndarray"
                elif isinstance(value[0], (int, float)):
                    _type = "scalar"
        elif isinstance(bounds, tuple):
            if not isinstance(bounds[0], np.ndarray):
                raise TypeError(f" {bounds[0]} needs to be a numpy array.")
            if not isinstance(bounds[1], np.ndarray):
                raise TypeError(f" {bounds[1]} needs to be a numpy array.")
            _type = "ndarray"
        else:
            raise TypeError(f" {bounds} needs to be a dictionary or numpy array.")
        return _type


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
