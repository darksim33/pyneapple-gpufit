"""Mapping from pyneapple model instances to pygpufit ModelID and parameter order."""

from __future__ import annotations


def resolve_model_id(model) -> tuple[int, list[str]]:
    """Determine the pygpufit ModelID and parameter order for *model*.

    Args:
        model: A pyneapple ParametricModel instance.

    Returns:
        A ``(model_id, param_names)`` pair where ``param_names`` is the
        ordered list of free parameter names as expected by the corresponding
        pygpufit CUDA kernel.  The names always match ``model.param_names``
        for supported configurations.

    Raises:
        ValueError: If the model class, flag combination, or fixed parameter
            configuration is not supported by pygpufit.
    """
    from pyneapple_gpufit._vendor.pygpufit.gpufit import ModelID

    name = type(model).__name__
    fixed = getattr(model, "fixed_params", {}) or {}

    if fixed:
        raise ValueError(
            f"GpuCurveFitSolver does not support model-level fixed parameters "
            f"(model.fixed_params={fixed}). Use CurveFitSolver instead."
        )

    if name == "MonoExpModel":
        if getattr(model, "fit_t1", False):
            raise ValueError(
                "MonoExpModel with T1 correction is not supported by pygpufit. "
                "Use CurveFitSolver instead."
            )
        return ModelID.MONOEXP, ["S0", "D"]

    if name == "BiExpModel":
        if getattr(model, "fit_t1", False):
            raise ValueError(
                "BiExpModel with T1 correction is not supported by pygpufit. "
                "Use CurveFitSolver instead."
            )
        if getattr(model, "fit_s0", False):
            return ModelID.BIEXP_S0, ["S0", "f1", "D1", "D2"]
        if getattr(model, "fit_reduced", True):
            return ModelID.BIEXP_RED, ["f1", "D1", "D2"]
        return ModelID.BIEXP, ["f1", "D1", "f2", "D2"]

    if name == "TriExpModel":
        if getattr(model, "fit_t1", False):
            raise ValueError(
                "TriExpModel with T1 correction is not supported by pygpufit. "
                "Use CurveFitSolver instead."
            )
        if getattr(model, "fit_s0", False):
            return ModelID.TRIEXP_S0, ["S0", "f1", "D1", "f2", "D2", "f3", "D3"]
        if getattr(model, "fit_reduced", True):
            return ModelID.TRIEXP_RED, ["f1", "D1", "f2", "D2", "D3"]
        return ModelID.TRIEXP, ["f1", "D1", "f2", "D2", "f3", "D3"]

    raise ValueError(
        f"Model '{name}' is not supported by GpuCurveFitSolver. "
        "Supported models: MonoExpModel, BiExpModel, TriExpModel "
        "(without T1 correction; without fit_s0=True for bi/triexp)."
    )
