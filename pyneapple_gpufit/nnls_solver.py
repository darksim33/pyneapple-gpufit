"""GPU NNLS solver stub — not yet implemented."""

from __future__ import annotations

from typing import Any

import numpy as np

from pyneapple.solvers.base import BaseSolver


class GpuNNLSSolver(BaseSolver):
    """GPU NNLS solver — not yet implemented.

    pygpufit only supports Levenberg-Marquardt curve fitting.  NNLS on GPU
    is not available in the current library version.  This class is
    registered as an entry-point so that the name ``gpufit_nnls`` is reserved
    for a future implementation.

    Raises:
        NotImplementedError: Always, when ``fit()`` is called.
    """

    def __init__(
        self,
        model: Any,
        max_iter: int = 250,
        tol: float = 1e-8,
        verbose: bool = False,
        **solver_kwargs,
    ):
        super().__init__(model=model, max_iter=max_iter, tol=tol, verbose=verbose)

    def fit(self, xdata: np.ndarray, ydata: np.ndarray, **kwargs) -> "GpuNNLSSolver":
        """Not implemented.

        Raises:
            NotImplementedError: pygpufit does not support NNLS.
        """
        raise NotImplementedError(
            "GpuNNLSSolver is not implemented. "
            "pygpufit only supports Levenberg-Marquardt curve fitting. "
            "Use NNLSSolver for distribution-based fitting."
        )
