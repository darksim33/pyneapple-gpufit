"""CUDA availability guard for pyneapple-gpufit."""

from __future__ import annotations


def require_cuda() -> None:
    """Import pygpufit and verify CUDA is available on this machine.

    Raises:
        ImportError: If the Gpufit native library cannot be loaded.
        RuntimeError: If CUDA is not available on this machine.
    """
    try:
        from pyneapple_gpufit._vendor.pygpufit import gpufit
    except OSError as exc:
        raise ImportError(
            "pygpufit could not load the GPU library (Gpufit.dll / libGpufit.so). "
            "Ensure your system has a compatible NVIDIA driver installed."
        ) from exc

    if not gpufit.cuda_available():
        raise RuntimeError(
            "CUDA is not available on this machine. "
            "GpuCurveFitSolver requires an NVIDIA GPU with a compatible CUDA driver."
        )
