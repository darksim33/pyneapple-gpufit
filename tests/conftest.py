"""Shared fixtures and markers for pyneapple-gpufit tests."""

from __future__ import annotations

import numpy as np
import pytest


# ── CUDA skip marker ──────────────────────────────────────────────────────────


def _cuda_available() -> bool:
    """Return True if a CUDA-capable GPU is available."""
    try:
        from pyneapple_gpufit._vendor.pygpufit import gpufit

        return gpufit.cuda_available()
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(
    not _cuda_available(), reason="CUDA GPU not available"
)


# ── b-values fixture ──────────────────────────────────────────────────────────


@pytest.fixture
def b_values() -> np.ndarray:
    """Standard IVIM b-values in s/mm²."""
    return np.array(
        [0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200],
        dtype=np.float64,
    )


# ── model fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def biexp_reduced_model():
    """BiExpModel in reduced mode (default): params = [f1, D1, D2]."""
    from pyneapple.models.biexp import BiExpModel

    return BiExpModel(fit_reduced=True)


@pytest.fixture
def biexp_full_model():
    """BiExpModel in full mode: params = [f1, D1, f2, D2]."""
    from pyneapple.models.biexp import BiExpModel

    return BiExpModel(fit_reduced=False)


@pytest.fixture
def monoexp_model():
    """MonoExpModel: params = [S0, D]."""
    from pyneapple.models.monoexp import MonoExpModel

    return MonoExpModel()


# ── p0 / bounds fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def biexp_reduced_p0() -> dict[str, float]:
    """Realistic initial guesses for reduced biexp model."""
    return {"f1": 0.2, "D1": 0.01, "D2": 0.001}


@pytest.fixture
def biexp_reduced_bounds() -> dict[str, tuple[float, float]]:
    """Parameter bounds for reduced biexp model."""
    return {
        "f1": (0.0, 1.0),
        "D1": (0.0001, 0.1),
        "D2": (0.00001, 0.01),
    }


@pytest.fixture
def monoexp_p0() -> dict[str, float]:
    """Initial guesses for monoexp model."""
    return {"S0": 1000.0, "D": 0.001}


@pytest.fixture
def monoexp_bounds() -> dict[str, tuple[float, float]]:
    """Parameter bounds for monoexp model."""
    return {"S0": (0.0, 5000.0), "D": (0.0001, 0.1)}


# ── ndarray / per-pixel fixtures ──────────────────────────────────────────────


@pytest.fixture
def biexp_reduced_tuple_bounds_batch() -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel bounds as (lower, upper) each shaped (10, 3) for a 10-pixel batch."""
    n_pixels = 10
    lower = np.tile(np.array([0.0, 0.0001, 0.00001]), (n_pixels, 1)).astype(np.float64)
    upper = np.tile(np.array([1.0, 0.1, 0.01]), (n_pixels, 1)).astype(np.float64)
    return lower, upper


@pytest.fixture
def biexp_reduced_ndarray_p0_batch() -> np.ndarray:
    """Per-pixel p0 as ndarray shaped (10, 3) for a 10-pixel batch."""
    n_pixels = 10
    return np.tile(np.array([0.2, 0.01, 0.001]), (n_pixels, 1)).astype(np.float64)


# ── synthetic signal fixtures ─────────────────────────────────────────────────


@pytest.fixture
def synthetic_biexp_signal(b_values) -> tuple[np.ndarray, dict[str, float]]:
    """Single-pixel synthetic biexp-reduced signal with known ground-truth params."""
    true_params = {"f1": 0.3, "D1": 0.012, "D2": 0.0012}
    f1, D1, D2 = true_params["f1"], true_params["D1"], true_params["D2"]
    signal = f1 * np.exp(-b_values * D1) + (1 - f1) * np.exp(-b_values * D2)
    return signal.astype(np.float64), true_params


@pytest.fixture
def synthetic_biexp_signal_batch(b_values) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Multi-pixel (10 pixels) synthetic biexp-reduced signals."""
    n_pixels = 10
    rng = np.random.default_rng(42)
    f1_vals = rng.uniform(0.1, 0.5, size=n_pixels)
    D1_vals = rng.uniform(0.005, 0.02, size=n_pixels)
    D2_vals = rng.uniform(0.0005, 0.002, size=n_pixels)
    signals = np.stack(
        [
            f1_vals[i] * np.exp(-b_values * D1_vals[i])
            + (1 - f1_vals[i]) * np.exp(-b_values * D2_vals[i])
            for i in range(n_pixels)
        ]
    ).astype(np.float64)
    true_params = {"f1": f1_vals, "D1": D1_vals, "D2": D2_vals}
    return signals, true_params
