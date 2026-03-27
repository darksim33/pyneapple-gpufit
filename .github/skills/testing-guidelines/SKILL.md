---
name: testing-guidelines
description: 'Conventions for writing pytest tests. Use when: writing new tests, adding fixtures, updating existing tests, running or debugging tests, test naming conventions, parametrize, mocking, markers.'
---

# Testing Guidelines

## When to use me

Use this skill when:
- Writing new tests in `tests/`
- Fixing or updating existing tests
- Adding test fixtures to `conftest.py`
- Running or debugging tests

## Organization

- All tests in `tests/`, named `test_<layer>_<concept>.py` (e.g., `test_solver_gpufit.py`)
- Shared fixtures in `tests/conftest.py`
- Import order: stdlib → third-party (`pytest`, `numpy`) → local (`from pyneapple_gpufit import ...`)

## Structure

**Class-based tests** (preferred) — group related tests by concept:

```python
class TestGpuCurveFitSolver:
    """Test suite for GpuCurveFitSolver functionality."""

    def test_initialization(self, model_params):
        """GpuCurveFitSolver initializes correctly with valid parameters."""
        solver = GpuCurveFitSolver(**model_params)
        assert solver.max_iter == 250
```

**Standalone functions** only for trivial, ungroupable tests.

### Naming

- Methods: `test_<what>_<condition>` — e.g., `test_fit_converges_with_good_initial_values`
- Classes: `Test<Class>` or `Test<Feature>` — e.g., `TestGpuCurveFitSolver`
- Fixtures: lowercase with type prefix — `solver_params`, `fit_result`, `sample_data`
- **Every test must have a docstring** explaining what is tested

## Fixtures

| Scope | Use for |
|-------|---------|
| `function` (default) | Mutable data, per-test isolation |
| `class` / `module` | Expensive setup shared across a group |
| `session` | File I/O, one-time setup |

Use `yield` for cleanup. Use `tmp_path` for temporary files (auto-cleaned).

## Assertions

- Use plain `assert` (pytest style), never `self.assertEqual()`
- Add messages for non-obvious assertions: `assert x > 0, f"Expected positive, got {x}"`
- NumPy: prefer `np.testing.assert_allclose(a, b, rtol=1e-5)` (detailed diff on failure)
- Exceptions: `with pytest.raises(ValueError, match="must be positive"):`

## Mocking

**Always use `pytest-mock`** (`mocker` fixture), not `unittest.mock`.

Key patterns: `mocker.patch()`, `mocker.Mock()`, `mocker.spy()`.

**Mock:** pygpufit/gpufit calls for unit tests, expensive computations.
**Don't mock:** the code under test, pure functions, simple data structures.

## CUDA skip marker

```python
import pytest

def _cuda_available():
    try:
        from pygpufit import gpufit
        return gpufit.cuda_available()
    except ImportError:
        return False

requires_cuda = pytest.mark.skipif(
    not _cuda_available(), reason="CUDA GPU not available"
)
```

Use `@requires_cuda` on integration tests that call actual GPU code.
Unit tests should mock `pygpufit.gpufit` and run without CUDA.

## Markers

| Marker | Use for |
|--------|---------|
| `@requires_cuda` | Tests that need an actual NVIDIA GPU |
| `@pytest.mark.slow` | Tests taking more than a few seconds |
| `@pytest.mark.integration` | End-to-end pipeline tests |
| `@pytest.mark.unit` | Isolated tests, no external dependencies |
| `@pytest.mark.skip(reason=...)` | Not yet implemented |

Run selectively: `pytest -m "not requires_cuda"`

## Parametrization

Use `@pytest.mark.parametrize` instead of repetitive test functions:

```python
@pytest.mark.parametrize("model_type", ["biexp", "triexp"])
def test_fit_different_models(self, model_type):
    """Solver handles different parametric model types."""
    ...
```

## Best Practices

1. **Independence** — no shared mutable state between tests; use fixtures
2. **Edge cases** — always test empty inputs, boundary values, error paths
3. **Speed** — small datasets for logic tests; mark slow tests; mock GPU calls
4. **Descriptive names** — `b_values` not `x`, `expected_signal` not `e`
5. **No test logic** — hardcode expected values, don't recalculate in the test
6. **Temporary files** — use `tmp_path` fixture (auto-cleaned by pytest)

## Running Tests

```bash
pytest                              # all tests (CUDA tests skip on CPU)
pytest tests/test_solver_gpufit.py  # specific file
pytest -m "not requires_cuda"       # CPU-only tests
pytest -m "requires_cuda"           # GPU integration tests only
pytest -x                           # stop on first failure
pytest --lf                         # rerun last failures
```
