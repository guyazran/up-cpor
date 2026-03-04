"""Tests for the CPOR Meta-Engine with pluggable classical planners.

The ``CPORMetaEngineImpl`` wraps CPOR to use any compatible classical
planner (e.g. ``tamer``, ``pyperplan``) as its internal search engine.
These tests require the ``up-tamer`` and ``up-pyperplan`` packages to be
installed.
"""

import pytest

import up_tamer  # noqa: F401 — test fails if not installed
import up_pyperplan  # noqa: F401 — test fails if not installed

from conftest import (
    CPOR_PROBLEMS,
    normalize_dot,
    read_expected_output,
    run_cpor_get_dot,
    run_engine_api,
    validate_contingent_plan,
)

from unified_planning.engines.results import PlanGenerationResultStatus

_BACKENDS = ["tamer", "pyperplan"]


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("problem_name", CPOR_PROBLEMS)
def test_meta_cpor_engine_api(problem_name, backend):
    """The Meta-CPOR planner should find a plan via the UP API."""
    result = run_engine_api(
        problem_name,
        engine_module="up_cpor.engine",
        engine_class="CPORMetaEngineImpl",
        engine_name="MetaCPORPlanning",
        meta_engine=True,
        planner_name=f"MetaCPORPlanning[{backend}]",
    )
    assert result["status"] == str(PlanGenerationResultStatus.SOLVED_SATISFICING), (
        f"Expected SOLVED_SATISFICING for {problem_name} "
        f"with backend {backend}, got {result['status']}"
    )
    assert result["has_plan"]


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("problem_name", CPOR_PROBLEMS)
def test_meta_cpor_plan_matches_cpor(problem_name, backend):
    """The Meta-CPOR plan should match the standard CPOR expected output."""
    expected_dot = read_expected_output(problem_name)
    assert expected_dot is not None and expected_dot.strip() != "", (
        f"No expected output (out.txt) for {problem_name}"
    )

    actual_dot = run_cpor_get_dot(problem_name)
    assert actual_dot is not None, (
        f"Meta-CPOR with {backend} returned no plan for {problem_name}"
    )

    assert normalize_dot(actual_dot) == normalize_dot(expected_dot), (
        f"Meta-CPOR[{backend}] DOT output differs from expected out.txt "
        f"for {problem_name}"
    )
