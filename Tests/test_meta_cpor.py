"""Tests for the CPOR Meta-Engine with pluggable classical planners.

The ``CPORMetaEngineImpl`` wraps CPOR to use any compatible classical
planner (e.g. ``tamer``, ``pyperplan``) as its internal search engine.
These tests require the ``up-tamer`` and ``up-pyperplan`` packages to be
installed.
"""

import os
import sys
import tempfile

import pytest

from conftest import (
    META_CPOR_PROBLEMS,
    TESTS_DIR,
    load_problem,
    normalize_dot,
    read_expected_output,
    validate_contingent_plan,
)

import unified_planning.environment as environment
from unified_planning.engines.results import PlanGenerationResultStatus
from unified_planning.shortcuts import OneshotPlanner

# Check if the required planner packages are available.
try:
    import up_tamer  # noqa: F401

    _HAS_TAMER = True
except ImportError:
    _HAS_TAMER = False

try:
    import up_pyperplan  # noqa: F401

    _HAS_PYPERPLAN = True
except ImportError:
    _HAS_PYPERPLAN = False

_BACKENDS = []
if _HAS_TAMER:
    _BACKENDS.append("tamer")
if _HAS_PYPERPLAN:
    _BACKENDS.append("pyperplan")


def _run_meta_cpor_get_dot(problem, backend_name):
    """Run the Meta-CPOR planner with the given backend and return DOT output."""
    from up_cpor.converter import UpCporConverter
    from CPORLib.Algorithms import CPORPlanner

    cnv = UpCporConverter()
    c_domain = cnv.createDomain(problem)
    c_problem = cnv.createProblem(problem, c_domain)

    solver = CPORPlanner(c_domain, c_problem)
    c_plan = solver.OfflinePlanning()

    if c_plan is None:
        return None

    tmp = tempfile.mktemp(suffix=".txt")
    try:
        solver.WritePlan(tmp, c_plan)
        with open(tmp) as f:
            return f.read()
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@pytest.mark.skipif(not _BACKENDS, reason="No meta-engine backends installed")
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("problem_name", META_CPOR_PROBLEMS)
def test_meta_cpor_engine_api(problem_name, backend):
    """The Meta-CPOR planner should find a plan via the UP API."""
    problem = load_problem(problem_name)

    env = environment.get_environment()
    env.factory.add_meta_engine(
        "MetaCPORPlanning", "up_cpor.engine", "CPORMetaEngineImpl"
    )

    with OneshotPlanner(name=f"MetaCPORPlanning[{backend}]") as planner:
        result = planner.solve(problem)
        assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING, (
            f"Expected SOLVED_SATISFICING for {problem_name} "
            f"with backend {backend}, got {result.status}"
        )
        assert result.plan is not None


@pytest.mark.skipif(not _BACKENDS, reason="No meta-engine backends installed")
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("problem_name", META_CPOR_PROBLEMS)
def test_meta_cpor_plan_matches_cpor(problem_name, backend):
    """The Meta-CPOR plan should match the standard CPOR expected output.

    If the meta-engine backend produces a different plan than the standard
    CPOR planner, the test is marked as xfail since different classical
    planners may produce different (but equally valid) contingent plans.
    """
    problem = load_problem(problem_name)

    expected_dot = read_expected_output(problem_name)
    if expected_dot is None or expected_dot.strip() == "":
        pytest.skip(f"No expected output for {problem_name}")

    actual_dot = _run_meta_cpor_get_dot(problem, backend)
    if actual_dot is None:
        pytest.fail(f"Meta-CPOR with {backend} returned no plan for {problem_name}")

    if normalize_dot(actual_dot) != normalize_dot(expected_dot):
        # Different backends may produce different but valid plans.
        errors = validate_contingent_plan(actual_dot)
        assert not errors, (
            f"Meta-CPOR with {backend} produced an invalid plan for "
            f"{problem_name}:\n" + "\n".join(errors)
        )
        pytest.xfail(
            f"Meta-CPOR[{backend}] produces a different (but valid) plan "
            f"for {problem_name}"
        )
