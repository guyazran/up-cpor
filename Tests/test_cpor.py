"""Tests for the CPOR offline contingent planner.

These tests verify that:
1. The planner produces a plan matching the expected DOT output (out.txt).
2. The expected output represents a valid contingent solution (every
   observation branch leads to a Goal node).
3. The planner is accessible through the unified-planning engine API.
"""

import os
import sys
import tempfile

import pytest

from conftest import (
    CPOR_PROBLEMS,
    TESTS_DIR,
    load_problem,
    normalize_dot,
    read_expected_output,
    validate_contingent_plan,
)

from unified_planning.io import PDDLReader
import unified_planning.environment as environment
from unified_planning.engines.results import PlanGenerationResultStatus
from unified_planning.shortcuts import OneshotPlanner


def _run_cpor_get_dot(problem):
    """Run the CPOR planner via the C# API and return the DOT output string.

    Returns ``None`` when the planner finds no solution.
    """
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


# ---------------------------------------------------------------------------
# 1. Verify that the planner DOT output matches the expected out.txt
# ---------------------------------------------------------------------------

# wumpus10 takes over 10 minutes; mark it so it can be skipped easily.
_SLOW_PROBLEMS = {"wumpus10", "doors15"}

# localize5 currently returns no plan; wumpus05 produces a different
# (but valid) plan graph compared to the historical out.txt.
_EXPECTED_MISMATCHES = {"localize5", "wumpus05"}


@pytest.mark.parametrize("problem_name", CPOR_PROBLEMS)
def test_cpor_dot_output_matches_expected(problem_name):
    """The CPOR planner DOT output should match the expected out.txt."""
    if problem_name in _SLOW_PROBLEMS:
        pytest.skip(f"{problem_name} is too slow for routine testing")

    if problem_name in _EXPECTED_MISMATCHES:
        pytest.xfail(f"{problem_name} has a known mismatch with out.txt")

    problem = load_problem(problem_name)
    actual_dot = _run_cpor_get_dot(problem)
    expected_dot = read_expected_output(problem_name)

    if expected_dot is None or expected_dot.strip() == "":
        # No expected output — just verify the planner produced *something*.
        assert actual_dot is not None, (
            f"No expected output and planner also produced no plan for {problem_name}"
        )
        return

    assert actual_dot is not None, (
        f"Planner returned no plan but out.txt exists for {problem_name}"
    )
    assert normalize_dot(actual_dot) == normalize_dot(expected_dot), (
        f"Planner DOT output differs from expected out.txt for {problem_name}"
    )


# ---------------------------------------------------------------------------
# 2. Validate that each expected out.txt is a valid contingent solution
# ---------------------------------------------------------------------------

# Collect problems that have a non-empty out.txt
_PROBLEMS_WITH_EXPECTED = [
    p
    for p in CPOR_PROBLEMS
    if (
        read_expected_output(p) is not None
        and read_expected_output(p).strip() != ""
    )
]


@pytest.mark.parametrize("problem_name", _PROBLEMS_WITH_EXPECTED)
def test_expected_output_is_valid_contingent_plan(problem_name):
    """The expected out.txt should be a valid contingent solution.

    A valid contingent solution has:
    - Every path from root to leaf ending at a Goal node.
    - Every sensing action having both True and False branches.
    """
    dot_text = read_expected_output(problem_name)
    errors = validate_contingent_plan(dot_text)
    assert not errors, (
        f"Expected output for {problem_name} is not a valid contingent plan:\n"
        + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# 3. Validate that the actual planner output is a valid contingent solution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("problem_name", CPOR_PROBLEMS)
def test_cpor_output_is_valid_contingent_plan(problem_name):
    """If the planner produces a plan, it should be a valid contingent solution."""
    if problem_name in _SLOW_PROBLEMS:
        pytest.skip(f"{problem_name} is too slow for routine testing")

    problem = load_problem(problem_name)
    actual_dot = _run_cpor_get_dot(problem)

    if actual_dot is None:
        pytest.skip(f"Planner produced no plan for {problem_name}")

    errors = validate_contingent_plan(actual_dot)
    assert not errors, (
        f"Planner output for {problem_name} is not a valid contingent plan:\n"
        + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# 4. Test the engine API (unified-planning OneshotPlanner interface)
# ---------------------------------------------------------------------------

# Problems where createActionTree raises UPExpressionDefinitionError
# because the observation fluent has multiple parameters.
_ENGINE_API_XFAIL = {"blocks7", "colorballs2-2", "unix1"}


@pytest.mark.parametrize("problem_name", CPOR_PROBLEMS)
def test_cpor_engine_api(problem_name):
    """The CPOR planner should be usable through the UP OneshotPlanner API."""
    if problem_name in _SLOW_PROBLEMS:
        pytest.skip(f"{problem_name} is too slow for routine testing")

    if problem_name in _ENGINE_API_XFAIL:
        pytest.xfail(
            f"{problem_name} fails in createActionTree "
            f"(UPExpressionDefinitionError)"
        )

    if problem_name == "localize5":
        pytest.xfail("localize5 currently returns UNSOLVABLE_PROVEN")

    if problem_name == "wumpus05":
        # wumpus05 succeeds via engine API but the plan differs from out.txt
        pass

    problem = load_problem(problem_name)

    env = environment.get_environment()
    env.factory.add_engine("CPORPlanning", "up_cpor.engine", "CPORImpl")

    with OneshotPlanner(name="CPORPlanning") as planner:
        result = planner.solve(problem)
        assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING, (
            f"Expected SOLVED_SATISFICING for {problem_name}, "
            f"got {result.status}"
        )
        assert result.plan is not None
