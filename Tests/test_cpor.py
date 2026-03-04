"""Tests for the CPOR offline contingent planner.

These tests verify that:
1. The planner produces a plan matching the expected DOT output (out.txt).
2. The expected output represents a valid contingent solution (every
   observation branch leads to a Goal node).
3. The planner is accessible through the unified-planning engine API.
"""

import pytest

from conftest import (
    CPOR_PROBLEMS,
    load_problem,
    normalize_dot,
    read_expected_output,
    run_cpor_get_dot,
    run_engine_api,
    validate_contingent_plan,
)

from unified_planning.engines.results import PlanGenerationResultStatus


# ---------------------------------------------------------------------------
# 1. Verify that the planner DOT output matches the expected out.txt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("problem_name", CPOR_PROBLEMS)
def test_cpor_dot_output_matches_expected(problem_name):
    """The CPOR planner DOT output should match the expected out.txt."""
    actual_dot = run_cpor_get_dot(problem_name)
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
    actual_dot = run_cpor_get_dot(problem_name)

    assert actual_dot is not None, (
        f"Planner produced no plan for {problem_name}"
    )

    errors = validate_contingent_plan(actual_dot)
    assert not errors, (
        f"Planner output for {problem_name} is not a valid contingent plan:\n"
        + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# 4. Test the engine API (unified-planning OneshotPlanner interface)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("problem_name", CPOR_PROBLEMS)
def test_cpor_engine_api(problem_name):
    """The CPOR planner should be usable through the UP OneshotPlanner API."""
    result = run_engine_api(
        problem_name,
        engine_module="up_cpor.engine",
        engine_class="CPORImpl",
        engine_name="CPORPlanning",
    )
    assert result["status"] == str(PlanGenerationResultStatus.SOLVED_SATISFICING), (
        f"Expected SOLVED_SATISFICING for {problem_name}, "
        f"got {result['status']}"
    )
    assert result["has_plan"]
