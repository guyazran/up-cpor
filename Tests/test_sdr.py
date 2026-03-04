"""Tests for the SDR online contingent replanner.

SDR is an online planner: it provides one action at a time and awaits
observations from the environment. These tests run the planner through a
simulation loop using unified-planning's ``SimulatedExecutionEnvironment``
and verify that the goal is reached.

Because the SimulatedExecutionEnvironment picks a random consistent
initial state, the exact action sequence is nondeterministic. The tests
therefore only check that the simulation terminates with the goal reached.
"""

import pytest

from conftest import CPOR_PROBLEMS, run_sdr_simulation

MAX_STEPS = 500


@pytest.mark.parametrize("problem_name", CPOR_PROBLEMS)
def test_sdr_reaches_goal(problem_name):
    """Run SDR online simulation and verify the goal is reached."""
    result = run_sdr_simulation(
        problem_name, use_sdr_simulator=False, max_steps=MAX_STEPS,
    )
    assert result["goal_reached"], (
        f"SDR simulation did not reach the goal for {problem_name} "
        f"(steps={result['steps']}, error={result.get('error', 'none')})"
    )
