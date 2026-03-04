"""Tests for the SDR online contingent replanner with SDRSimulator.

These tests use the custom ``SDRSimulator`` (backed by the C# Simulator
class) instead of unified-planning's ``SimulatedExecutionEnvironment``.

SDR is an online planner, so the simulation is nondeterministic — the
tests verify that the goal is reached within a reasonable number of steps.
"""

import pytest

from conftest import CPOR_PROBLEMS, run_sdr_simulation

MAX_STEPS = 500


@pytest.mark.parametrize("problem_name", CPOR_PROBLEMS)
def test_sdr_simulator_reaches_goal(problem_name):
    """Run SDR online simulation with SDRSimulator and verify goal is reached."""
    result = run_sdr_simulation(
        problem_name, use_sdr_simulator=True, max_steps=MAX_STEPS,
    )
    assert result["goal_reached"], (
        f"SDR simulation did not reach the goal for {problem_name} "
        f"(steps={result['steps']}, error={result.get('error', 'none')})"
    )
