"""Tests for the SDR online contingent replanner with SDRSimulator.

These tests use the custom ``SDRSimulator`` (backed by the C# Simulator
class) instead of unified-planning's ``SimulatedExecutionEnvironment``.

SDR is an online planner, so the simulation is nondeterministic — the
tests verify that the goal is reached within a reasonable number of steps.
"""

import pytest

from conftest import SDR_SIMULATOR_PROBLEMS, load_problem

import unified_planning.environment as environment
from unified_planning.shortcuts import ActionSelector
from up_cpor.simulator import SDRSimulator

MAX_STEPS = 500


@pytest.mark.parametrize("problem_name", SDR_SIMULATOR_PROBLEMS)
def test_sdr_simulator_reaches_goal(problem_name):
    """Run SDR online simulation with SDRSimulator and verify goal is reached."""
    problem = load_problem(problem_name)

    env = environment.get_environment()
    env.factory.add_engine("SDRPlanning", "up_cpor.engine", "SDRImpl")

    with ActionSelector(name="SDRPlanning", problem=problem) as solver:
        simulated_env = SDRSimulator(problem)
        steps = 0
        while not simulated_env.is_goal_reached():
            action = solver.get_action()
            observation = simulated_env.apply(action)
            solver.update(observation)
            steps += 1
            assert steps <= MAX_STEPS, (
                f"SDR did not reach the goal within {MAX_STEPS} steps "
                f"for {problem_name}"
            )

    assert simulated_env.is_goal_reached(), (
        f"SDR simulation ended without reaching the goal for {problem_name}"
    )
