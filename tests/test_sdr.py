import os
import sys
from pathlib import Path

import pytest
import unified_planning.environment as environment
from unified_planning.io import PDDLReader
from unified_planning.model.contingent import SimulatedExecutionEnvironment
from unified_planning.shortcuts import ActionSelector

from up_cpor.simulator import SDRSimulator
from sdr_test_utils import reset_sdr_seeds, normalize_observation, assert_json_snapshot

# Set environment variables for Python.NET on macOS
# using the Mono runtime installed via Homebrew.
if sys.platform == "darwin":
    os.environ["PYTHONNET_RUNTIME"] = "mono"
    os.environ["PYTHONNET_MONO_LIBMONO"] = "/opt/homebrew/opt/mono/lib/libmonosgen-2.0.dylib"

TESTS_DIR = Path(__file__).resolve().parent
DOMAINS = ("blocks2", "blocks3", "doors5")
SIMULATOR_CONFIG = {
    "blocks2": {"max_steps": 20, "stop_on_goal": True},
    "blocks3": {"max_steps": 4, "stop_on_goal": False},
    "doors5": {"max_steps": 80, "stop_on_goal": True},
}


@pytest.fixture(scope="session", autouse=True)
def register_sdr_engine():
    env = environment.get_environment()
    env.credits_stream = None
    env.factory.add_engine("SDRPlanning", "up_cpor.engine", "SDRImpl")


def _parse_problem(domain: str):
    reader = PDDLReader()
    domain_dir = TESTS_DIR / domain
    return reader.parse_problem(str(domain_dir / "d.pddl"), str(domain_dir / "p.pddl"))


def _run_online_trace(problem, simulator_cls, max_steps: int, stop_on_goal: bool):
    reset_sdr_seeds(0)

    all_action_names = {a.name for a in problem.actions}
    trace = []

    with ActionSelector(name="SDRPlanning", problem=problem) as solver:
        simulator = simulator_cls(problem)

        if stop_on_goal:
            while (not simulator.is_goal_reached()) and len(trace) < max_steps:
                action = solver.get_action()
                assert action is not None, "SDR returned no action before reaching goal."
                assert action.action.name in all_action_names, f"Unknown action: {action}"

                observation = simulator.apply(action)
                solver.update(observation)
                trace.append({"action": str(action), "observation": normalize_observation(observation)})

            goal_reached = simulator.is_goal_reached()
            assert goal_reached, f"Goal was not reached within {max_steps} steps."
        else:
            for _ in range(max_steps):
                action = solver.get_action()
                if action is None:
                    break
                assert action.action.name in all_action_names, f"Unknown action: {action}"

                observation = simulator.apply(action)
                solver.update(observation)
                trace.append({"action": str(action), "observation": normalize_observation(observation)})

            goal_reached = None

    return {"goal_reached": goal_reached, "steps": len(trace), "trace": trace}


@pytest.mark.parametrize("domain", DOMAINS)
def test_sdr_online_trace_matches_snapshot_with_up_simulator(domain: str, register_sdr_engine):
    problem = _parse_problem(domain)
    actual = _run_online_trace(problem, SimulatedExecutionEnvironment, max_steps=120, stop_on_goal=True)
    snapshot_path = TESTS_DIR / domain / "sdr_online_up.json"
    assert_json_snapshot(actual, snapshot_path, f"{domain}[UP]")


@pytest.mark.parametrize("domain", DOMAINS)
def test_sdr_online_trace_matches_snapshot_with_sdr_simulator(domain: str, register_sdr_engine):
    problem = _parse_problem(domain)
    cfg = SIMULATOR_CONFIG[domain]
    actual = _run_online_trace(
        problem,
        SDRSimulator,
        max_steps=cfg["max_steps"],
        stop_on_goal=cfg["stop_on_goal"],
    )
    snapshot_path = TESTS_DIR / domain / "sdr_online_sdrsim.json"
    assert_json_snapshot(actual, snapshot_path, f"{domain}[SDRSimulator]")
