import os
import json
import subprocess
import sys
from pathlib import Path

import pytest
from unified_planning.model.contingent import SimulatedExecutionEnvironment

from domains import DOMAINS, TESTS_DIR
from up_test_utils import make_test_environment, parse_test_problem, use_test_environment
from up_cpor.converter import UpCporConverter
from up_cpor.simulator import SDRSimulator
from sdr_test_utils import reset_sdr_seeds, normalize_observation, assert_json_snapshot
from CPORLib.Parsing import Parser

# Set environment variables for Python.NET on macOS
# using the Mono runtime installed via Homebrew.
if sys.platform == "darwin":
    os.environ["PYTHONNET_RUNTIME"] = "mono"
    os.environ["PYTHONNET_MONO_LIBMONO"] = "/opt/homebrew/opt/mono/lib/libmonosgen-2.0.dylib"

SIMULATOR_CONFIG = {
    "blocks2": {"max_steps": 20, "stop_on_goal": True},
    "blocks3": {"max_steps": 4, "stop_on_goal": True},
    "blocks7": {"max_steps": 120, "stop_on_goal": True},
    "colorballs2-2": {"max_steps": 120, "stop_on_goal": True},
    "doors5": {"max_steps": 80, "stop_on_goal": True},
}


def _run_online_trace(problem, simulator_cls, max_steps: int, stop_on_goal: bool):
    reset_sdr_seeds(0)

    all_action_names = {a.name for a in problem.actions}
    trace = []

    with use_test_environment(problem.environment):
        with problem.environment.factory.ActionSelector(problem=problem, name="SDRPlanning") as solver:
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


def _run_online_trace_in_subprocess(domain: str, mode: str, tmp_path: Path):
    output_path = tmp_path / f"{domain}_{mode}_trace.json"
    runner = TESTS_DIR / "sdr_trace_runner.py"
    completed = subprocess.run(
        [sys.executable, str(runner), "--mode", mode, "--domain", domain, "--output", str(output_path)],
        cwd=TESTS_DIR.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"Trace runner failed for {domain}[{mode}]\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("domain", DOMAINS)
def test_sdr_online_trace_matches_snapshot_with_up_simulator(domain: str, tmp_path: Path):
    actual = _run_online_trace_in_subprocess(domain, "up", tmp_path)
    snapshot_path = TESTS_DIR / domain / "sdr_online_up.json"
    assert_json_snapshot(actual, snapshot_path, f"{domain}[UP]")


@pytest.mark.parametrize("domain", DOMAINS)
def test_sdr_online_trace_matches_snapshot_with_sdr_simulator(domain: str, tmp_path: Path):
    actual = _run_online_trace_in_subprocess(domain, "sdrsim", tmp_path)
    snapshot_path = TESTS_DIR / domain / "sdr_online_sdrsim.json"
    assert_json_snapshot(actual, snapshot_path, f"{domain}[SDRSimulator]")


def test_sdr_parser_rejects_malformed_grounded_observation():
    env = make_test_environment()
    problem = parse_test_problem("colorballs2-2", env)
    converter = UpCporConverter()
    parser = Parser()
    c_domain = converter.createDomain(problem)

    with pytest.raises(Exception, match=r"Unknown constant o2,"):
        parser.ParseFormula("(not (obj-at o2, p2-2))", c_domain)


def test_sdr_direct_solver_replans_after_false_obj_at_observation():
    reset_sdr_seeds(0)
    env = make_test_environment()
    problem = parse_test_problem("colorballs2-2", env)
    converter = UpCporConverter()
    c_domain = converter.createDomain(problem)
    c_problem = converter.createProblem(problem, c_domain)
    solver = converter.createSDRSolver(c_domain, c_problem)

    first_action = converter.SDRGet_action(solver, problem)
    assert str(first_action) == "move(p1-1, p2-1)"
    assert converter.SDRupdate(solver, None) is True

    second_action = converter.SDRGet_action(solver, problem)
    assert str(second_action) == "move(p2-1, p2-2)"
    assert converter.SDRupdate(solver, None) is True

    third_action = converter.SDRGet_action(solver, problem)
    assert str(third_action) == "observe-ball(p2-2, o2)"

    expr_manager = problem.environment.expression_manager
    observation = {
        expr_manager.FluentExp(
            problem.fluent("obj-at"),
            (
                expr_manager.ObjectExp(problem.object("o2")),
                expr_manager.ObjectExp(problem.object("p2-2")),
            ),
        ): expr_manager.Bool(False)
    }

    assert converter.SDRupdate(solver, observation) is True

    next_action = converter.SDRGet_action(solver, problem)
    assert next_action is not None
    assert next_action.action.name in {action.name for action in problem.actions}
