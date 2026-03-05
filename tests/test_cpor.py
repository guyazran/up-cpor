import os
import sys

# Set environment variables for Python.NET on macOS
# using the Mono runtime installed via Homebrew.
if sys.platform == "darwin":
    os.environ["PYTHONNET_RUNTIME"] = "mono"
    os.environ["PYTHONNET_MONO_LIBMONO"] = "/opt/homebrew/opt/mono/lib/libmonosgen-2.0.dylib"

from pathlib import Path

import pytest
import unified_planning.environment as environment
from unified_planning.io import PDDLReader
from unified_planning.engines.results import PlanGenerationResultStatus
from unified_planning.shortcuts import OneshotPlanner

from cpor_test_utils import parse_dot, assert_dot_equal
from domains import DOMAINS, TESTS_DIR

from up_cpor.converter import UpCporConverter
from CPORLib.Algorithms import CPORPlanner


@pytest.fixture(scope="session", autouse=True)
def register_cpor_engine():
    env = environment.get_environment()
    env.factory.add_engine("CPORPlanning", "up_cpor.engine", "CPORImpl")


def _run_cpor_and_write_dot(domain_dir: Path, output_path: Path):
    reader = PDDLReader()
    problem = reader.parse_problem(str(domain_dir / "d.pddl"), str(domain_dir / "p.pddl"))

    converter = UpCporConverter()
    c_domain = converter.createDomain(problem)
    c_problem = converter.createProblem(problem, c_domain)

    planner = CPORPlanner(c_domain, c_problem)
    solution = planner.OfflinePlanning()
    assert solution is not None, f"CPOR failed to find a solution for {domain_dir.name}"

    planner.WritePlan(str(output_path), solution)
    return planner, solution


@pytest.mark.parametrize("domain", DOMAINS)
def test_cpor_plan_found(domain: str, register_cpor_engine, tmp_path: Path):
    domain_dir = TESTS_DIR / domain
    reader = PDDLReader()
    problem = reader.parse_problem(str(domain_dir / "d.pddl"), str(domain_dir / "p.pddl"))

    with OneshotPlanner(name="CPORPlanning") as planner:
        result = planner.solve(problem)

    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING, (
        f"CPORPlanning failed to find a plan for {domain}: {result.status}"
    )


@pytest.mark.parametrize("domain", DOMAINS)
def test_cpor_matches_expected_plan(domain: str, tmp_path: Path):
    domain_dir = TESTS_DIR / domain
    expected_dot = domain_dir / "out.txt"
    assert expected_dot.exists(), f"Missing expected output: {expected_dot}"

    actual_dot = tmp_path / f"{domain}_actual.dot"
    _run_cpor_and_write_dot(domain_dir, actual_dot)
    assert actual_dot.exists(), f"CPOR did not produce output DOT for {domain}"

    assert_dot_equal(parse_dot(expected_dot), parse_dot(actual_dot), domain)


@pytest.mark.parametrize("domain", DOMAINS)
def test_cpor_generated_plan_is_valid(domain: str, tmp_path: Path):
    domain_dir = TESTS_DIR / domain
    actual_dot = tmp_path / f"{domain}_validity.dot"
    planner, solution = _run_cpor_and_write_dot(domain_dir, actual_dot)
    assert planner.ValidatePlanGraph(solution), f"CPOR returned an invalid contingent plan for {domain}"
