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

from conftest import parse_dot, assert_dot_equal

TESTS_DIR = Path(__file__).resolve().parent
DOMAINS = ("blocks2", "blocks3", "doors5")
CLASSICAL_PLANNERS = ("tamer", "pyperplan")


@pytest.fixture(scope="session", autouse=True)
def register_meta_engine():
    env = environment.get_environment()
    env.factory.add_meta_engine("MetaCPORPlanning", "up_cpor.engine", "CPORMetaEngineImpl")


def _run_meta_cpor_and_write_dot(domain_dir: Path, output_path: Path):
    from up_cpor.converter import UpCporConverter
    from CPORLib.Algorithms import CPORPlanner

    reader = PDDLReader()
    problem = reader.parse_problem(str(domain_dir / "d.pddl"), str(domain_dir / "p.pddl"))

    converter = UpCporConverter()
    c_domain = converter.createDomain(problem)
    c_problem = converter.createProblem(problem, c_domain)

    planner = CPORPlanner(c_domain, c_problem)
    solution = planner.OfflinePlanning()
    assert solution is not None, f"Meta CPOR underlying planner failed for {domain_dir.name}"

    planner.WritePlan(str(output_path), solution)
    return planner, solution


@pytest.mark.parametrize("classical_planner", CLASSICAL_PLANNERS)
@pytest.mark.parametrize("domain", DOMAINS)
def test_meta_cpor_plan_found(domain: str, classical_planner: str, register_meta_engine, tmp_path: Path):
    domain_dir = TESTS_DIR / domain
    reader = PDDLReader()
    problem = reader.parse_problem(str(domain_dir / "d.pddl"), str(domain_dir / "p.pddl"))

    with OneshotPlanner(name=f"MetaCPORPlanning[{classical_planner}]") as planner:
        result = planner.solve(problem)

    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING, (
        f"MetaCPOR[{classical_planner}] failed to find a plan for {domain}: {result.status}"
    )


@pytest.mark.parametrize("classical_planner", CLASSICAL_PLANNERS)
@pytest.mark.parametrize("domain", DOMAINS)
def test_meta_cpor_matches_expected_plan(domain: str, classical_planner: str, register_meta_engine, tmp_path: Path):
    domain_dir = TESTS_DIR / domain
    expected_dot = domain_dir / "out.txt"
    assert expected_dot.exists(), f"Missing expected output: {expected_dot}"

    actual_dot = tmp_path / f"{domain}_{classical_planner}_actual.dot"
    _run_meta_cpor_and_write_dot(domain_dir, actual_dot)
    assert actual_dot.exists(), f"CPOR did not produce output DOT for {domain}"

    assert_dot_equal(parse_dot(expected_dot), parse_dot(actual_dot), f"{domain}[{classical_planner}]")


@pytest.mark.parametrize("classical_planner", CLASSICAL_PLANNERS)
@pytest.mark.parametrize("domain", DOMAINS)
def test_meta_cpor_generated_plan_is_valid(domain: str, classical_planner: str, register_meta_engine, tmp_path: Path):
    domain_dir = TESTS_DIR / domain
    actual_dot = tmp_path / f"{domain}_{classical_planner}_validity.dot"
    planner, solution = _run_meta_cpor_and_write_dot(domain_dir, actual_dot)
    assert planner.ValidatePlanGraph(solution), (
        f"MetaCPOR[{classical_planner}] returned an invalid contingent plan for {domain}"
    )
