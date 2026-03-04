import os
import sys
import site

# Set environment variables for Python.NET on macOS
# using the Mono runtime installed via Homebrew.
if sys.platform == "darwin":
    os.environ["PYTHONNET_RUNTIME"] = "mono"
    os.environ["PYTHONNET_MONO_LIBMONO"] = "/opt/homebrew/opt/mono/lib/libmonosgen-2.0.dylib"

import re
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, Set

import pytest
from unified_planning.io import PDDLReader

TESTS_DIR = Path(__file__).resolve().parent


from up_cpor.converter import UpCporConverter
from CPORLib.Algorithms import CPORPlanner


DOMAINS = ("blocks2", "blocks3", "doors5")

_EDGE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)\s*(?:\[[^\]]*\])?\s*;\s*$")
_NODE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*\[(.*)\]\s*;\s*$")
_LABEL_RE = re.compile(r'label\s*=\s*"([^"]*)"')
_BOX_RE = re.compile(r'shape\s*=\s*"box"', re.IGNORECASE)
_ID_PREFIX_RE = re.compile(r"^\s*\d+\)\s*")


def _normalize_action_label(label: str) -> str:
    return _ID_PREFIX_RE.sub("", label).strip().lower()


def _parse_dot(path: Path) -> Dict[str, object]:
    action_nodes: Counter[Tuple[str, str]] = Counter()
    observation_nodes: Counter[Tuple[str, bool]] = Counter()
    edges: Counter[Tuple[str, str]] = Counter()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line == "{" or line == "}" or line.startswith("digraph"):
            continue

        edge_match = _EDGE_RE.match(line)
        if edge_match:
            src, dst = edge_match.groups()
            edges[(src, dst)] += 1
            continue

        node_match = _NODE_RE.match(line)
        if not node_match:
            continue

        node_id, attrs = node_match.groups()
        if node_id == "_nil":
            continue

        label_match = _LABEL_RE.search(attrs)
        if label_match is None:
            continue

        label = label_match.group(1)
        if _BOX_RE.search(attrs):
            observation_nodes[(node_id, label.strip().lower() == "true")] += 1
        else:
            action_nodes[(node_id, _normalize_action_label(label))] += 1

    roots = tuple(sorted(dst for (src, dst), count in edges.items() for _ in range(count) if src == "_nil"))
    return {
        "action_nodes": action_nodes,
        "observation_nodes": observation_nodes,
        "edges": edges,
        "roots": roots,
    }


def _format_counter(counter: Counter, limit: int = 12) -> str:
    items = sorted(counter.items())
    trimmed = items[:limit]
    formatted = ", ".join(f"{item} x{count}" for item, count in trimmed)
    if len(items) > limit:
        formatted += f", ... ({len(items) - limit} more)"
    return formatted or "<none>"


def _assert_dot_equal(expected: Dict[str, object], actual: Dict[str, object], domain: str) -> None:
    messages = []
    for key in ("action_nodes", "observation_nodes", "edges"):
        expected_counter = expected[key]
        actual_counter = actual[key]
        missing = expected_counter - actual_counter
        extra = actual_counter - expected_counter
        if missing:
            messages.append(f"{key} missing: {_format_counter(missing)}")
        if extra:
            messages.append(f"{key} extra: {_format_counter(extra)}")

    if expected["roots"] != actual["roots"]:
        messages.append(f"roots mismatch: expected={expected['roots']} actual={actual['roots']}")

    assert not messages, f"DOT mismatch for {domain}\n" + "\n".join(messages)


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
def test_cpor_matches_expected_plan(domain: str, tmp_path: Path):
    domain_dir = TESTS_DIR / domain
    expected_dot = domain_dir / "out.txt"
    assert expected_dot.exists(), f"Missing expected output: {expected_dot}"

    actual_dot = tmp_path / f"{domain}_actual.dot"
    _run_cpor_and_write_dot(domain_dir, actual_dot)
    assert actual_dot.exists(), f"CPOR did not produce output DOT for {domain}"

    expected_graph = _parse_dot(expected_dot)
    actual_graph = _parse_dot(actual_dot)
    _assert_dot_equal(expected_graph, actual_graph, domain)


@pytest.mark.parametrize("domain", DOMAINS)
def test_cpor_generated_plan_is_valid(domain: str, tmp_path: Path):
    domain_dir = TESTS_DIR / domain
    actual_dot = tmp_path / f"{domain}_validity.dot"
    planner, solution = _run_cpor_and_write_dot(domain_dir, actual_dot)
    assert planner.ValidatePlanGraph(solution), f"CPOR returned an invalid contingent plan for {domain}"

