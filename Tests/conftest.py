import sys
import os
import re
from collections import defaultdict

import pytest

# Remove repo root from sys.path so the local CPORLib C# source directory
# does not shadow the C# DLL namespace loaded via pythonnet.  Pytest may
# re-add it, so we also block it in the ``_clean_sys_path`` fixture below.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
while REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import get_environment

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

# All problems that PDDLReader can load (listed in test_cpor.py, not commented out)
CPOR_PROBLEMS = [
    "blocks2",
    "blocks3",
    "blocks7",
    "colorballs2-2",
    "doors5",
    "doors15",
    "localize5",
    "unix1",
    "wumpus05",
    "wumpus10",
]

# Problems from the original test_sdr.py (SimulatedExecutionEnvironment)
SDR_PROBLEMS = [
    "blocks2",
    "blocks3",
    "blocks7",
    "doors5",
    "localize5",
    "unix1",
    "wumpus05",
]

# Problems from the original test_sdr_simulator.py (SDRSimulator)
SDR_SIMULATOR_PROBLEMS = [
    "blocks2",
    "doors5",
    "wumpus05",
]

# Problems from the original test_meta_cpor.py
META_CPOR_PROBLEMS = [
    "blocks2",
    "blocks3",
    "doors5",
    "wumpus05",
]


@pytest.fixture(autouse=True)
def _suppress_credits():
    """Suppress UP engine credits output during tests."""
    get_environment().credits_stream = None


@pytest.fixture(autouse=True)
def _clean_sys_path():
    """Ensure the repo root stays off sys.path during each test.

    Pytest may re-add ``rootdir`` between tests; this fixture removes it
    so that the CPORLib *directory* never shadows the C# DLL namespace.
    """
    while REPO_ROOT in sys.path:
        sys.path.remove(REPO_ROOT)
    yield
    # Clean up after the test too, in case it was re-added.
    while REPO_ROOT in sys.path:
        sys.path.remove(REPO_ROOT)


def load_problem(problem_name):
    """Load a PDDL problem from the Tests directory."""
    reader = PDDLReader()
    d_path = os.path.join(TESTS_DIR, problem_name, "d.pddl")
    p_path = os.path.join(TESTS_DIR, problem_name, "p.pddl")
    return reader.parse_problem(d_path, p_path)


def read_expected_output(problem_name):
    """Read expected output from out.txt for a given problem."""
    out_path = os.path.join(TESTS_DIR, problem_name, "out.txt")
    if not os.path.exists(out_path):
        return None
    with open(out_path) as f:
        return f.read()


def normalize_dot(dot_text):
    """Normalize DOT text by stripping internal plan-step IDs from labels.

    Labels like ``"3001)senseclear~b1"`` become ``"senseclear~b1"``.
    This makes comparison resilient to internal ID renumbering across runs.
    """
    lines = []
    for line in dot_text.strip().split("\n"):
        line = re.sub(r'label="\d+\)', 'label="', line)
        lines.append(line.rstrip())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DOT graph parsing and contingent-plan validation
# ---------------------------------------------------------------------------

def parse_dot_graph(dot_text):
    """Parse a DOT-format contingent plan into nodes, edges, and root.

    Returns:
        nodes: dict mapping node_id (str) -> label (str)
        children: dict mapping node_id -> list of child node_ids
        root_id: the root node id (child of _nil)
    """
    nodes = {}
    children = defaultdict(list)
    root_id = None

    for line in dot_text.strip().split("\n"):
        line = line.strip().rstrip(";")

        node_match = re.match(r'(\w+)\s*\[label="([^"]*)"', line)
        if node_match:
            nid = node_match.group(1)
            label = node_match.group(2)
            if nid != "_nil":
                nodes[nid] = label

        edge_match = re.match(r"(\w+)\s*->\s*(\w+)", line)
        if edge_match:
            src = edge_match.group(1)
            dst = edge_match.group(2)
            if src == "_nil":
                root_id = dst
            else:
                children[src].append(dst)

    return nodes, children, root_id


def _extract_action_name(label):
    """Extract action name from a DOT label like ``'3001)senseclear~b1'``."""
    match = re.match(r"\d+\)\s*(.*)", label)
    return match.group(1).strip() if match else label.strip()


def validate_contingent_plan(dot_text):
    """Validate that a DOT contingent plan is a valid contingent solution.

    A valid contingent plan satisfies:
    - Every path from root to a leaf ends at a Goal node.
    - Every sensing action (node with True/False observation children)
      has both a True and a False branch.

    Returns a list of error strings (empty if valid).
    """
    nodes, children, root_id = parse_dot_graph(dot_text)
    if root_id is None:
        return ["No root node found"]

    errors = []

    def validate(nid, path, visited):
        if nid in visited:
            return  # Shared subgraph, already validated
        visited.add(nid)

        label = nodes.get(nid, "")
        child_ids = children.get(nid, [])

        if label in ("True", "False"):
            # Observation node — must have exactly one action child
            if len(child_ids) != 1:
                errors.append(
                    f"Observation node '{label}' has {len(child_ids)} children "
                    f"(expected 1) at path {path}"
                )
            for cid in child_ids:
                validate(cid, path + [label], visited)
            return

        action = _extract_action_name(label)

        if action == "Goal":
            # Valid leaf
            return

        if action == "Deadend":
            errors.append(f"Dead-end node reached at path {path}")
            return

        if not child_ids:
            errors.append(f"Non-goal leaf '{action}' with no children at path {path}")
            return

        # Check whether this is a sensing action (has True/False children)
        child_labels = {nodes.get(cid, ""): cid for cid in child_ids}
        has_true = "True" in child_labels
        has_false = "False" in child_labels

        if has_true or has_false:
            if not (has_true and has_false):
                missing = "True" if not has_true else "False"
                errors.append(
                    f"Sensing action '{action}' missing '{missing}' branch "
                    f"at path {path}"
                )

        for cid in child_ids:
            validate(cid, path + [action], visited)

    validate(root_id, [], set())
    return errors
