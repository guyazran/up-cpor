import sys
import os
import re
import subprocess
import json
import tempfile
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

# All problems that PDDLReader can load (listed in test_cpor.py, not commented out).
# This is the single canonical problem list used by ALL planners.
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

# Timeout in seconds for planner invocations that may hang due to bugs.
PLANNER_TIMEOUT = 120

# Preamble that removes the repo root from sys.path before importing
# pythonnet-based modules, preventing the CPORLib directory from
# shadowing the C# DLL namespace.
_SUBPROCESS_PREAMBLE = f"""\
import sys, os
repo_root = {REPO_ROOT!r}
# Remove the repo root (and CWD aliases) from sys.path so the CPORLib/
# source directory does not shadow the C# DLL namespace via pythonnet.
for _p in [repo_root, '', '.', os.getcwd()]:
    while _p in sys.path:
        sys.path.remove(_p)
"""


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


def _run_subprocess(script, result_file, timeout=PLANNER_TIMEOUT):
    """Run a Python script in a subprocess with a timeout.

    The subprocess is killed if it exceeds *timeout* seconds.  This
    provides reliable timeout behaviour even when the C# planner (via
    pythonnet/mono) blocks in native code that cannot be interrupted by
    Python signals or threads.

    The script must write its result to *result_file*.  Stdout/stderr
    are discarded (the C# planner prints progress information there).
    Raises ``pytest.fail`` on timeout or non-zero exit.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"Timed out after {timeout}s (planner is hanging — likely a bug)"
        )
    if result.returncode != 0:
        # Combine stdout+stderr so we see C# errors too
        combined = (result.stdout + "\n" + result.stderr)[-3000:]
        pytest.fail(
            f"Planner subprocess failed (exit {result.returncode}):\n"
            f"{combined}"
        )
    if not os.path.exists(result_file):
        pytest.fail("Planner subprocess did not produce a result file")


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


def run_cpor_get_dot(problem_name, timeout=PLANNER_TIMEOUT):
    """Run the CPOR planner in a subprocess and return the DOT output string.

    Returns ``None`` when the planner finds no solution.
    """
    result_file = tempfile.mktemp(suffix=".json")
    dot_file = tempfile.mktemp(suffix=".dot")
    script = _SUBPROCESS_PREAMBLE + f"""\
import json
from unified_planning.io import PDDLReader
from up_cpor.converter import UpCporConverter
from CPORLib.Algorithms import CPORPlanner

tests_dir = {TESTS_DIR!r}
reader = PDDLReader()
problem = reader.parse_problem(
    os.path.join(tests_dir, {problem_name!r}, "d.pddl"),
    os.path.join(tests_dir, {problem_name!r}, "p.pddl"),
)
cnv = UpCporConverter()
c_domain = cnv.createDomain(problem)
c_problem = cnv.createProblem(problem, c_domain)
solver = CPORPlanner(c_domain, c_problem)
c_plan = solver.OfflinePlanning()
if c_plan is None:
    with open({result_file!r}, "w") as f:
        json.dump({{"has_plan": False}}, f)
else:
    solver.WritePlan({dot_file!r}, c_plan)
    with open({dot_file!r}) as f:
        dot_text = f.read()
    os.unlink({dot_file!r})
    with open({result_file!r}, "w") as f:
        json.dump({{"has_plan": True, "dot": dot_text}}, f)
"""
    try:
        _run_subprocess(script, result_file, timeout=timeout)
        with open(result_file) as f:
            data = json.load(f)
        if not data["has_plan"]:
            return None
        return data["dot"]
    finally:
        for p in (result_file, dot_file):
            if os.path.exists(p):
                os.unlink(p)


def run_engine_api(problem_name, engine_module, engine_class, engine_name,
                   timeout=PLANNER_TIMEOUT, meta_engine=False,
                   planner_name=None):
    """Run a planner through the UP engine API in a subprocess.

    Returns a dict with keys ``status`` and ``has_plan``.
    """
    result_file = tempfile.mktemp(suffix=".json")
    add_method = "add_meta_engine" if meta_engine else "add_engine"
    if planner_name is None:
        planner_name = engine_name
    script = _SUBPROCESS_PREAMBLE + f"""\
import json
from unified_planning.io import PDDLReader
import unified_planning.environment as environment
from unified_planning.shortcuts import OneshotPlanner

tests_dir = {TESTS_DIR!r}
reader = PDDLReader()
problem = reader.parse_problem(
    os.path.join(tests_dir, {problem_name!r}, "d.pddl"),
    os.path.join(tests_dir, {problem_name!r}, "p.pddl"),
)
env = environment.get_environment()
env.factory.{add_method}({engine_name!r}, {engine_module!r}, {engine_class!r})

with OneshotPlanner(name={planner_name!r}) as planner:
    result = planner.solve(problem)
    with open({result_file!r}, "w") as f:
        json.dump({{
            "status": str(result.status),
            "has_plan": result.plan is not None,
        }}, f)
"""
    try:
        _run_subprocess(script, result_file, timeout=timeout)
        with open(result_file) as f:
            return json.load(f)
    finally:
        if os.path.exists(result_file):
            os.unlink(result_file)


def run_sdr_simulation(problem_name, use_sdr_simulator=False,
                       max_steps=500, timeout=PLANNER_TIMEOUT):
    """Run the SDR online simulation in a subprocess.

    Returns a dict with keys ``goal_reached`` and ``steps``.
    """
    result_file = tempfile.mktemp(suffix=".json")
    if use_sdr_simulator:
        env_setup = "from up_cpor.simulator import SDRSimulator\n    simulated_env = SDRSimulator(problem)"
    else:
        env_setup = "from unified_planning.model.contingent import SimulatedExecutionEnvironment\n    simulated_env = SimulatedExecutionEnvironment(problem)"

    script = _SUBPROCESS_PREAMBLE + f"""\
import json
from unified_planning.io import PDDLReader
import unified_planning.environment as environment
from unified_planning.shortcuts import ActionSelector

tests_dir = {TESTS_DIR!r}
reader = PDDLReader()
problem = reader.parse_problem(
    os.path.join(tests_dir, {problem_name!r}, "d.pddl"),
    os.path.join(tests_dir, {problem_name!r}, "p.pddl"),
)
env = environment.get_environment()
env.factory.add_engine("SDRPlanning", "up_cpor.engine", "SDRImpl")

with ActionSelector(name="SDRPlanning", problem=problem) as solver:
    {env_setup}
    steps = 0
    while not simulated_env.is_goal_reached():
        action = solver.get_action()
        observation = simulated_env.apply(action)
        solver.update(observation)
        steps += 1
        if steps > {max_steps}:
            with open({result_file!r}, "w") as f:
                json.dump({{"goal_reached": False, "steps": steps,
                           "error": "exceeded max steps"}}, f)
            raise SystemExit(0)

goal_reached = simulated_env.is_goal_reached()
with open({result_file!r}, "w") as f:
    json.dump({{"goal_reached": goal_reached, "steps": steps}}, f)
"""
    try:
        _run_subprocess(script, result_file, timeout=timeout)
        with open(result_file) as f:
            return json.load(f)
    finally:
        if os.path.exists(result_file):
            os.unlink(result_file)


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
