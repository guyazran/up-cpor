import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

_EDGE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)\s*(?:\[[^\]]*\])?\s*;\s*$")
_NODE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*\[(.*)\]\s*;\s*$")
_LABEL_RE = re.compile(r'label\s*=\s*"([^"]*)"')
_BOX_RE = re.compile(r'shape\s*=\s*"box"', re.IGNORECASE)
_ID_PREFIX_RE = re.compile(r"^\s*\d+\)\s*")


def sanitize_sys_path_for_pythonnet() -> None:
    # The CPORLib/ C# source directory at the workspace root would shadow the
    # .NET assembly namespace loaded via pythonnet when pytest inserts the rootdir
    # into sys.path (including as '' = CWD). Remove those entries early so CLR
    # imports resolve correctly via the installed package in site-packages.
    workspace = os.path.normpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def is_workspace(p: str) -> bool:
        if not p:  # empty string means CWD
            return os.path.normpath(os.getcwd()) == workspace
        return os.path.normpath(p) == workspace

    sys.path[:] = [p for p in sys.path if not is_workspace(p)]


def parse_dot(path: Path) -> Dict[str, object]:
    action_nodes: Counter = Counter()
    observation_nodes: Counter = Counter()
    edges: Counter = Counter()

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
            normalized = _ID_PREFIX_RE.sub("", label).strip().lower()
            action_nodes[(node_id, normalized)] += 1

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


def assert_dot_equal(expected: Dict[str, object], actual: Dict[str, object], label: str) -> None:
    messages = []
    for key in ("action_nodes", "observation_nodes", "edges"):
        missing = expected[key] - actual[key]
        extra = actual[key] - expected[key]
        if missing:
            messages.append(f"{key} missing: {_format_counter(missing)}")
        if extra:
            messages.append(f"{key} extra: {_format_counter(extra)}")

    if expected["roots"] != actual["roots"]:
        messages.append(f"roots mismatch: expected={expected['roots']} actual={actual['roots']}")

    assert not messages, f"DOT mismatch for {label}\n" + "\n".join(messages)
