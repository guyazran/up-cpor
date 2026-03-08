import difflib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from pysmt.environment import reset_env as reset_pysmt_env


def reset_sdr_seeds(seed: int = 0) -> None:
    random.seed(seed)
    reset_pysmt_env()

    import System

    asm = None
    for candidate in System.AppDomain.CurrentDomain.GetAssemblies():
        if candidate.GetName().Name == "CPORLib":
            asm = candidate
            break
    assert asm is not None, "CPORLib assembly is not loaded."

    rg_type = asm.GetType("CPORLib.Tools.RandomGenerator")
    assert rg_type is not None, "CPORLib.Tools.RandomGenerator type is not available."

    flags = System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic
    init_method = None
    for method in rg_type.GetMethods(flags):
        if method.Name == "Init" and method.GetParameters().Length == 1:
            init_method = method
            break

    assert init_method is not None, "CPORLib.Tools.RandomGenerator.Init(int) was not found."
    init_method.Invoke(None, System.Array[System.Object]([System.Int32(seed)]))


def normalize_observation(observation: Optional[Dict[Any, Any]]) -> Optional[List[List[Any]]]:
    if observation is None:
        return None
    if len(observation) == 0:
        return []

    normalized: List[List[Any]] = []
    for fluent, value in observation.items():
        text = str(value).strip().lower()
        if text == "true":
            parsed_value: Any = True
        elif text == "false":
            parsed_value = False
        else:
            parsed_value = str(value)
        normalized.append([str(fluent), parsed_value])

    normalized.sort(key=lambda x: (x[0], str(x[1])))
    return normalized


def assert_json_snapshot(actual: Dict[str, Any], snapshot_path: Path, label: str) -> None:
    expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if expected != actual:
        expected_json = json.dumps(expected, indent=2, sort_keys=True)
        actual_json = json.dumps(actual, indent=2, sort_keys=True)
        diff = "\n".join(
            difflib.unified_diff(
                expected_json.splitlines(),
                actual_json.splitlines(),
                fromfile=f"{snapshot_path} (expected)",
                tofile=f"{label} (actual)",
                lineterm="",
            )
        )
        raise AssertionError(f"Snapshot mismatch for {label}\n{diff}")
