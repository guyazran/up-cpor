import os
import sys

# Set environment variables for Python.NET on macOS
# using the Mono runtime installed via Homebrew.
if sys.platform == "darwin":
    os.environ["PYTHONNET_RUNTIME"] = "mono"
    os.environ["PYTHONNET_MONO_LIBMONO"] = "/opt/homebrew/opt/mono/lib/libmonosgen-2.0.dylib"

import pytest
from unified_planning.shortcuts import (
    BoolType,
    Fluent,
    InstantaneousAction,
    Object,
    Problem,
    UserType,
)
from up_cpor.converter import UpCporConverter


class _FakeCporPlanNode:
    def __init__(self, action: str, single_child=None):
        self.Action = action
        self.SingleChild = single_child
        self.FalseObservationChild = None
        self.TrueObservationChild = None


def _object_names(arity: int):
    return [f"o{i}" for i in range(1, arity + 1)]


def _fluent_name(arity: int) -> str:
    return f"obs{arity}"


def _action_name(arity: int) -> str:
    return f"sense{arity}"


def _build_problem_for_arities(arities: tuple[int, ...]) -> Problem:
    obj_type = UserType("obj")
    problem = Problem("converter_observation_arities")

    for i in range(1, max(arities) + 1):
        problem.add_object(Object(f"o{i}", obj_type))

    for arity in arities:
        fluent_parameters = {f"f{i}": obj_type for i in range(1, arity + 1)}
        action_parameters = {f"a{i}": obj_type for i in range(1, arity + 1)}
        problem.add_fluent(Fluent(_fluent_name(arity), BoolType(), **fluent_parameters))
        problem.add_action(InstantaneousAction(_action_name(arity), **action_parameters))

    return problem


def _build_cpor_action(arity: int) -> str:
    objects = _object_names(arity)
    return (
        f"(:action {_action_name(arity)}~{'~'.join(objects)}\n"
        f" :observe ({_fluent_name(arity)} {' '.join(objects)})\n)"
    )


def _build_linear_plan(arities: tuple[int, ...]):
    node = None
    for arity in reversed(arities):
        node = _FakeCporPlanNode(_build_cpor_action(arity), single_child=node)
    return node


def _assert_action_instance(node, arity: int):
    assert node is not None
    assert node.action_instance.action.name == _action_name(arity)
    assert [str(param) for param in node.action_instance.actual_parameters] == _object_names(arity)


@pytest.mark.parametrize("arity", [1, 2, 3, 4, 5])
def test_converter_create_action_tree_supports_single_action_arities(arity: int):
    problem = _build_problem_for_arities((arity,))
    converter = UpCporConverter()
    root = converter.createActionTree(_build_linear_plan((arity,)), problem)
    _assert_action_instance(root, arity)
    assert root.children == []


def test_converter_create_action_tree_supports_linear_plan_mixed_arities():
    arities = (1, 2, 3, 4, 5)
    problem = _build_problem_for_arities(arities)
    converter = UpCporConverter()
    node = converter.createActionTree(_build_linear_plan(arities), problem)

    for index, arity in enumerate(arities):
        _assert_action_instance(node, arity)
        if index == len(arities) - 1:
            assert node.children == []
        else:
            assert len(node.children) == 1
            observation_map, child = node.children[0]
            assert observation_map == {}
            node = child
