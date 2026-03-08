import os
import sys

# Set environment variables for Python.NET on macOS
# using the Mono runtime installed via Homebrew.
if sys.platform == "darwin":
    os.environ["PYTHONNET_RUNTIME"] = "mono"
    os.environ["PYTHONNET_MONO_LIBMONO"] = "/opt/homebrew/opt/mono/lib/libmonosgen-2.0.dylib"

import pytest
import unified_planning.environment as up_environment
from unified_planning.model import Fluent, InstantaneousAction, Object, Problem
from up_cpor.converter import UpCporConverter


class _FakeCporPlanNode:
    def __init__(self, action: str, single_child=None):
        self.Action = action
        self.SingleChild = single_child
        self.FalseObservationChild = None
        self.TrueObservationChild = None


class _FakeSolver:
    def __init__(self):
        self.observations = []

    def SetObservation(self, observation):
        self.observations.append(observation)
        return True


def _object_names(arity: int):
    return [f"o{i}" for i in range(1, arity + 1)]


def _fluent_name(arity: int) -> str:
    return f"obs{arity}"


def _action_name(arity: int) -> str:
    return f"sense{arity}"


def _build_problem_for_arities(arities: tuple[int, ...]) -> Problem:
    env = up_environment.Environment()
    obj_type = env.type_manager.UserType("obj")
    bool_type = env.type_manager.BoolType()
    problem = Problem("converter_observation_arities", environment=env)

    for i in range(1, max(arities) + 1):
        problem.add_object(Object(f"o{i}", obj_type, environment=env))

    for arity in arities:
        fluent_parameters = {f"f{i}": obj_type for i in range(1, arity + 1)}
        action_parameters = {f"a{i}": obj_type for i in range(1, arity + 1)}
        problem.add_fluent(Fluent(_fluent_name(arity), bool_type, environment=env, **fluent_parameters))
        problem.add_action(InstantaneousAction(_action_name(arity), _env=env, **action_parameters))

    return problem


def _build_observation_problem():
    env = up_environment.Environment()
    obj_type = env.type_manager.UserType("obj")
    pos_type = env.type_manager.UserType("pos")
    bool_type = env.type_manager.BoolType()
    problem = Problem("converter_observation", environment=env)

    for name, obj_type_name in (("o1", obj_type), ("o2", obj_type), ("p2-1", pos_type), ("p2-2", pos_type)):
        problem.add_object(Object(name, obj_type_name, environment=env))

    fluent = Fluent("obj-at", bool_type, environment=env, obj=obj_type, pos=pos_type)
    problem.add_fluent(fluent)
    return problem, fluent


def _ground_observation(problem: Problem, fluent: Fluent, obj_name: str, pos_name: str, value: bool):
    expr_manager = problem.environment.expression_manager
    grounded = expr_manager.FluentExp(
        fluent,
        (
            expr_manager.ObjectExp(problem.object(obj_name)),
            expr_manager.ObjectExp(problem.object(pos_name)),
        ),
    )
    return {grounded: expr_manager.Bool(value)}


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


def test_converter_sdrupdate_forwards_boolean_observations_to_solver():
    problem, fluent = _build_observation_problem()
    converter = UpCporConverter()
    solver = _FakeSolver()

    assert converter.SDRupdate(solver, None) is True
    assert solver.observations[-1] is None

    assert converter.SDRupdate(solver, {}) is True
    assert solver.observations[-1] is None

    false_observation = _ground_observation(problem, fluent, "o2", "p2-2", False)
    assert converter.SDRupdate(solver, false_observation) is True
    assert solver.observations[-1] == "false"

    true_observation = _ground_observation(problem, fluent, "o1", "p2-1", True)
    assert converter.SDRupdate(solver, true_observation) is True
    assert solver.observations[-1] == "true"


def test_converter_sdrupdate_rejects_multiple_observations():
    problem, fluent = _build_observation_problem()
    converter = UpCporConverter()
    solver = _FakeSolver()

    observation = {}
    observation.update(_ground_observation(problem, fluent, "o1", "p2-1", True))
    observation.update(_ground_observation(problem, fluent, "o2", "p2-2", False))

    with pytest.raises(ValueError, match="at most one grounded observation"):
        converter.SDRupdate(solver, observation)
