import pytest

from unified_planning.model import Fluent, InstantaneousAction, Problem, UPState
from unified_planning.model.contingent import SensingAction
from unified_planning.model.contingent.contingent_problem import ContingentProblem

from up_test_utils import make_contingent_problem_from_possible_initial_states
from viplan_hh.viplan_hh_cases import SORTING_BOOKS_SIMPLE, parse_problem, state_specs_to_upstates


def _case_tag_names(problem):
    return {
        fluent.name
        for fluent in problem.fluents
        if fluent.name.startswith("possible_initial_state_case_")
    }


def _build_boolean_problem_with_possible_states():
    problem = Problem("possible_states_helper")
    em = problem.environment.expression_manager

    source_only = Fluent("source_only")
    shared_true = Fluent("shared_true")
    varying = Fluent("varying")
    reached = Fluent("reached")

    problem.add_fluent(source_only, default_initial_value=False)
    problem.add_fluent(shared_true, default_initial_value=False)
    problem.add_fluent(varying, default_initial_value=False)
    problem.add_fluent(reached, default_initial_value=False)

    action = InstantaneousAction("finish")
    action.add_precondition(shared_true())
    action.add_effect(reached, True)
    problem.add_action(action)
    problem.add_goal(reached())

    # The helper must ignore the source problem's explicit initial state.
    problem.set_initial_value(source_only(), True)

    s1 = UPState(
        {
            shared_true(): em.TRUE(),
            varying(): em.TRUE(),
        },
        problem,
    )
    s0 = s1.make_child({varying(): em.FALSE()})
    return problem, source_only, shared_true, varying, reached, s0, s1


def test_make_contingent_problem_from_possible_initial_states_preserves_problem_structure():
    problem, source_only, shared_true, varying, reached, s0, s1 = (
        _build_boolean_problem_with_possible_states()
    )

    contingent_problem = make_contingent_problem_from_possible_initial_states(
        problem, (s0, s1)
    )
    em = contingent_problem.environment.expression_manager

    assert isinstance(contingent_problem, ContingentProblem)
    assert contingent_problem.name == problem.name
    assert [action.name for action in contingent_problem.actions] == [
        action.name for action in problem.actions
    ]
    assert not any(isinstance(action, SensingAction) for action in contingent_problem.actions)
    assert [str(goal) for goal in contingent_problem.goals] == [
        str(goal) for goal in problem.goals
    ]
    fluent_names = {fluent.name for fluent in contingent_problem.fluents}
    assert {fluent.name for fluent in problem.fluents}.issubset(fluent_names)

    assert contingent_problem.initial_value(shared_true()) == em.TRUE()
    assert contingent_problem.initial_value(source_only()) == em.FALSE()
    assert contingent_problem.initial_value(reached()) == em.FALSE()

    assert contingent_problem.explicit_initial_values == {shared_true(): em.TRUE()}
    assert varying() not in contingent_problem.initial_values
    assert varying() in contingent_problem.hidden_fluents
    assert em.Not(varying()) in contingent_problem.hidden_fluents

    case_tags = _case_tag_names(contingent_problem)
    assert case_tags == {
        "possible_initial_state_case_0",
        "possible_initial_state_case_1",
    }
    assert len(contingent_problem.oneof_constraints) == 1
    assert {str(branch) for branch in contingent_problem.oneof_constraints[0]} == case_tags
    assert len(contingent_problem.or_constraints) == 2
    assert {
        tuple(sorted(str(item) for item in constraint))
        for constraint in contingent_problem.or_constraints
    } == {
        ("(not possible_initial_state_case_0)", "(not varying)"),
        ("(not possible_initial_state_case_1)", "varying"),
    }


def test_make_contingent_problem_from_possible_initial_states_deduplicates_identical_states():
    problem, _, shared_true, varying, _, s0, s1 = _build_boolean_problem_with_possible_states()

    contingent_problem = make_contingent_problem_from_possible_initial_states(
        problem, (s0, s0, s1)
    )

    assert len(contingent_problem.oneof_constraints) == 1
    assert len(contingent_problem.oneof_constraints[0]) == 2
    assert len({str(branch) for branch in contingent_problem.oneof_constraints[0]}) == 2
    assert len(contingent_problem.or_constraints) == 2
    assert contingent_problem.initial_value(shared_true()).is_true()
    assert varying() in contingent_problem.hidden_fluents


def test_make_contingent_problem_from_possible_initial_states_rejects_empty_state_sets():
    problem, *_ = _build_boolean_problem_with_possible_states()

    with pytest.raises(ValueError, match="At least one possible initial state"):
        make_contingent_problem_from_possible_initial_states(problem, ())


def test_make_contingent_problem_from_possible_initial_states_rejects_states_from_other_problems():
    problem, *_ = _build_boolean_problem_with_possible_states()
    _, _, _, _, _, other_state, _ = _build_boolean_problem_with_possible_states()

    with pytest.raises(ValueError, match="same problem object"):
        make_contingent_problem_from_possible_initial_states(problem, (other_state,))


def test_make_contingent_problem_from_possible_initial_states_handles_viplan_hh_cases():
    problem = parse_problem(SORTING_BOOKS_SIMPLE)
    possible_states = state_specs_to_upstates(problem, SORTING_BOOKS_SIMPLE.representative_states)

    contingent_problem = make_contingent_problem_from_possible_initial_states(
        problem, possible_states
    )
    em = contingent_problem.environment.expression_manager
    ontop = contingent_problem.fluent("ontop")
    hardback = contingent_problem.object("hardback_1")
    table = contingent_problem.object("table_1")
    reachable = contingent_problem.fluent("reachable")
    shelf = contingent_problem.object("shelf_1")

    assert isinstance(contingent_problem, ContingentProblem)
    assert len(contingent_problem.actions) == len(problem.actions)
    assert len(contingent_problem.goals) == len(problem.goals)
    assert len(contingent_problem.all_objects) == len(problem.all_objects)
    assert not any(isinstance(action, SensingAction) for action in contingent_problem.actions)

    case_tags = _case_tag_names(contingent_problem)
    assert case_tags == {
        "possible_initial_state_case_0",
        "possible_initial_state_case_1",
    }
    assert len(contingent_problem.oneof_constraints) == 1
    assert {str(branch) for branch in contingent_problem.oneof_constraints[0]} == case_tags
    assert len(contingent_problem.or_constraints) == 1
    assert contingent_problem.initial_value(ontop(hardback, table)) == em.FALSE()
    assert reachable(shelf) in contingent_problem.hidden_fluents
    assert tuple(sorted(str(item) for item in contingent_problem.or_constraints[0])) == (
        "(not possible_initial_state_case_1)",
        "reachable(shelf_1)",
    )
