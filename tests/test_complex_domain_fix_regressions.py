import os
import re
import sys

if sys.platform == "darwin":
    os.environ["PYTHONNET_RUNTIME"] = "mono"
    os.environ["PYTHONNET_MONO_LIBMONO"] = "/opt/homebrew/opt/mono/lib/libmonosgen-2.0.dylib"

import clr
import unified_planning.environment as up_environment
from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from unified_planning.model import Fluent, InstantaneousAction, Object, Problem, UPState, Variable
from unified_planning.model.contingent import SensingAction
from unified_planning.model.contingent.contingent_problem import ContingentProblem

from up_cpor.converter import UpCporConverter
import up_cpor.engine as cpor_engine
import System
from up_test_utils import make_contingent_problem_from_possible_initial_states
from System.Collections.Generic import Dictionary, HashSet, ISet, List
from System.Reflection import BindingFlags
from CPORLib.Algorithms import CPORPlanner
from CPORLib.FFCS import Array2D, Constants, InputConverter, SparseArray
from CPORLib.LogicalUtilities import CompoundFormula, GroundedPredicate, Predicate, PredicateFormula
from CPORLib.Parsing import CPORStack
from CPORLib.PlanningModel import Domain as CporDomain, PartiallySpecifiedState, PlanningAction, Problem as CporProblem
from CPORLib.Tools import Options


def _normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\)\(", ") (", text)
    return text


def _convert_problem(problem):
    converter = UpCporConverter()
    c_domain = converter.createDomain(problem)
    c_problem = converter.createProblem(problem, c_domain)
    return converter, c_domain, c_problem


def _find_c_action(c_domain, name: str):
    for action in c_domain.Actions:
        if str(action.Name) == name:
            return action
    raise AssertionError(f"Missing translated action: {name}")


def _get_private_method(instance, name: str):
    method = instance.GetType().GetMethod(
        name,
        BindingFlags.Instance | BindingFlags.NonPublic,
    )
    assert method is not None, f"Missing private method {name}"
    return method


def _invoke_private(instance, name: str, *args):
    return _get_private_method(instance, name).Invoke(instance, args)


def _predicate_signatures(predicates) -> set[tuple[str, bool, tuple[str, ...]]]:
    return {
        (
            str(predicate.Name),
            bool(predicate.Negation),
            tuple(str(constant.Name) for constant in predicate.Constants),
        )
        for predicate in predicates
    }


def _state_signatures(state) -> set[tuple[str, bool, tuple[str, ...]]]:
    return _predicate_signatures(state)


def _build_quantified_translation_problem():
    env = up_environment.Environment()
    em = env.expression_manager
    bool_type = env.type_manager.BoolType()
    obj_type = env.type_manager.UserType("obj")

    problem = ContingentProblem("quantified_translation", env)
    o1 = Object("o1", obj_type, environment=env)
    o2 = Object("o2", obj_type, environment=env)
    problem.add_object(o1)
    problem.add_object(o2)

    marked = Fluent("marked", bool_type, environment=env, item=obj_type)
    done = Fluent("done", bool_type, environment=env, item=obj_type)
    problem.add_fluent(marked, default_initial_value=False)
    problem.add_fluent(done, default_initial_value=False)

    action = InstantaneousAction("act", _env=env, item=obj_type, other=obj_type)
    item = action.parameter("item")
    other = action.parameter("other")
    variable = Variable("v", obj_type, environment=env)

    exists_marked = em.Exists(
        em.FluentExp(marked, (em.VariableExp(variable),)),
        variable,
    )
    action.add_precondition(
        em.Or(
            em.Equals(em.ParameterExp(item), em.ParameterExp(other)),
            exists_marked,
        )
    )
    action.add_precondition(
        em.Forall(
            em.FluentExp(marked, (em.VariableExp(variable),)),
            variable,
        )
    )
    action.add_effect(
        em.FluentExp(done, (em.VariableExp(variable),)),
        True,
        forall=(variable,),
    )
    problem.add_action(action)
    problem.add_goal(em.FluentExp(done, (em.ObjectExp(o1),)))
    return problem


def _build_supported_kind_problem():
    env = up_environment.Environment()
    em = env.expression_manager
    bool_type = env.type_manager.BoolType()
    obj_type = env.type_manager.UserType("obj")

    problem = ContingentProblem("supported_kind", env)
    o1 = Object("o1", obj_type, environment=env)
    o2 = Object("o2", obj_type, environment=env)
    problem.add_object(o1)
    problem.add_object(o2)

    marked = Fluent("marked", bool_type, environment=env, item=obj_type)
    done = Fluent("done", bool_type, environment=env, item=obj_type)
    problem.add_fluent(marked, default_initial_value=False)
    problem.add_fluent(done, default_initial_value=False)

    action = InstantaneousAction("act", _env=env, item=obj_type, other=obj_type)
    item = action.parameter("item")
    other = action.parameter("other")
    variable = Variable("v", obj_type, environment=env)

    action.add_precondition(
        em.Or(
            em.Equals(em.ParameterExp(item), em.ParameterExp(other)),
            em.FluentExp(marked, (em.ParameterExp(item),)),
        )
    )
    action.add_precondition(
        em.Forall(
            em.FluentExp(marked, (em.VariableExp(variable),)),
            variable,
        )
    )
    action.add_effect(
        em.FluentExp(done, (em.VariableExp(variable),)),
        True,
        forall=(variable,),
    )
    problem.add_action(action)
    problem.add_goal(em.FluentExp(done, (em.ObjectExp(o1),)))
    return problem


def _build_hidden_goal_problem():
    env = up_environment.Environment()
    em = env.expression_manager
    bool_type = env.type_manager.BoolType()

    problem = ContingentProblem("hidden_goal_problem", env)
    visible_goal = Fluent("visible_goal", bool_type, environment=env)
    hidden_goal = Fluent("hidden_goal", bool_type, environment=env)
    problem.add_fluent(visible_goal, default_initial_value=False)
    problem.add_fluent(hidden_goal, default_initial_value=False)
    problem.set_initial_value(visible_goal(), True)
    problem.add_unknown_initial_constraint(em.FluentExp(hidden_goal))
    problem.add_goal(em.FluentExp(visible_goal))
    problem.add_goal(em.FluentExp(hidden_goal))
    return problem


def _build_disjunctive_precondition_problem():
    env = up_environment.Environment()
    em = env.expression_manager
    bool_type = env.type_manager.BoolType()

    problem = ContingentProblem("disjunctive_preconditions", env)
    p = Fluent("p", bool_type, environment=env)
    q = Fluent("q", bool_type, environment=env)
    done = Fluent("done", bool_type, environment=env)
    problem.add_fluent(p, default_initial_value=False)
    problem.add_fluent(q, default_initial_value=False)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_unknown_initial_constraint(em.Or(em.FluentExp(p), em.FluentExp(q)))

    action = InstantaneousAction("finish", _env=env)
    action.add_precondition(em.Or(em.FluentExp(p), em.FluentExp(q)))
    action.add_effect(em.FluentExp(done), True)
    problem.add_action(action)
    problem.add_goal(em.FluentExp(done))
    return problem


def _build_precondition_failure_problem():
    env = up_environment.Environment()
    em = env.expression_manager
    bool_type = env.type_manager.BoolType()

    problem = ContingentProblem("precondition_failure", env)
    flag = Fluent("flag", bool_type, environment=env)
    done = Fluent("done", bool_type, environment=env)
    problem.add_fluent(flag, default_initial_value=False)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_unknown_initial_constraint(em.FluentExp(flag))

    action = InstantaneousAction("finish", _env=env)
    action.add_precondition(em.FluentExp(flag))
    action.add_effect(em.FluentExp(done), True)
    problem.add_action(action)
    problem.add_goal(em.FluentExp(done))
    return problem


def _build_initial_values_problem():
    env = up_environment.Environment()
    bool_type = env.type_manager.BoolType()
    problem = ContingentProblem("initial_values_regression", env)

    visible_false = Fluent("visible_false", bool_type, environment=env)
    hidden_false = Fluent("hidden_false", bool_type, environment=env)
    visible_true = Fluent("visible_true", bool_type, environment=env)
    problem.add_fluent(visible_false, default_initial_value=False)
    problem.add_fluent(hidden_false, default_initial_value=False)
    problem.add_fluent(visible_true, default_initial_value=False)
    problem.set_initial_value(visible_false(), False)
    problem.set_initial_value(hidden_false(), False)
    problem.set_initial_value(visible_true(), True)
    problem._hidden_fluents.add(hidden_false())
    return problem


def _build_case_problem(possible_states):
    problem = Problem("case_problem")
    em = problem.environment.expression_manager
    x = Fluent("x")
    y = Fluent("y")
    for fluent in (x, y):
        problem.add_fluent(fluent, default_initial_value=False)
    states = []
    for x_value, y_value in possible_states:
        states.append(
            UPState(
                {
                    x(): em.TRUE() if x_value else em.FALSE(),
                    y(): em.TRUE() if y_value else em.FALSE(),
                },
                problem,
            )
        )
    return make_contingent_problem_from_possible_initial_states(problem, tuple(states))


def _build_case_problem_with_three_fluents(possible_states):
    problem = Problem("case_problem_three_fluents")
    em = problem.environment.expression_manager
    x = Fluent("x")
    y = Fluent("y")
    z = Fluent("z")
    for fluent in (x, y, z):
        problem.add_fluent(fluent, default_initial_value=False)

    states = []
    for x_value, y_value, z_value in possible_states:
        states.append(
            UPState(
                {
                    x(): em.TRUE() if x_value else em.FALSE(),
                    y(): em.TRUE() if y_value else em.FALSE(),
                    z(): em.TRUE() if z_value else em.FALSE(),
                },
                problem,
            )
        )
    return make_contingent_problem_from_possible_initial_states(problem, tuple(states))


def _build_type_hierarchy_problem():
    env = up_environment.Environment()
    bool_type = env.type_manager.BoolType()
    parent_type = env.type_manager.UserType("parent")
    child_type = env.type_manager.UserType("child", parent_type)

    problem = ContingentProblem("hierarchy_regression", env)
    problem.add_object(Object("c1", child_type, environment=env))

    done = Fluent("done", bool_type, environment=env)
    problem.add_fluent(done, default_initial_value=False)
    action = InstantaneousAction("use", _env=env, item=parent_type)
    action.add_effect(done, True)
    problem.add_action(action)
    return problem


def _build_promoted_conditional_effect_problem():
    env = up_environment.Environment()
    em = env.expression_manager
    bool_type = env.type_manager.BoolType()

    problem = ContingentProblem("promoted_conditions", env)
    ready = Fluent("ready", bool_type, environment=env)
    left = Fluent("left", bool_type, environment=env)
    right = Fluent("right", bool_type, environment=env)
    out_left = Fluent("out_left", bool_type, environment=env)
    out_right = Fluent("out_right", bool_type, environment=env)
    for fluent in (ready, left, right, out_left, out_right):
        problem.add_fluent(fluent, default_initial_value=False)

    action = InstantaneousAction("act", _env=env)
    action.add_effect(
        em.FluentExp(out_left),
        True,
        em.And(em.FluentExp(ready), em.FluentExp(left)),
    )
    action.add_effect(
        em.FluentExp(out_right),
        True,
        em.And(em.FluentExp(ready), em.FluentExp(right)),
    )
    problem.add_action(action)
    return problem


def _build_observation_branch_problem():
    env = up_environment.Environment()
    em = env.expression_manager
    bool_type = env.type_manager.BoolType()
    problem = ContingentProblem("observation_branch_problem", env)
    flag = Fluent("flag", bool_type, environment=env)
    done = Fluent("done", bool_type, environment=env)
    problem.add_fluent(flag, default_initial_value=False)
    problem.add_fluent(done, default_initial_value=False)
    sense = SensingAction("sense", _env=env)
    sense.add_observed_fluent(em.FluentExp(flag))
    problem.add_action(sense)
    finish = InstantaneousAction("finish", _env=env)
    finish.add_effect(em.FluentExp(done), True)
    problem.add_action(finish)
    return problem, em.FluentExp(flag)


class _FakeCporPlanNode:
    def __init__(
        self,
        action: str,
        node_id: int,
        *,
        single_child=None,
        false_child=None,
        true_child=None,
    ):
        self.Action = action
        self.ID = node_id
        self.SingleChild = single_child
        self.FalseObservationChild = false_child
        self.TrueObservationChild = true_child


class _FakeAction:
    def __init__(self, name: str):
        self.name = name


class _FakeActionInstance:
    def __init__(self, name: str, *actual_parameters):
        self.action = _FakeAction(name)
        self.actual_parameters = actual_parameters


def test_cpor_supported_kind_accepts_complex_contingent_features():
    problem = _build_supported_kind_problem()

    assert problem.kind.has_disjunctive_conditions()
    assert problem.kind.has_equalities()
    assert problem.kind.has_universal_conditions()
    assert problem.kind.has_forall_effects()
    assert cpor_engine.CPORImpl.supports(problem.kind)


def test_converter_translates_equalities_quantifiers_and_forall_effects():
    problem = _build_quantified_translation_problem()
    _, c_domain, _ = _convert_problem(problem)

    translated = _normalize(str(_find_c_action(c_domain, "act")))

    assert "(= ?item ?other)" in translated
    assert "(or (= ?item ?other) (marked o1) (marked o2))" in translated
    assert translated.count("(marked o1)") >= 2
    assert translated.count("(marked o2)") >= 2
    assert "(done o1)" in translated
    assert "(done o2)" in translated


def test_converter_promotes_common_conditional_effect_conditions():
    problem = _build_promoted_conditional_effect_problem()
    _, c_domain, _ = _convert_problem(problem)

    translated = _normalize(str(_find_c_action(c_domain, "act")))

    assert ":precondition (ready)" in translated
    assert "(when (and (ready) (left)) (out_left))" in translated
    assert "(when (and (ready) (right)) (out_right))" in translated


def test_converter_observation_branches_preserve_false_and_true_values():
    problem, observed_fluent = _build_observation_branch_problem()
    converter = UpCporConverter()
    false_leaf = _FakeCporPlanNode("(:action finish\n)", 2)
    true_leaf = _FakeCporPlanNode("(:action finish\n)", 3)
    root = _FakeCporPlanNode(
        "(:action sense\n :observe (flag)\n)",
        1,
        false_child=false_leaf,
        true_child=true_leaf,
    )

    converted = converter.createActionTree(root, problem)

    assert len(converted.children) == 2
    false_observation, _ = converted.children[0]
    true_observation, _ = converted.children[1]
    assert false_observation == {
        observed_fluent: problem.environment.expression_manager.FALSE()
    }
    assert true_observation == {
        observed_fluent: problem.environment.expression_manager.TRUE()
    }


def test_create_problem_keeps_explicit_false_knowns_and_skips_hidden_initial_fluents():
    problem = _build_initial_values_problem()
    _, _, c_problem = _convert_problem(problem)

    known_predicates = _predicate_signatures(c_problem.Known)

    assert ("visible_false", True, ()) in known_predicates
    assert ("visible_true", False, ()) in known_predicates
    assert ("hidden_false", True, ()) not in known_predicates


def test_create_problem_infers_missing_case_literals_from_defaults():
    contingent_problem = _build_case_problem(((True, False), (False, True)))
    _, _, c_problem = _convert_problem(contingent_problem)

    hidden_strings = {_normalize(str(hidden)) for hidden in c_problem.Hidden}

    assert hidden_strings == {
        "(oneof (possible_initial_state_case_0) (possible_initial_state_case_1))",
        "(or (not (possible_initial_state_case_0)) (x))",
        "(or (not (possible_initial_state_case_1)) (y))",
        "(or (not (possible_initial_state_case_0)) (not (y)))",
        "(or (not (possible_initial_state_case_1)) (not (x)))",
    }


def test_create_problem_compacts_case_tag_constraints_when_exclusions_are_fewer_than_cases():
    contingent_problem = _build_case_problem(((False, False), (True, False), (False, True)))
    _, _, c_problem = _convert_problem(contingent_problem)

    hidden_strings = [_normalize(str(hidden)) for hidden in c_problem.Hidden]

    assert hidden_strings == ["(or (not (x)) (not (y)))"]


def test_domain_ground_action_by_name_accepts_subtype_constants_for_supertype_parameters():
    problem = _build_type_hierarchy_problem()
    _, c_domain, _ = _convert_problem(problem)

    grounded = c_domain.GroundActionByName(["use", "c1"], HashSet[Predicate](), False)

    assert grounded is not None
    assert str(grounded.Name).startswith("use")


def test_tagged_goal_distinguishes_always_known_and_hidden_goal_literals():
    problem = _build_hidden_goal_problem()
    _, _, c_problem = _convert_problem(problem)

    _, _, tagged_problem = (
        c_problem.GetInitialBelief()
        .GetPartiallySpecifiedState()
        .GetTaggedDomainAndProblem(Options.DeadendStrategies.Lazy, False)
    )
    goal_names = {str(predicate.Name) for predicate in tagged_problem.Goal.GetAllPredicates()}

    assert "visible_goal" in goal_names
    assert "Khidden_goal" in goal_names
    assert "hidden_goal" not in goal_names


def test_apply_offline_precondition_failure_revises_belief_and_keeps_lazy_goal_compilation():
    problem = _build_precondition_failure_problem()
    _, _, c_problem = _convert_problem(problem)
    pss = c_problem.GetInitialBelief().GetPartiallySpecifiedState()

    _, failed, true_state, false_state = pss.ApplyOffline("finish")
    observed = _predicate_signatures(pss.Observed)

    assert failed
    assert true_state is None
    assert false_state is None

    _, _, tagged_problem = pss.GetTaggedDomainAndProblem(
        Options.DeadendStrategies.Lazy,
        True,
    )
    goal_names = {str(predicate.Name) for predicate in tagged_problem.Goal.GetAllPredicates()}
    assert "done" in goal_names
    assert not all(name.startswith("KNot") for name in goal_names)


def test_apply_offline_precondition_failure_does_not_pollute_observed():
    # Regression: AddObserved(fFailedPreconditions) was removed from ApplyOffline.
    # A precondition failure must leave m_lObserved unchanged; any growth would
    # allow the planner to build plans that rely on failure-derived "knowledge"
    # that ValidatePlanGraph cannot reproduce from a fresh belief state.
    problem = _build_precondition_failure_problem()
    _, _, c_problem = _convert_problem(problem)
    pss = c_problem.GetInitialBelief().GetPartiallySpecifiedState()

    observed_before = _predicate_signatures(pss.Observed)
    _, failed, _, _ = pss.ApplyOffline("finish")
    observed_after = _predicate_signatures(pss.Observed)

    assert failed
    assert observed_after == observed_before


def test_tagged_actions_preserve_disjunctive_knowledge_preconditions():
    problem = _build_disjunctive_precondition_problem()
    _, _, c_problem = _convert_problem(problem)

    _, tagged_domain, _ = (
        c_problem.GetInitialBelief()
        .GetPartiallySpecifiedState()
        .GetTaggedDomainAndProblem(Options.DeadendStrategies.Lazy, False)
    )
    translated = _normalize(str(_find_c_action(tagged_domain, "finish")))

    assert "(or (p) (q))" in translated
    assert "(or (Kp) (Kq))" in translated


def test_cporplanner_initial_tag_count_ignores_synthetic_case_tag_volume():
    contingent_problem = _build_case_problem_with_three_fluents(
        (
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
        )
    )
    _, c_domain, c_problem = _convert_problem(contingent_problem)
    planner = CPORPlanner(c_domain, c_problem)

    initial_tags_count = _invoke_private(planner, "GetInitialTagsCount")

    assert initial_tags_count == 3


def test_belief_state_prefers_meaningful_predicates_over_synthetic_case_tags():
    contingent_problem = _build_case_problem(((True, False), (False, True)))
    _, _, c_problem = _convert_problem(contingent_problem)
    belief_state = c_problem.GetInitialBelief()

    candidates = List[Predicate]()
    candidates.Add(GroundedPredicate("possible_initial_state_case_0"))
    candidates.Add(GroundedPredicate("x"))

    chosen = _invoke_private(belief_state, "ChoosePreferredPredicate", candidates)

    assert str(chosen.Name) == "x"


def test_belief_state_prefers_goal_refuting_states_when_goal_is_not_known():
    problem = _build_precondition_failure_problem()
    _, _, c_problem = _convert_problem(problem)
    belief_state = c_problem.GetInitialBelief()

    satisfying = HashSet[Predicate]()
    satisfying.Add(GroundedPredicate("done"))
    refuting = HashSet[Predicate]()
    refuting.Add(GroundedPredicate("done").Negate())

    states = List[ISet[Predicate]]()
    states.Add(satisfying)
    states.Add(refuting)

    _invoke_private(belief_state, "PreferUnsatisfiedGoalState", states)

    assert ("done", True, ()) in _state_signatures(states[0])


def test_belief_state_can_force_meaningful_disagreement_instead_of_case_tag_disagreement():
    contingent_problem = _build_case_problem(((True, False), (False, False)))
    _, _, c_problem = _convert_problem(contingent_problem)
    belief_state = c_problem.GetInitialBelief()

    state0 = HashSet[Predicate]()
    state0.Add(GroundedPredicate("x"))
    state0.Add(GroundedPredicate("possible_initial_state_case_0"))
    state1 = HashSet[Predicate]()
    state1.Add(GroundedPredicate("x"))
    state1.Add(GroundedPredicate("possible_initial_state_case_1"))

    states = List[ISet[Predicate]]()
    states.Add(state0)
    states.Add(state1)

    alternative_state = _invoke_private(
        belief_state,
        "ChooseStateWithMeaningfulDisagreement",
        states,
    )

    assert alternative_state is not None
    assert ("x", False, ()) not in _state_signatures(alternative_state)
    assert ("x", True, ()) in _state_signatures(alternative_state)


def test_get_next_state_handles_goal_states_without_replanning():
    env = up_environment.Environment()
    em = env.expression_manager
    bool_type = env.type_manager.BoolType()
    problem = ContingentProblem("goal_state_regression", env)
    done = Fluent("done", bool_type, environment=env)
    problem.add_fluent(done, default_initial_value=False)
    problem.set_initial_value(done(), True)
    problem.add_goal(em.FluentExp(done))

    _, c_domain, c_problem = _convert_problem(problem)
    planner = CPORPlanner(c_domain, c_problem)
    state_stack = CPORStack[PartiallySpecifiedState]()
    state_stack.Push(c_problem.GetInitialBelief().GetPartiallySpecifiedState())
    closed_states = List[PartiallySpecifiedState]()
    visited_states = Dictionary[PartiallySpecifiedState, PartiallySpecifiedState]()

    next_state, done_flag, handled = planner.GetNextState(
        state_stack,
        closed_states,
        visited_states,
    )

    assert next_state is not None
    assert done_flag
    assert handled


def test_engine_collapses_consecutive_navigation_steps():
    actions = [
        _FakeActionInstance("navigate-to", "document_1"),
        _FakeActionInstance("navigate-to", "cabinet_1"),
        _FakeActionInstance("grasp", "document_1"),
    ]

    collapsed = cpor_engine._collapse_consecutive_navigations(actions)

    assert [action.action.name for action in collapsed] == ["navigate-to", "grasp"]
    assert collapsed[0].actual_parameters == ("cabinet_1",)


def test_engine_rewrites_container_stash_patterns():
    actions = [
        _FakeActionInstance("navigate-to", "document_1"),
        _FakeActionInstance("grasp", "document_1"),
        _FakeActionInstance("navigate-to", "cabinet_1"),
        _FakeActionInstance("place-next-to", "document_1", "cabinet_1"),
        _FakeActionInstance("open-container", "cabinet_1"),
        _FakeActionInstance("navigate-to", "document_1"),
        _FakeActionInstance("grasp", "document_1"),
        _FakeActionInstance("navigate-to", "cabinet_1"),
        _FakeActionInstance("place-inside", "document_1", "cabinet_1"),
    ]

    rewritten = cpor_engine._rewrite_container_stash_pattern(actions)
    assert rewritten is not None
    assert [action.action.name for action in rewritten] == [
        "navigate-to",
        "open-container",
        "navigate-to",
        "grasp",
        "navigate-to",
        "place-inside",
    ]


def test_up_sequential_simulator_caches_are_shared_across_instances_for_the_same_problem():
    problem = Problem("shared_simulator_cache")
    done = Fluent("done")
    problem.add_fluent(done, default_initial_value=False)

    finish = InstantaneousAction("finish")
    finish.add_effect(done, True)
    problem.add_action(finish)
    problem.add_goal(done())

    simulator_a = UPSequentialSimulator(problem, error_on_failed_checks=False)
    simulator_b = UPSequentialSimulator(problem, error_on_failed_checks=False)

    assert simulator_a._up_cpor_action_info_cache is simulator_b._up_cpor_action_info_cache
    assert simulator_a._up_cpor_transition_cache is simulator_b._up_cpor_transition_cache
    assert simulator_a._up_cpor_goal_cache is simulator_b._up_cpor_goal_cache

    initial_state = simulator_a.get_initial_state()
    next_state = simulator_a.apply(initial_state, finish)

    assert next_state is not None
    assert len(simulator_a._up_cpor_transition_cache) > 0
    assert len(simulator_b._up_cpor_transition_cache) == len(simulator_a._up_cpor_transition_cache)
    assert simulator_b.is_goal(next_state)


def test_ff_utils_preserve_initializers_and_large_table_limits():
    matrix = Array2D[int](1)
    matrix.Init(0, 3, 7)
    sparse = SparseArray[int](2000)

    assert Constants.MAX_PREDICATES >= 16384
    assert Constants.MAX_OPERATORS >= 16384
    assert Constants.MAX_RELEVANT_FACTS >= 500000
    assert matrix[0, 0] == 7
    assert matrix[0, 2] == 7
    assert sparse.Get(123) == 0


def test_input_converter_handles_empty_compound_formulas():
    converter = InputConverter()
    convert_formula_method = next(
        method
        for method in converter.GetType().GetMethods(BindingFlags.Instance | BindingFlags.NonPublic)
        if method.Name == "Convert"
        and len(method.GetParameters()) == 1
        and method.GetParameters()[0].ParameterType.FullName == "CPORLib.LogicalUtilities.Formula"
    )

    converted = convert_formula_method.Invoke(converter, (CompoundFormula("and"),))

    assert converted is not None
