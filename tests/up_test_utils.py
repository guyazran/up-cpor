import random
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterable

import unified_planning.environment as up_environment
from unified_planning.io import PDDLReader
from unified_planning.model import Fluent
from unified_planning.model.fluent import get_all_fluent_exp
from unified_planning.model.problem import Problem
from unified_planning.model.state import State
from unified_planning.model.contingent import SimulatedExecutionEnvironment
from unified_planning.model.contingent.contingent_problem import ContingentProblem
from unified_planning.model.contingent.execution_environment import all_smt
from unified_planning.model.mixins.fluents_set import FluentsSetMixin
from unified_planning.model.mixins.initial_state import InitialStateMixin
from unified_planning.model.mixins.metrics import MetricsMixin
from unified_planning.model.mixins.objects_set import ObjectsSetMixin
from unified_planning.model.mixins.time_model import TimeModelMixin
from unified_planning.model.mixins.user_types_set import UserTypesSetMixin
from up_cpor.caching_simulator import CachingSequentialSimulator

from domains import TESTS_DIR


@lru_cache(maxsize=None)
def _make_cached_test_environment(sdr: bool, cpor: bool, meta_cpor: bool):
    env = up_environment.Environment()
    env.credits_stream = None

    if sdr:
        env.factory.add_engine("SDRPlanning", "up_cpor.engine", "SDRImpl")
    if cpor:
        env.factory.add_engine("CPORPlanning", "up_cpor.engine", "CPORImpl")
    if meta_cpor:
        env.factory.add_meta_engine("MetaCPORPlanning", "up_cpor.engine", "CPORMetaEngineImpl")

    env._test_cache_key = (sdr, cpor, meta_cpor)
    return env


def make_test_environment(*, sdr: bool = False, cpor: bool = False, meta_cpor: bool = False):
    return _make_cached_test_environment(sdr, cpor, meta_cpor)


@lru_cache(maxsize=None)
def _parse_cached_test_problem(domain: str, sdr: bool, cpor: bool, meta_cpor: bool):
    env = _make_cached_test_environment(sdr, cpor, meta_cpor)
    reader = PDDLReader(env)
    domain_dir = TESTS_DIR / domain
    return reader.parse_problem(str(domain_dir / "d.pddl"), str(domain_dir / "p.pddl"))


def parse_test_problem(domain: str, env):
    cache_key = getattr(env, "_test_cache_key", None)
    if cache_key is None:
        reader = PDDLReader(env)
        domain_dir = TESTS_DIR / domain
        return reader.parse_problem(str(domain_dir / "d.pddl"), str(domain_dir / "p.pddl"))

    return _parse_cached_test_problem(domain, *cache_key).clone()


def _clone_problem_as_contingent(problem: Problem) -> ContingentProblem:
    contingent_problem = ContingentProblem(
        problem.name,
        problem.environment,
        initial_defaults=problem.initial_defaults,
    )
    UserTypesSetMixin._clone_to(problem, contingent_problem)
    ObjectsSetMixin._clone_to(problem, contingent_problem)
    FluentsSetMixin._clone_to(problem, contingent_problem)
    InitialStateMixin._clone_to(problem, contingent_problem)
    TimeModelMixin._clone_to(problem, contingent_problem)

    contingent_problem._actions = [action.clone() for action in problem._actions]
    contingent_problem._events = [event.clone() for event in problem._events]
    contingent_problem._processes = [process.clone() for process in problem._processes]
    contingent_problem._timed_effects = {
        timing: [effect.clone() for effect in effects]
        for timing, effects in problem._timed_effects.items()
    }
    contingent_problem._timed_goals = {
        interval: [goal for goal in goals]
        for interval, goals in problem._timed_goals.items()
    }
    contingent_problem._goals = problem._goals[:]
    contingent_problem._trajectory_constraints = problem._trajectory_constraints[:]
    contingent_problem._fluents_assigned = {
        timing: assignments.copy()
        for timing, assignments in problem._fluents_assigned.items()
    }
    MetricsMixin._clone_to(problem, contingent_problem, new_actions=contingent_problem)
    return contingent_problem


def _ground_boolean_fluents(problem: Problem) -> tuple:
    grounded_fluents = []
    for fluent in problem.fluents:
        if not fluent.type.is_bool_type():
            raise TypeError(
                "make_contingent_problem_from_possible_initial_states only supports "
                "problems with boolean fluents."
            )
        grounded_fluents.extend(get_all_fluent_exp(problem, fluent))
    return tuple(sorted(grounded_fluents, key=str))


def _make_hidden_case_tags(problem: ContingentProblem, count: int) -> tuple:
    existing_names = {fluent.name for fluent in problem.fluents}
    case_tags = []
    next_index = 0
    em = problem.environment.expression_manager

    for _ in range(count):
        while True:
            candidate_name = f"possible_initial_state_case_{next_index}"
            next_index += 1
            if candidate_name not in existing_names:
                existing_names.add(candidate_name)
                break

        case_fluent = Fluent(candidate_name, environment=problem.environment)
        problem.add_fluent(case_fluent, default_initial_value=False)
        case_tags.append(em.FluentExp(case_fluent))

    return tuple(case_tags)


def _coerce_state_boolean_value(value, index: int, fluent_exp) -> bool:
    if isinstance(value, bool):
        return value
    if hasattr(value, "is_bool_constant") and value.is_bool_constant():
        return value.is_true()
    raise TypeError(
        "Possible initial states must assign boolean constants; "
        f"state {index} returned {value!r} for {fluent_exp}."
    )


def _validate_state_problem_compatibility(
    state: State,
    problem: Problem,
    index: int,
):
    state_problem = getattr(state, "_fluent_set", None)
    state_environment = getattr(state_problem, "environment", None)
    if state_environment is not None and state_environment is not problem.environment:
        raise ValueError(
            "Possible initial states must be defined in the same environment as the "
            f"input problem; state {index} is not."
        )
    if state_problem is not None and state_problem is not problem:
        raise ValueError(
            "Possible initial states must be defined for the same problem object "
            f"passed to the helper; state {index} is not."
        )


def _explicit_state_assignments_for_problem(
    state: State,
    problem: Problem,
    index: int,
) -> dict:
    _validate_state_problem_compatibility(state, problem, index)

    if hasattr(state, "_values"):
        assignments = {}
        current = state
        while current is not None:
            for fluent_exp, value in getattr(current, "_values", {}).items():
                assignments.setdefault(
                    fluent_exp,
                    _coerce_state_boolean_value(value, index, fluent_exp),
                )
            current = getattr(current, "_father", None)

        normalized_assignments = {}
        for fluent_exp, value in assignments.items():
            if not fluent_exp.is_fluent_exp():
                raise ValueError(
                    "Possible initial states must only assign grounded fluent expressions; "
                    f"state {index} contains {fluent_exp}."
                )
            if not fluent_exp.type.is_bool_type():
                raise TypeError(
                    "make_contingent_problem_from_possible_initial_states only supports "
                    "problems with boolean fluents."
                )
            if any(not arg.is_constant() for arg in fluent_exp.args):
                raise ValueError(
                    "Possible initial states must only assign grounded fluent expressions; "
                    f"state {index} contains {fluent_exp}."
                )
            normalized_assignments[fluent_exp] = value
        return normalized_assignments

    grounded_fluents = _ground_boolean_fluents(problem)
    assignments = {}
    for fluent_exp in grounded_fluents:
        try:
            value = state.get_value(fluent_exp)
        except Exception as error:
            raise ValueError(
                "Possible initial states must define values for every grounded fluent "
                f"of the input problem; state {index} does not define {fluent_exp}."
            ) from error
        assignments[fluent_exp] = _coerce_state_boolean_value(value, index, fluent_exp)
    return assignments


def make_contingent_problem_from_possible_initial_states(
    problem: Problem,
    possible_initial_states: Iterable[State],
) -> ContingentProblem:
    """Convert a classical problem plus a finite initial-state set into a contingent problem.

    The returned problem preserves the original domain model, goals, and objects,
    but replaces the source problem's initial state with the provided states'
    explicit assignments. If a grounded fluent is missing from a state, its value
    is treated as irrelevant and left unconstrained by that state. Facts
    explicitly shared by every state are materialized as known initial values;
    the remaining uncertainty is encoded with hidden case tags: exactly one case
    tag is true initially, and each state-specific explicit literal is enforced
    only under its corresponding tag.
    """

    states = tuple(possible_initial_states)
    if not states:
        raise ValueError("At least one possible initial state is required.")

    unique_state_assignments = []
    seen_signatures = set()
    for index, state in enumerate(states):
        if not isinstance(state, State):
            raise TypeError(
                "Every possible initial state must implement unified_planning.model.state.State; "
                f"found {type(state)!r} at index {index}."
            )

        assignments = _explicit_state_assignments_for_problem(state, problem, index)
        signature = tuple((str(fluent_exp), value) for fluent_exp, value in sorted(assignments.items(), key=lambda item: str(item[0])))
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            unique_state_assignments.append(assignments)

    contingent_problem = _clone_problem_as_contingent(problem)
    contingent_problem._initial_value.clear()
    em = contingent_problem.environment.expression_manager

    shared_fluents = None
    for assignments in unique_state_assignments:
        state_fluents = set(assignments.keys())
        if shared_fluents is None:
            shared_fluents = state_fluents
        else:
            shared_fluents &= state_fluents
    assert shared_fluents is not None

    shared_assignments = {}
    for fluent_exp in sorted(shared_fluents, key=str):
        values = {assignments[fluent_exp] for assignments in unique_state_assignments}
        if len(values) == 1:
            shared_value = next(iter(values))
            default_value = contingent_problem.initial_value(fluent_exp)
            default_bool = None if default_value is None else default_value.is_true()
            if default_bool != shared_value:
                contingent_problem.set_initial_value(fluent_exp, shared_value)
            shared_assignments[fluent_exp] = shared_value

    residual_state_assignments = []
    residual_fluents = set()
    for assignments in unique_state_assignments:
        residual_assignments = {
            fluent_exp: value
            for fluent_exp, value in assignments.items()
            if fluent_exp not in shared_assignments
        }
        residual_state_assignments.append(residual_assignments)
        residual_fluents.update(residual_assignments.keys())

    if not any(residual_state_assignments):
        return contingent_problem

    for fluent_exp in residual_fluents:
        contingent_problem._hidden_fluents.add(fluent_exp)
        contingent_problem._hidden_fluents.add(em.Not(fluent_exp))

    case_tags = _make_hidden_case_tags(contingent_problem, len(residual_state_assignments))
    contingent_problem.add_oneof_initial_constraint(case_tags)

    for case_tag, assignments in zip(case_tags, residual_state_assignments):
        for fluent_exp, value in sorted(assignments.items(), key=lambda item: str(item[0])):
            literal = fluent_exp if value else em.Not(fluent_exp)
            contingent_problem.add_or_initial_constraint([em.Not(case_tag), literal])

    return contingent_problem


class DeterministicSimulatedExecutionEnvironment(SimulatedExecutionEnvironment):
    _MAX_ENUMERATED_MODEL_COUNT = 100_000
    _MIN_SYMBOLS_FOR_SCALABLE_FALLBACK = 100

    def _estimate_model_count_upper_bound(self, problem) -> int:
        count = 1

        for constraint in problem.oneof_constraints:
            count *= len(constraint)
            if count > self._MAX_ENUMERATED_MODEL_COUNT:
                return count

        for constraint in problem.or_constraints:
            count *= (2 ** len(constraint)) - 1
            if count > self._MAX_ENUMERATED_MODEL_COUNT:
                return count

        return count

    def _sample_deterministic_model_without_enumeration(self, formula, symbols):
        from pysmt.shortcuts import Not, Solver

        fixed_assignments = []
        sampled_model = {}

        with Solver(name="z3") as solver:
            solver.add_assertion(formula)

            for symbol in symbols:
                prefer_true = random.choice((True, False))
                preferred_literal = symbol if prefer_true else Not(symbol)
                fallback_literal = Not(symbol) if prefer_true else symbol

                if solver.solve(assumptions=[*fixed_assignments, preferred_literal]):
                    fixed_assignments.append(preferred_literal)
                    sampled_model[symbol] = prefer_true
                else:
                    assert solver.solve(assumptions=[*fixed_assignments, fallback_literal])
                    fixed_assignments.append(fallback_literal)
                    sampled_model[symbol] = not prefer_true

        return sampled_model

    def _randomly_set_full_initial_state(self, problem):
        from pysmt.shortcuts import And, ExactlyOne, Not, Or, Symbol

        fnode_to_symbol = {}
        symbol_to_fnode = {}
        hidden_fluents = [hf for hf in sorted(problem.hidden_fluents, key=str) if not hf.is_not()]

        for cnt, hf in enumerate(hidden_fluents):
            symbol = Symbol(f"v_{cnt}")
            fnode_to_symbol[hf] = symbol
            symbol_to_fnode[symbol] = hf

        constraints = []
        for constraint in problem.oneof_constraints:
            args = []
            for item in sorted(constraint, key=str):
                if item.is_not():
                    args.append(Not(fnode_to_symbol[item.arg(0)]))
                else:
                    args.append(fnode_to_symbol[item])
            constraints.append(ExactlyOne(args))

        for constraint in problem.or_constraints:
            args = []
            for item in sorted(constraint, key=str):
                if item.is_not():
                    args.append(Not(fnode_to_symbol[item.arg(0)]))
                else:
                    args.append(fnode_to_symbol[item])
            constraints.append(Or(args))
            if len(constraints) >= self._max_constraints:
                break

        symbols = sorted(symbol_to_fnode.keys(), key=lambda symbol: symbol.symbol_name())
        formula = And(constraints)

        # Enumerating all satisfying assignments is stable for the existing small
        # regression domains. Only switch to the scalable path for genuinely large
        # uncertainty spaces such as doors15, where the initial uncertainty is
        # roughly 15^7.
        should_enumerate_all_models = (
            len(symbols) < self._MIN_SYMBOLS_FOR_SCALABLE_FALLBACK
            or self._estimate_model_count_upper_bound(problem) <= self._MAX_ENUMERATED_MODEL_COUNT
        )

        if should_enumerate_all_models:
            models = list(all_smt(formula, symbols))
            models.sort(
                key=lambda model: tuple((symbol.symbol_name(), str(model[symbol])) for symbol in symbols)
            )
            sampled_model = random.choice(models)
        else:
            sampled_model = self._sample_deterministic_model_without_enumeration(formula, symbols)

        for symbol in symbols:
            value = sampled_model[symbol]
            if hasattr(value, "is_bool_constant"):
                assert value.is_bool_constant()
                bool_value = value.is_true()
            else:
                bool_value = bool(value)
            self._deterministic_problem.set_initial_value(symbol_to_fnode[symbol], bool_value)


@contextmanager
def use_test_environment(env):
    previous_env = up_environment.GLOBAL_ENVIRONMENT
    up_environment.GLOBAL_ENVIRONMENT = env
    try:
        yield
    finally:
        up_environment.GLOBAL_ENVIRONMENT = previous_env
