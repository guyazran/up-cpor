from typing import cast

from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from unified_planning.exceptions import UPConflictingEffectsException, UPInvalidActionError
from unified_planning.model.mixins.objects_set import ObjectsSetMixin


class CachingSequentialSimulator(UPSequentialSimulator):
    """UPSequentialSimulator subclass with transition and goal caches.

    Repeated calls to ``apply`` with the same action and the same values on
    the fluents that matter for that action skip the full evaluation and
    return a pre-computed state delta instead.  ``is_goal`` is similarly
    memoised on goal-relevant fluent values.

    The caches are per-instance and are populated lazily on first use, so
    the class is a drop-in replacement for ``UPSequentialSimulator`` wherever
    the same problem is validated against many states.
    """

    def __init__(self, problem, error_on_failed_checks: bool = True, **kwargs):
        super().__init__(problem, error_on_failed_checks=error_on_failed_checks, **kwargs)
        self._action_info_cache = {}
        self._transition_cache = {}
        self._goal_cache = {}
        self._goal_fluents = self._sorted_fluent_deps(self._problem.goals)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sorted_fluent_deps(expressions) -> tuple:
        deps = set()
        for expr in expressions:
            deps.update(expr.environment.free_vars_extractor.get(expr))
        return tuple(sorted(deps, key=str))

    def _action_info(self, grounded_action) -> dict:
        if grounded_action.simulated_effect is not None:
            return {"cacheable": False}

        expanded = []
        for effect in grounded_action.effects:
            expanded.extend(effect.expand_effect(cast(ObjectsSetMixin, self._problem)))

        relevant_exprs = list(grounded_action.preconditions)
        for eff in expanded:
            relevant_exprs.extend([eff.fluent, eff.condition, eff.value])
        relevant_exprs.extend(self._state_invariants)

        return {
            "cacheable": True,
            "preconditions": tuple(grounded_action.preconditions),
            "expanded_effects": tuple(expanded),
            "relevant_fluents": self._sorted_fluent_deps(relevant_exprs),
        }

    @staticmethod
    def _project(state, fluents) -> tuple:
        return tuple(state.get_value(f) for f in fluents)

    def _compute_transition(self, state, action_info):
        for pre in action_info["preconditions"]:
            evaluated = self._se.evaluate(pre, state)
            if not evaluated.is_bool_constant() or not evaluated.bool_constant_value():
                return None

        updated, assigned = {}, set()
        em = self._problem.environment.expression_manager
        try:
            for eff in action_info["expanded_effects"]:
                fluent, value = self._evaluate_effect(eff, state, updated, assigned, em)
                if fluent is not None:
                    updated[fluent] = value
        except (UPConflictingEffectsException, UPInvalidActionError):
            return None

        next_state = state.make_child(updated)
        for inv in self._state_invariants:
            evaluated = self._se.evaluate(inv, next_state)
            if not evaluated.is_bool_constant() or not evaluated.bool_constant_value():
                return None

        return tuple(sorted(updated.items(), key=lambda item: str(item[0])))

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def apply(self, state, action_or_action_instance, parameters=None):
        action, params = self._get_action_and_parameters(action_or_action_instance, parameters)
        action_key = (action, params)

        info = self._action_info_cache.get(action_key)
        if info is None:
            grounded = self._ground_action(action, params)
            if grounded is None:
                return None
            info = self._action_info(grounded)
            self._action_info_cache[action_key] = info

        if not info["cacheable"]:
            return super().apply(state, action_or_action_instance, parameters)

        transition_key = (action_key, self._project(state, info["relevant_fluents"]))
        cached = self._transition_cache.get(transition_key, False)
        if cached is False:
            cached = self._compute_transition(state, info)
            self._transition_cache[transition_key] = cached

        return None if cached is None else state.make_child(dict(cached))

    def is_goal(self, state):
        if not self._goal_fluents:
            return super().is_goal(state)
        goal_key = self._project(state, self._goal_fluents)
        result = self._goal_cache.get(goal_key)
        if result is None:
            result = super().is_goal(state)
            self._goal_cache[goal_key] = result
        return result
