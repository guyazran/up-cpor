import random
from contextlib import contextmanager

import unified_planning.environment as up_environment
from unified_planning.io import PDDLReader
from unified_planning.model.contingent import SimulatedExecutionEnvironment
from unified_planning.model.contingent.execution_environment import all_smt

from domains import TESTS_DIR


def make_test_environment(*, sdr: bool = False, cpor: bool = False, meta_cpor: bool = False):
    env = up_environment.Environment()
    env.credits_stream = None

    if sdr:
        env.factory.add_engine("SDRPlanning", "up_cpor.engine", "SDRImpl")
    if cpor:
        env.factory.add_engine("CPORPlanning", "up_cpor.engine", "CPORImpl")
    if meta_cpor:
        env.factory.add_meta_engine("MetaCPORPlanning", "up_cpor.engine", "CPORMetaEngineImpl")

    return env


def parse_test_problem(domain: str, env):
    reader = PDDLReader(env)
    domain_dir = TESTS_DIR / domain
    return reader.parse_problem(str(domain_dir / "d.pddl"), str(domain_dir / "p.pddl"))


class DeterministicSimulatedExecutionEnvironment(SimulatedExecutionEnvironment):
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
        models = list(all_smt(And(constraints), symbols))
        models.sort(
            key=lambda model: tuple((symbol.symbol_name(), str(model[symbol])) for symbol in symbols)
        )
        sampled_model = random.choice(models)

        for symbol in symbols:
            value = sampled_model[symbol]
            assert value.is_bool_constant()
            self._deterministic_problem.set_initial_value(symbol_to_fnode[symbol], value.is_true())


@contextmanager
def use_test_environment(env):
    previous_env = up_environment.GLOBAL_ENVIRONMENT
    up_environment.GLOBAL_ENVIRONMENT = env
    try:
        yield
    finally:
        up_environment.GLOBAL_ENVIRONMENT = previous_env
