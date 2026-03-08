from contextlib import contextmanager

import unified_planning.environment as up_environment
from unified_planning.io import PDDLReader

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


@contextmanager
def use_test_environment(env):
    previous_env = up_environment.GLOBAL_ENVIRONMENT
    up_environment.GLOBAL_ENVIRONMENT = env
    try:
        yield
    finally:
        up_environment.GLOBAL_ENVIRONMENT = previous_env
