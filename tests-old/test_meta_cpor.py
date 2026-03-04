import os
import sys

# Set environment variables for Python.NET on macOS
# using the Mono runtime installed via Homebrew
if sys.platform == "darwin":
    # set PYTHONNET_RUNTIME=mono
    os.environ["PYTHONNET_RUNTIME"] = "mono"

    # set PYTHONNET_MONO_LIBMONO="$(brew --prefix mono)/lib/libmonosgen-2.0.dylib"
    os.environ["PYTHONNET_MONO_LIBMONO"] = "/opt/homebrew/opt/mono/lib/libmonosgen-2.0.dylib"

from unified_planning.io import PDDLReader
import unified_planning.environment as environment
from unified_planning.engines.results import PlanGenerationResultStatus
from unified_planning.shortcuts import *


if __name__ == "__main__":

    # Creating a PDDL reader
    reader = PDDLReader()

    prob_arr = ['blocks2', 'blocks3', 'doors5', 'wumpus05']

    for prob in prob_arr:
        print(f"###########################Problem: {prob} start###########################")
        # Parsing a PDDL problem from file
        problem = reader.parse_problem(
            f"../tests-old/{prob}/d.pddl",
            f"../tests-old/{prob}/p.pddl"
        )

        env = environment.get_environment()
        env.factory.add_meta_engine('MetaCPORPlanning', 'up_cpor.engine', 'CPORMetaEngineImpl')

        with OneshotPlanner(name='MetaCPORPlanning[tamer]') as planner:
            result = planner.solve(problem)
            if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
                print(f'{planner.name} found a valid plan!')
                print(f'Success')
            else:
                print('No plan found!')

        with OneshotPlanner(name='MetaCPORPlanning[pyperplan]') as planner:
            result = planner.solve(problem)
            if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
                print(f'{planner.name} found a valid plan!')
                print(f'Success')
            else:
                print('No plan found!')
