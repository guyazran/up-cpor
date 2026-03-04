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

    prob_arr = [
        'blocks2',
        'blocks3',
        # 'blocks7',  # arity error
        # 'colorballs2-2',  # arity error
        'doors5',
        # 'doors15',  # hang after "done planning"
        # 'localize5',  # no plan found, even though a plan exists
        # 'localize5noisy',  # PDDL parsing fails
        # 'medpks010',  # PDDL parsing fails
        # 'unix1',  # arity error
        'wumpus05',
        # 'wumpus10'  # hangs on infinite planning loop
    ]  

    for prob in prob_arr:
        print(f"###########################Problem: {prob} start###########################")
        # Parsing a PDDL problem from file
        problem = reader.parse_problem(
            f"../tests-old/{prob}/d.pddl",
            f"../tests-old/{prob}/p.pddl"
        )

        env = environment.get_environment()
        env.factory.add_engine('CPORPlanning', 'up_cpor.engine', 'CPORImpl')

        with OneshotPlanner(name='CPORPlanning') as planner:
            result = planner.solve(problem)
            if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
                print(f'{planner.name} found a valid plan!')
                print(f'Success')
            else:
                print('No plan found!')
                assert False, f'Failed on problem {prob}'
