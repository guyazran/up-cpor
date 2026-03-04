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
from unified_planning.shortcuts import *

from up_cpor.simulator import SDRSimulator

if __name__ == "__main__":

    # Creating a PDDL reader
    reader = PDDLReader()

    prob_arr = ['blocks2', 'doors5', 'wumpus05']

    for prob in prob_arr:
        print(f"###########################Problem: {prob} start###########################")
        # Parsing a PDDL problem from file
        problem = reader.parse_problem(
            f"../tests-old/{prob}/d.pddl",
            f"../tests-old/{prob}/p.pddl"
        )

        env = environment.get_environment()
        env.factory.add_engine('SDRPlanning', 'up_cpor.engine', 'SDRImpl')

        with ActionSelector(name='SDRPlanning', problem=problem) as solver:
            simulatedEnv = SDRSimulator(problem)
            while not simulatedEnv.is_goal_reached():
                action = solver.get_action()
                observation = simulatedEnv.apply(action)
                solver.update(observation)
