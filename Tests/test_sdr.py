import os
# set PYTHONNET_RUNTIME=mono
os.environ["PYTHONNET_RUNTIME"] = "mono"

# set PYTHONNET_MONO_LIBMONO="$(brew --prefix mono)/lib/libmonosgen-2.0.dylib"
os.environ["PYTHONNET_MONO_LIBMONO"] = "/opt/homebrew/opt/mono/lib/libmonosgen-2.0.dylib"

from unified_planning.io import PDDLReader
import unified_planning.environment as environment
from unified_planning.model.contingent.environment import SimulatedEnvironment
from unified_planning.shortcuts import *


if __name__ == "__main__":

    # Creating a PDDL reader
    reader = PDDLReader()

    prob_arr = [
        'blocks2',
        'blocks3',
        'blocks7',
        # 'colorballs2-2',  # PDDL parsing fails
        'doors5',
        # 'doors15',  # stuck after planning complete
        'localize5',
        # 'localize5noisy',  # PDDL parsing fails
        # 'medpks010',  # PDDL parsing fails
        'unix1',
        'wumpus05',
        # 'wumpus10'  # PDDL parsing fails
    ]

    for prob in prob_arr:
        print(f"###########################Problem: {prob} start###########################")
        # Parsing a PDDL problem from file
        problem = reader.parse_problem(
            f"../Tests/{prob}/d.pddl",
            f"../Tests/{prob}/p.pddl"
        )

        env = environment.get_environment()
        env.factory.add_engine('SDRPlanning', 'up_cpor.engine', 'SDRImpl')

        with ActionSelector(name='SDRPlanning', problem=problem) as solver:
            simulatedEnv = SimulatedEnvironment(problem)
            while not simulatedEnv.is_goal_reached():
                action = solver.get_action()
                observation = simulatedEnv.apply(action)
                solver.update(observation)

