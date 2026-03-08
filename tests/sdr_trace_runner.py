import argparse
import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from cpor_test_utils import sanitize_sys_path_for_pythonnet

sanitize_sys_path_for_pythonnet()

import test_sdr
from unified_planning.model.contingent import SimulatedExecutionEnvironment

from up_test_utils import make_test_environment, parse_test_problem
from up_cpor.simulator import SDRSimulator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("up", "sdrsim"), required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    env = make_test_environment(sdr=True)
    problem = parse_test_problem(args.domain, env)

    if args.mode == "up":
        actual = test_sdr._run_online_trace(problem, SimulatedExecutionEnvironment, max_steps=120, stop_on_goal=True)
    else:
        cfg = test_sdr.SIMULATOR_CONFIG[args.domain]
        actual = test_sdr._run_online_trace(
            problem,
            SDRSimulator,
            max_steps=cfg["max_steps"],
            stop_on_goal=cfg["stop_on_goal"],
        )

    Path(args.output).write_text(json.dumps(actual, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
