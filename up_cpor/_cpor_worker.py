import pickle
import sys
import traceback

from up_cpor.engine import _repair_pickled_contingent_problem, _solve_cpor_locally


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m up_cpor._cpor_worker INPUT_PICKLE RESULT_PICKLE", file=sys.stderr)
        return 2

    input_path, result_path = argv
    try:
        with open(input_path, "rb") as input_file:
            payload = pickle.load(input_file)

        problem = _repair_pickled_contingent_problem(payload["problem"])
        result = _solve_cpor_locally(
            problem,
            payload["converter"],
            payload["random_seed"],
            payload["engine_name"],
        )
        output = ("result", result)
    except BaseException:
        output = ("exception", traceback.format_exc())

    with open(result_path, "wb") as result_file:
        pickle.dump(output, result_file, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
