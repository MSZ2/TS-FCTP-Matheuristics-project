import argparse
import random

from utils import GRASPExperiment, read_instance
from GRASP import GRASP   # vectorised + batch-parallel solver


# ============================================================
# GRASP FACTORY
# ============================================================

def build_grasp(data, iters: int, workers: int = None):
    """
    Creates a fresh GRASP instance per run.
    workers=None lets GRASP pick (cpu_count - 1); workers=1 forces
    the fully serial code path.
    """
    return GRASP(data, iters=iters, n_workers=workers)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("instance", type=str)

    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "multi"])

    parser.add_argument("--iters", type=int, default=100,
                        help="iterations per GRASP run")

    parser.add_argument("--runs", type=int, default=10,
                        help="number of runs in multi mode")

    parser.add_argument("--workers", type=int, default=None,
                        help="parallel worker processes per GRASP run "
                             "(default: cpu_count - 1; use 1 for serial)")

    args = parser.parse_args()

    # load data
    data = read_instance(args.instance)

    print("\nInstance loaded:")
    print(f"I={data.I}, J={data.J}, K={data.K}")

    # ========================================================
    # SINGLE RUN MODE
    # ========================================================
    if args.mode == "single":

        grasp = build_grasp(data, args.iters, args.workers)

        best, cost, time_taken, time_to_best, eval_count = grasp.run()

        print("\n========== SINGLE RUN RESULT ==========")
        print(f"Cost: {cost}")
        print(f"Time to best (t_i)  : {time_to_best:.2f}s")
        print(f"Total time (ttot_i) : {time_taken:.2f}s")
        print(f"Eval calls (eval_i) : {eval_count}")
        print(f"Valid: {best.is_valid()}")
        print("======================================\n")

    # ========================================================
    # MULTI RUN MODE (NEW UTILS)
    # ========================================================
    else:

        experiment = GRASPExperiment(
            grasp_class=lambda data, iters: build_grasp(data, iters, args.workers),
            data=data,
            iters=args.iters,
            instance_path=args.instance
        )

        stats = experiment.run_multiple(num_runs=args.runs, verbose=True)

        experiment.print_report(stats)
