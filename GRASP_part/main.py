import argparse
import random

from utils import GRASPExperiment, read_instance
from GRASP import GRASP   # your solver


# ============================================================
# GRASP FACTORY
# ============================================================

def build_grasp(data, iters: int):
    """
    Creates a fresh GRASP instance per run.
    """
    return GRASP(data, iters=iters)


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

    args = parser.parse_args()

    # load data
    data = read_instance(args.instance)

    print("\nInstance loaded:")
    print(f"I={data.I}, J={data.J}, K={data.K}")

    # ========================================================
    # SINGLE RUN MODE
    # ========================================================
    if args.mode == "single":

        grasp = build_grasp(data, args.iters)

        best, cost, time_taken = grasp.run()

        print("\n========== SINGLE RUN RESULT ==========")
        print(f"Cost: {cost}")
        print(f"Time: {time_taken:.2f}s")
        print(f"Valid: {best.is_valid()}")
        print("======================================\n")

    # ========================================================
    # MULTI RUN MODE (NEW UTILS)
    # ========================================================
    else:

        experiment = GRASPExperiment(
            grasp_class=GRASP,
            data=data,
            iters=args.iters,
            instance_path=args.instance
        )

        stats = experiment.run_multiple(num_runs=args.runs, verbose=True)

        experiment.print_report(stats)