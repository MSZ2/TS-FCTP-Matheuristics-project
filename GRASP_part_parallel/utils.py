import numpy as np
import os
import platform
import time
from dataclasses import dataclass
@dataclass
class TSFCTPData:
    I: int          # suppliers
    J: int          # DCs
    K: int          # customers
    s: np.ndarray   # supply[I]
    d: np.ndarray   # demand[K]
    b: np.ndarray   # var cost i->j
    f: np.ndarray   # fixed cost i->j
    c: np.ndarray   # var cost j->k
    g: np.ndarray   # fixed cost j->k

    def validate(self):
        assert self.s.sum() >= self.d.sum()

def read_instance(filename):
    with open(filename, 'r') as f:
        nums = [int(x) for line in f for x in line.split()]
    it = iter(nums)
    I, J, K = next(it), next(it), next(it)
    s = np.array([next(it) for _ in range(I)], dtype=int)
    d = np.array([next(it) for _ in range(K)], dtype=int)
    b = np.array([[next(it) for _ in range(J)] for _ in range(I)], dtype=int)
    f = np.array([[next(it) for _ in range(J)] for _ in range(I)], dtype=int)
    c = np.array([[next(it) for _ in range(K)] for _ in range(J)], dtype=int)
    g = np.array([[next(it) for _ in range(K)] for _ in range(J)], dtype=int)
    print(f"I={I}, J={J}, K={K}, supply={s.sum()}, demand={d.sum()}")
    data = TSFCTPData(I, J, K, s, d, b, f, c, g)
    data.validate()
    return data

# ============================================================
# OPTIONAL: OPTIMAL VALUE LOADING
# ============================================================

def get_results_filename(instance_path: str):
    base = os.path.basename(instance_path)
    # Resolve relative to this file's location (repo_root/GRASP_part_parallel/..),
    # not the caller's CWD -- main.py is documented to run from inside
    # GRASP_part_parallel/, where "CPLEX_part/results/..." doesn't exist.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prefix = os.path.join(repo_root, 'CPLEX_part', 'results')
    if base.startswith("small_"):
        return os.path.join(prefix, "results_small.txt")
    if base.startswith("medium_"):
        return os.path.join(prefix, "results_medium.txt")
    if base.startswith("large_"):
        return os.path.join(prefix, "results_large.txt")

    return None


def load_optimal_for_instance(instance_path: str):
    """
    Returns (optimal_value, optimal_time) if available.
    Safe fallback: (None, None)
    """
    file = get_results_filename(instance_path)

    if file is None or not os.path.exists(file):
        return None, None

    target = os.path.basename(instance_path)

    try:
        with open(file, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 4:
                    continue

                # Match by basename: the results file stores paths like
                # "generated_instances/medium_1.txt", while callers may
                # pass e.g. "../generated_instances/medium_1.txt".
                if os.path.basename(parts[0]) == target:
                    try:
                        return float(parts[2]), float(parts[3])
                    except:
                        return None, None
    except:
        pass

    return None, None


# ============================================================
# EXPERIMENT WRAPPER
# ============================================================

class GRASPExperiment:
    """
    Wrapper over GRASP for single + multi run experiments.
    """

    def __init__(self, grasp_class, data, iters, instance_path=None):
        self.grasp_class = grasp_class
        self.data = data
        self.iters = iters
        self.instance_path = instance_path

        self.optimal_value, self.optimal_time = (
            load_optimal_for_instance(instance_path)
            if instance_path else (None, None)
        )

        self.results = []

    # --------------------------------------------------------
    # SINGLE RUN
    # --------------------------------------------------------
    def single_run(self, run_id: int, seed: int):
        import random

        random.seed(seed)

        grasp = self.grasp_class(self.data, iters=self.iters)

        # GRASP.run() returns (best, best_cost, total_time, time_to_best,
        # eval_count) -- total_time/time_to_best are measured internally
        # by GRASP itself (wall-clock at the moment `best` was first
        # found), not re-derived here.
        best, best_cost, total_time, time_to_best, eval_count = grasp.run()

        return {
            "run_id": run_id,
            "best_cost": best_cost,
            "time_to_best": time_to_best,
            "total_time": total_time,
            "eval_count": eval_count,
            "iterations": self.iters
        }

    # --------------------------------------------------------
    # MULTI RUN
    # --------------------------------------------------------
    def run_multiple(self, num_runs: int = 10, verbose: bool = True):
        self.results = []

        for r in range(num_runs):
            seed = 42 + r * 100

            if verbose:
                print(f"\n--- Run {r+1}/{num_runs} seed={seed} ---")

            res = self.single_run(r, seed)
            self.results.append(res)

            if verbose:
                print(f"Best cost: {res['best_cost']}")
                print(f"Time     : {res['total_time']:.2f}s")

        return self._compute_stats(self.results)

    # --------------------------------------------------------
    # STATS
    # --------------------------------------------------------
    def _compute_stats(self, results):
        costs = [r['best_cost'] for r in results]
        best_ma = min(costs)

        # Bestsol: optimal/best-known from the literature if available
        # and better than what MA found, otherwise the MA's own best.
        if self.optimal_value is not None and self.optimal_value <= best_ma:
            ref = self.optimal_value
        else:
            ref = best_ma

        gaps = [100 * abs(c - ref) / ref for c in costs]

        stats = {
            'num_runs': len(results),
            'best_ma': best_ma,
            'bestsol': ref,
            'optimal': self.optimal_value,
            'opt_time': self.optimal_time,

            'mean_t_best': np.mean([r['time_to_best'] for r in results]),
            'mean_t_total': np.mean([r['total_time'] for r in results]),
            'mean_eval': np.mean([r['eval_count'] for r in results]),

            'mean_gap': np.mean(gaps),
            'std_gap': np.std(gaps),

            'results': results,
            'gaps': gaps
        }

        return stats

    # --------------------------------------------------------
    # PRINT REPORT -- builds one report text, then prints AND
    # saves it, so the file and the terminal always match.
    # --------------------------------------------------------
    def print_report(self, stats):
        d = self.data
        lines = []

        def add(line=""):
            lines.append(line)

        add("=" * 60)
        add("GRASP REPORT")
        add("=" * 60)
        add(f"Instance: {self.instance_path}")
        add(f"Dimensions: I={d.I}, J={d.J}, K={d.K}")
        add(f"Platform (MA run): {platform.platform()}, "
            f"{platform.processor() or platform.machine()}, "
            f"{os.cpu_count()} CPUs")
        add()
        add(f"Optimal / best-known (if available): {self.optimal_value}")
        add(f"Time to obtain optimal / best-known : {self.optimal_time}")
        add(f"Bestsol (reference used for gap %)  : {stats['bestsol']}")
        add(f"Best MA cost across {stats['num_runs']} runs      : {stats['best_ma']}")
        add()
        add("--- Per-run results ---")
        add(f"{'run':>4} {'sol_i':>12} {'t_i(s)':>10} {'ttot_i(s)':>10} "
            f"{'eval_i':>10} {'gap_i(%)':>10}")
        for i, r in enumerate(stats['results']):
            add(f"{i+1:>4} {r['best_cost']:>12} {r['time_to_best']:>10.2f} "
                f"{r['total_time']:>10.2f} {r['eval_count']:>10} "
                f"{stats['gaps'][i]:>10.2f}")
        add()
        add(f"--- Averages over {stats['num_runs']} runs ---")
        add(f"Mean time to best (t)     : {stats['mean_t_best']:.2f} s")
        add(f"Mean total time (ttot)    : {stats['mean_t_total']:.2f} s")
        add(f"Mean eval calls (eval)    : {stats['mean_eval']:.1f}")
        add(f"Mean gap (agap) ± std (σ) : {stats['mean_gap']:.2f} ± {stats['std_gap']:.2f} %")

        report_text = "\n".join(lines)
        print("\n" + report_text)

        os.makedirs("reports", exist_ok=True)
        rep_name = os.path.join(
            "reports",
            f"GRASP_report_{os.path.basename(self.instance_path)}"
        )
        with open(rep_name, "w") as f:
            f.write(report_text + "\n")