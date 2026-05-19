import numpy as np
import os
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

    if base.startswith("small_"):
        return "results_small.txt"
    if base.startswith("medium_"):
        return "results_medium.txt"
    if base.startswith("large_"):
        return "results_large.txt"

    return None


def load_optimal_for_instance(instance_path: str):
    """
    Returns (optimal_value, optimal_time) if available.
    Safe fallback: (None, None)
    """
    file = get_results_filename(instance_path)

    if file is None or not os.path.exists(file):
        return None, None

    try:
        with open(file, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 4:
                    continue

                if parts[0] == instance_path:
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

        start = time.time()
        best, best_cost, _ = grasp.run()
        total_time = time.time() - start

        return {
            "run_id": run_id,
            "best_cost": best_cost,
            "time_to_best": total_time,   # fallback (no internal tracking)
            "total_time": total_time,
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
    # STATS (your format preserved)
    # --------------------------------------------------------
    def _compute_stats(self, results):
        costs = [r['best_cost'] for r in results]
        best_ma = min(costs)

        ref = self.optimal_value if self.optimal_value else best_ma

        gaps = [100 * abs(c - ref) / ref for c in costs]

        stats = {
            'num_runs': len(results),
            'best_ma': best_ma,
            'optimal': self.optimal_value,
            'opt_time': self.optimal_time,

            'mean_t_best': np.mean([r['time_to_best'] for r in results]),
            'mean_t_total': np.mean([r['total_time'] for r in results]),

            'mean_gap': np.mean(gaps),
            'std_gap': np.std(gaps),

            'results': results,
            'gaps': gaps
        }

        return stats

    # --------------------------------------------------------
    # PRINT REPORT (your format preserved)
    # --------------------------------------------------------
    def print_report(self, stats):
        print("\n" + "=" * 60)
        print("GRASP REPORT")
        print("=" * 60)

        print(f"Instance: {self.instance_path}")
        print(f"Optimal (if known): {self.optimal_value}")
        print(f"Best MA cost: {stats['best_ma']}")

        print(f"\n--- Averages over {stats['num_runs']} runs ---")
        print(f"Time to best : {stats['mean_t_best']:.2f} s")
        print(f"Total time   : {stats['mean_t_total']:.2f} s")
        print(f"Gap (%)      : {stats['mean_gap']:.2f} ± {stats['std_gap']:.2f}")

        os.makedirs("reports", exist_ok=True)

        rep_name = os.path.join(
            "reports",
            f"GRASP_report_{os.path.basename(self.instance_path)}"
        )

        with open(rep_name, "w") as f:
            f.write(f"Instance: {self.instance_path}\n")
            f.write(f"Best MA: {stats['best_ma']}\n")

            for i, r in enumerate(stats['results']):
                f.write(
                    f"Run {i+1}: {r['best_cost']} "
                    f"gap={stats['gaps'][i]:.2f}%\n"
                )