import numpy as np
import random
import time
import os
from dataclasses import dataclass
from typing import List, Tuple, Set
import sys


# ============================================================
# DATA
# ============================================================

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
# SOLUTION
# ============================================================

class TSFCTPSolution:
    def __init__(self, data: TSFCTPData):
        self.data = data
        self.x = np.zeros((data.I, data.J), dtype=int)
        self.y = np.zeros((data.J, data.K), dtype=int)
        self.total_cost = 0

    def calculate_cost(self):
        var = (self.x * self.data.b).sum() + (self.y * self.data.c).sum()
        fix = 0
        for i in range(self.data.I):
            for j in range(self.data.J):
                if self.x[i, j] > 0:
                    fix += self.data.f[i, j]
        for j in range(self.data.J):
            for k in range(self.data.K):
                if self.y[j, k] > 0:
                    fix += self.data.g[j, k]
        self.total_cost = var + fix
        return self.total_cost

    def is_valid(self) -> bool:
        for i in range(self.data.I):
            if self.x[i, :].sum() > self.data.s[i]:
                return False
        for k in range(self.data.K):
            if self.y[:, k].sum() != self.data.d[k]:
                return False
        for j in range(self.data.J):
            if self.x[:, j].sum() != self.y[j, :].sum():
                return False
        return not (self.x < 0).any() and not (self.y < 0).any()

    def copy(self):
        s = TSFCTPSolution(self.data)
        s.x = self.x.copy()
        s.y = self.y.copy()
        s.total_cost = self.total_cost
        return s

    @property
    def open_dc(self) -> Set[int]:
        return {j for j in range(self.data.J)
                if self.x[:, j].sum() > 0 or self.y[j, :].sum() > 0}


# ============================================================
# FAST GREEDY REPAIR (sets x from y)
# ============================================================

def repair_first_stage(sol: TSFCTPSolution):
    d = sol.data
    sol.x.fill(0)
    remaining = d.s.copy()
    for j in range(d.J):
        demand = sol.y[j, :].sum()
        if demand == 0:
            continue
        order = sorted(range(d.I),
                       key=lambda i: d.b[i, j] + d.f[i, j] / demand)
        for i in order:
            if remaining[i] <= 0:
                continue
            take = min(remaining[i], demand)
            sol.x[i, j] = take
            remaining[i] -= take
            demand -= take
            if demand == 0:
                break
        if demand > 0:
            return False
    return True


# ============================================================
# CONSTRUCTION (GRASP, reactive alpha)
# ============================================================

class ImprovedGRASPConstruction:
    def __init__(self, data: TSFCTPData, alpha: float = 0.2):
        self.data = data
        self.alpha = alpha

    def construct(self) -> TSFCTPSolution:
        sol = TSFCTPSolution(self.data)
        rem_s = self.data.s.copy()
        active_first = set()
        active_second = set()

        order = list(range(self.data.K))
        order.sort(key=lambda k: self.data.d[k], reverse=True)

        for k in order:
            need = self.data.d[k]
            while need > 0:
                cand = self._evaluate_candidates(k, rem_s, active_first, active_second)
                if not cand:
                    cand = self._evaluate_candidates(k, rem_s, set(), set())
                min_score = cand[0][0]
                max_score = cand[-1][0]
                threshold = min_score + self.alpha * (max_score - min_score)
                rcl = [c for s, c in cand if s <= threshold]
                i, j = random.choice(rcl)

                max_q = min(rem_s[i], need)
                if (i, j) in active_first:
                    qty = max_q
                else:
                    min_q = max(1, max_q // 5)
                    qty = random.randint(min_q, max_q)

                sol.x[i, j] += qty
                sol.y[j, k] += qty
                active_first.add((i, j))
                active_second.add((j, k))
                rem_s[i] -= qty
                need -= qty

        sol.calculate_cost()
        return sol

    def _evaluate_candidates(self, k, rem_s, active_first, active_second):
        d = self.data
        cand = []
        for i in range(d.I):
            if rem_s[i] <= 0:
                continue
            for j in range(d.J):
                var = d.b[i, j] + d.c[j, k]
                exp_flow = min(rem_s[i], d.d[k])
                fix_pen = 0.0
                if exp_flow > 0:
                    if (i, j) not in active_first:
                        fix_pen += d.f[i, j] / exp_flow
                    if (j, k) not in active_second:
                        fix_pen += d.g[j, k] / exp_flow
                score = var + fix_pen + random.uniform(0, 0.1)
                cand.append((score, (i, j)))
        cand.sort(key=lambda x: x[0])
        return cand


# ============================================================
# LOCAL SEARCH (three correct moves)
# ============================================================

class AggressiveLocalSearch:
    def __init__(self, data: TSFCTPData):
        self.data = data

    def improve(self, sol: TSFCTPSolution) -> TSFCTPSolution:
        current = sol.copy()
        improved = True
        max_iter = 100
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            if self._consolidate_flows(current):
                improved = True; continue
            if self._switch_dc(current):
                improved = True; continue
            if self._remove_unprofitable_links(current):
                improved = True; continue
        current.calculate_cost()
        return current

    def _consolidate_flows(self, sol):
        d = self.data
        for j in range(d.J):
            srcs = [(i, sol.x[i, j]) for i in range(d.I) if sol.x[i, j] > 0]
            if len(srcs) <= 1:
                continue
            srcs.sort(key=lambda x: d.b[x[0], j])
            best_i = srcs[0][0]
            others = [(i, flow) for i, flow in srcs[1:]]
            total_extra = sum(flow for _, flow in others)
            if sol.x[best_i, :].sum() + total_extra > d.s[best_i]:
                continue
            backup = {i: sol.x[i, j] for i, _ in others}
            for i, flow in others:
                sol.x[best_i, j] += flow
                sol.x[i, j] = 0
            if sol.is_valid():
                return True
            for i, flow in others:
                sol.x[best_i, j] -= flow
                sol.x[i, j] = backup[i]
        return False

    def _switch_dc(self, sol):
        d = self.data
        for j1 in range(d.J):
            for j2 in range(d.J):
                if j1 == j2: continue
                for k in range(d.K):
                    flow = sol.y[j1, k]
                    if flow == 0: continue
                    var_y = flow * (d.c[j2, k] - d.c[j1, k])
                    fix_y = 0
                    if sol.y[j1, k] == flow:          # arc (j1,k) will become 0
                        fix_y -= d.g[j1, k]
                    if sol.y[j2, k] == 0:              # arc (j2,k) is currently 0
                        fix_y += d.g[j2, k]
                    for i in range(d.I):
                        if sol.x[i, j1] >= flow:
                            var_x = flow * (d.b[i, j2] - d.b[i, j1])
                            fix_x = 0
                            if sol.x[i, j1] == flow:   # arc (i,j1) becomes 0
                                fix_x -= d.f[i, j1]
                            if sol.x[i, j2] == 0:       # arc (i,j2) currently 0
                                fix_x += d.f[i, j2]
                            total_delta = var_y + fix_y + var_x + fix_x
                            if total_delta < 0:
                                backup = (sol.x[i, j1], sol.x[i, j2],
                                          sol.y[j1, k], sol.y[j2, k])
                                sol.y[j1, k] -= flow
                                sol.y[j2, k] += flow
                                sol.x[i, j1] -= flow
                                sol.x[i, j2] += flow
                                if sol.is_valid():
                                    return True
                                sol.y[j1, k], sol.y[j2, k] = backup[2], backup[3]
                                sol.x[i, j1], sol.x[i, j2] = backup[0], backup[1]
                            break   # try next i
        return False

    def _remove_unprofitable_links(self, sol):
        d = self.data
        for i in range(d.I):
            for j in range(d.J):
                flow = sol.x[i, j]
                if flow == 0: continue
                for j2 in range(d.J):
                    if j2 == j or sol.x[i, j2] == 0: continue
                    var_x = flow * (d.b[i, j2] - d.b[i, j])
                    fix_x = 0
                    if flow == sol.x[i, j]:            # arc (i,j) will disappear
                        fix_x -= d.f[i, j]
                    # move corresponding y flow
                    remaining = flow
                    moved = []
                    temp_y = sol.y.copy()
                    for k in range(d.K):
                        if remaining <= 0: break
                        if temp_y[j, k] > 0:
                            take = min(remaining, temp_y[j, k])
                            temp_y[j, k] -= take
                            temp_y[j2, k] += take
                            moved.append((k, take))
                            remaining -= take
                    if remaining > 0: continue
                    var_y = 0
                    fix_y = 0
                    for k, amt in moved:
                        var_y += amt * (d.c[j2, k] - d.c[j, k])
                        if temp_y[j, k] == 0 and sol.y[j, k] > 0:
                            fix_y -= d.g[j, k]
                        if sol.y[j2, k] == 0 and amt > 0:
                            fix_y += d.g[j2, k]
                    total_delta = var_x + fix_x + var_y + fix_y
                    if total_delta < 0:
                        backup_x = (sol.x[i, j], sol.x[i, j2])
                        backup_y = {}
                        for k, amt in moved:
                            backup_y[(j, k)] = sol.y[j, k]
                            backup_y[(j2, k)] = sol.y[j2, k]
                        sol.x[i, j] -= flow
                        sol.x[i, j2] += flow
                        for k, amt in moved:
                            sol.y[j, k] -= amt
                            sol.y[j2, k] += amt
                        if sol.is_valid():
                            return True
                        sol.x[i, j], sol.x[i, j2] = backup_x
                        for k, amt in moved:
                            sol.y[j, k] = backup_y[(j, k)]
                            sol.y[j2, k] = backup_y[(j2, k)]
        return False


# ============================================================
# PATH RELINKING (flow blend)
# ============================================================

class PathRelinking:
    def __init__(self, data: TSFCTPData):
        self.data = data
        self.ls = AggressiveLocalSearch(data)

    def relink(self, a: TSFCTPSolution, b: TSFCTPSolution) -> TSFCTPSolution:
        d = self.data
        # Step 1: DC structure alignment
        current = a.copy()
        for j in b.open_dc - current.open_dc:
            current.open_dc.add(j)   # will be filled later
        for j in current.open_dc - b.open_dc:
            self._close_dc(current, j)

        # Step 2: partial flow blend (half of demand from a, half from b)
        for k in range(d.K):
            if random.random() < 0.5:
                j_from = np.argmax(current.y[:, k])
                j_to = np.argmax(b.y[:, k])
                if j_from != j_to:
                    delta = current.y[j_from, k] // 2
                    current.y[j_from, k] -= delta
                    current.y[j_to, k] += delta

        # Step 3: full reassignment to cheapest open DC (cleanup)
        for k in range(d.K):
            best_j = min(current.open_dc,
                         key=lambda j: d.c[j, k] * d.d[k] + d.g[j, k])
            if best_j != np.argmax(current.y[:, k]):
                qty = current.y[:, k].sum()
                current.y[:, k] = 0
                current.y[best_j, k] = qty

        if not repair_first_stage(current):
            return a.copy()
        current.calculate_cost()
        current = self.ls.improve(current)
        return current

    def _close_dc(self, sol, j):
        d = self.data
        for k in range(d.K):
            if sol.y[j, k] > 0:
                alts = [jj for jj in range(d.J) if jj != j and jj in sol.open_dc]
                if not alts:
                    alts = [jj for jj in range(d.J) if jj != j]
                best_j = min(alts, key=lambda jj: d.c[jj, k])
                sol.y[best_j, k] += sol.y[j, k]
                sol.y[j, k] = 0
        sol.open_dc.discard(j)


# ============================================================
# ELITE POOL
# ============================================================

class ElitePool:
    def __init__(self, size=10):
        self.size = size
        self.pool = []

    def add(self, sol: TSFCTPSolution):
        self.pool.append(sol.copy())
        self.pool.sort(key=lambda s: s.total_cost)
        self.pool = self.pool[:self.size]

    def best(self):
        return self.pool[0].copy() if self.pool else None

    def random(self):
        return random.choice(self.pool).copy() if self.pool else None


# ============================================================
# GRASP WITH DESTRUCTIVE PERTURBATION
# ============================================================

class GRASPSolver:
    def __init__(self, data: TSFCTPData, instance_path: str):
        self.data = data
        self.instance_path = instance_path
        self.ls = AggressiveLocalSearch(data)
        self.pr = PathRelinking(data)
        self.optimal_value, self.optimal_time = self._load_optimal()

    def _load_optimal(self):
        fname = os.path.basename(self.instance_path)
        if fname.startswith('small_'):
            res_file = "results_small.txt"
        elif fname.startswith('medium_'):
            res_file = "results_medium.txt"
        elif fname.startswith('large_'):
            res_file = "results_large.txt"
        else:
            return None, None
        try:
            with open(res_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts[0] == self.instance_path:
                        return float(parts[2]), float(parts[3])
        except:
            pass
        return None, None

    def _ruin_and_recreate(self, sol: TSFCTPSolution, strength=0.4):
        """Destroy a fraction of open DCs and rebuild."""
        d = self.data
        if len(sol.open_dc) <= 1:
            return sol.copy()
        # choose DCs to remove (prefer expensive ones)
        dc_list = list(sol.open_dc)
        # score: high variable cost per unit -> remove first
        dc_scores = []
        for j in dc_list:
            total_demand = sol.y[j, :].sum()
            if total_demand == 0:
                dc_scores.append((0, j))
                continue
            avg_var = (sol.x[:, j] * d.b[:, j]).sum() / total_demand
            dc_scores.append((avg_var, j))
        dc_scores.sort(reverse=True)
        n_remove = max(1, int(len(dc_list) * strength))
        remove_set = set([j for _, j in dc_scores[:n_remove]])

        # create new solution, keep only non-removed DC assignments
        new_sol = TSFCTPSolution(d)
        for k in range(d.K):
            # keep flows to DCs not removed
            for j in sol.open_dc:
                if j not in remove_set and sol.y[j, k] > 0:
                    new_sol.y[j, k] = sol.y[j, k]
            # if no flow remained, assign randomly to any DC (not removed)
            if new_sol.y[:, k].sum() == 0:
                candidates = [j for j in range(d.J) if j not in remove_set]
                if not candidates:
                    candidates = list(sol.open_dc - remove_set)
                if candidates:
                    j_rand = random.choice(candidates)
                    new_sol.y[j_rand, k] = d.d[k]
        if not repair_first_stage(new_sol):
            return sol.copy()
        new_sol.calculate_cost()
        return new_sol

    def single_run(self, seed: int, max_iters: int = 1000) -> dict:
        random.seed(seed)
        np.random.seed(seed)
        start = time.time()
        best = None
        best_cost = float('inf')
        time_to_best = 0
        stag_count = 0
        elite = ElitePool(10)

        for it in range(max_iters):
            # Reactive alpha
            if stag_count >= 20:
                alpha = random.uniform(0.4, 0.7)
            else:
                alpha = random.uniform(0.1, 0.2)

            sol = ImprovedGRASPConstruction(self.data, alpha).construct()
            sol = self.ls.improve(sol)
            sol.calculate_cost()

            if sol.total_cost < best_cost:
                best = sol.copy()
                best_cost = sol.total_cost
                time_to_best = time.time() - start
                stag_count = 0
            else:
                stag_count += 1

            elite.add(sol)

            # Path relinking every 10 iterations
            if it % 10 == 0 and len(elite.pool) >= 2:
                a = elite.best()
                b = elite.random()
                pr_sol = self.pr.relink(a, b)
                if pr_sol and pr_sol.is_valid():
                    pr_sol.calculate_cost()
                    if pr_sol.total_cost < best_cost:
                        best = pr_sol.copy()
                        best_cost = pr_sol.total_cost
                        time_to_best = time.time() - start
                        stag_count = 0
                    elite.add(pr_sol)

            # Destructive perturbation when stuck
            if stag_count >= 25:
                # perturb from best or random elite
                if elite.pool:
                    base = elite.random() if random.random() < 0.5 else elite.best()
                else:
                    base = best
                if base:
                    destroyed = self._ruin_and_recreate(base, strength=0.5)
                    destroyed = self.ls.improve(destroyed)
                    destroyed.calculate_cost()
                    if destroyed.total_cost < best_cost:
                        best = destroyed.copy()
                        best_cost = destroyed.total_cost
                        time_to_best = time.time() - start
                        stag_count = 0
                    else:
                        stag_count = 15   # give some time after perturbation
                    elite.add(destroyed)

        total_time = time.time() - start
        return {
            'best_cost': best_cost,
            'time_to_best': time_to_best,
            'total_time': total_time,
            'iterations': max_iters,
            'best_solution': best
        }

    def run_multiple(self, num_runs=10):
        results = []
        for r in range(num_runs):
            seed = 42 + r * 100
            print(f"Run {r+1}/{num_runs} ...")
            res = self.single_run(seed)
            results.append(res)
            print(f"  Best = {res['best_cost']}, time to best = {res['time_to_best']:.2f}s")
        return self._compute_stats(results)

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

    def print_report(self, stats):
        print("\n" + "="*60)
        print("GRASP REPORT")
        print("="*60)
        print(f"Instance: {self.instance_path}")
        print(f"Optimal (if known): {self.optimal_value}")
        print(f"Best MA cost: {stats['best_ma']}")
        print(f"\n--- Averages over {stats['num_runs']} runs ---")
        print(f"Time to best : {stats['mean_t_best']:.2f} s")
        print(f"Total time   : {stats['mean_t_total']:.2f} s")
        print(f"Gap (%)      : {stats['mean_gap']:.2f} ± {stats['std_gap']:.2f}")

        rep_name = f"GRASP_report_{os.path.basename(self.instance_path)}"
        with open(rep_name, 'w') as f:
            f.write(f"Instance: {self.instance_path}\n")
            f.write(f"Best MA: {stats['best_ma']}\n")
            for i, r in enumerate(stats['results']):
                f.write(f"Run {i+1}: {r['best_cost']} gap={stats['gaps'][i]:.2f}%\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python grasp_final.py <instance_file> [num_runs]")
        sys.exit(1)
    fname = sys.argv[1]
    nruns = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    data = read_instance(fname)
    solver = GRASPSolver(data, fname)
    stats = solver.run_multiple(nruns)
    solver.print_report(stats)