import numpy as np
import random
import time
from dataclasses import dataclass


# ============================================================
# DATA
# ============================================================

@dataclass
class TSFCTPData:
    I: int
    J: int
    K: int
    s: np.ndarray   # supply[i]
    d: np.ndarray   # demand[k]
    b: np.ndarray   # variable cost supplier i -> DC j  (I x J)
    f: np.ndarray   # fixed cost supplier i -> DC j     (I x J)
    c: np.ndarray   # variable cost DC j -> customer k  (J x K)
    g: np.ndarray   # fixed cost DC j -> customer k     (J x K)

    def validate(self):
        assert np.sum(self.s) >= np.sum(self.d), "Total supply must cover total demand"


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
    print(f"I={I}, J={J}, K={K}, supply={np.sum(s)}, demand={np.sum(d)}")
    data = TSFCTPData(I, J, K, s, d, b, f, c, g)
    data.validate()
    return data


# ============================================================
# SOLUTION
# ============================================================

class Solution:
    def __init__(self, data):
        self.data = data
        self.x = np.zeros((data.I, data.J), dtype=int)  # flow supplier->DC
        self.y = np.zeros((data.J, data.K), dtype=int)  # flow DC->customer
        self.cost = 0
        self.open_dc = set()

    def copy(self):
        s = Solution(self.data)
        s.x = self.x.copy()
        s.y = self.y.copy()
        s.cost = self.cost
        s.open_dc = set(self.open_dc)
        return s

    def compute_cost(self):
        d = self.data
        self.cost = int(
            np.sum(self.x * d.b) +
            np.sum(self.y * d.c) +
            np.sum((self.x > 0) * d.f) +
            np.sum((self.y > 0) * d.g)
        )
        self.open_dc = {
            j for j in range(d.J)
            if np.sum(self.x[:, j]) > 0 or np.sum(self.y[j, :]) > 0
        }
        return self.cost

    def is_valid(self):
        d = self.data
        for k in range(d.K):
            if np.sum(self.y[:, k]) != d.d[k]:
                return False
        for i in range(d.I):
            if np.sum(self.x[i, :]) > d.s[i]:
                return False
        for j in range(d.J):
            if np.sum(self.x[:, j]) != np.sum(self.y[j, :]):
                return False
        return True


# ============================================================
# REPAIR  (FIX #1: amortized fixed cost in supplier selection)
# ============================================================

def repair(sol, data):
    """
    Recompute x given a fixed y, assigning each DC's demand to suppliers.

    KEY FIX over original: sorts suppliers by  b[i,j] + f[i,j]/demand_j
    (amortized fixed cost) rather than b[i,j] alone. When demand_j is
    large the fixed cost is negligible; when small it matters a lot.
    """
    sol.x[:] = 0
    rem_s = data.s.copy()
    for j in range(data.J):
        demand = int(np.sum(sol.y[j, :]))
        if demand == 0:
            continue
        rem = demand
        order = sorted(
            range(data.I),
            key=lambda i: data.b[i, j] + data.f[i, j] / demand
        )
        for i in order:
            cap = rem_s[i]
            if cap <= 0:
                continue
            take = min(cap, rem)
            sol.x[i, j] += take
            rem_s[i] -= take
            rem -= take
            if rem == 0:
                break
        if rem > 0:
            return False   # infeasible
    return True


# ============================================================
# CONSTRUCTION
# ============================================================

class Construction:
    """
    Greedy randomized construction.

    FIX over original:
    - Removed the 0.97 open-DC bias (arbitrary magic constant that distorts
      the cost estimate and makes the RCL unfair).
    - Uses amortized fixed costs (f/demand, g/demand) consistently, which
      better reflects the true per-unit cost of opening a new arc.
    """

    def __init__(self, data, alpha=0.3):
        self.data = data
        self.alpha = alpha

    def build(self):
        d = self.data
        sol = Solution(d)
        rem_s = d.s.copy()

        for k in np.argsort(-d.d):          # largest demand first
            need = int(d.d[k])
            while need > 0:
                cand = []
                for i in range(d.I):
                    if rem_s[i] <= 0:
                        continue
                    for j in range(d.J):
                        q = min(rem_s[i], need)
                        eff = max(need, 1)   # denominator for amortized fixed cost
                        unit = d.b[i, j] + d.c[j, k]
                        if sol.x[i, j] == 0:
                            unit += d.f[i, j] / eff
                        if sol.y[j, k] == 0:
                            unit += d.g[j, k] / eff
                        cand.append((unit, i, j, q))

                if not cand:
                    return sol

                cand.sort(key=lambda x: x[0])
                cmin, cmax = cand[0][0], cand[-1][0]
                limit = cmin + self.alpha * (cmax - cmin)
                rcl = [x for x in cand if x[0] <= limit]
                _, i, j, q = random.choice(rcl)

                sol.x[i, j] += q
                sol.y[j, k] += q
                rem_s[i] -= q
                need -= q

        sol.compute_cost()
        return sol


# ============================================================
# LOCAL SEARCH
# ============================================================

class LocalSearch:
    """
    Six neighbourhood moves, applied in best-improvement order until
    no move in any neighbourhood helps.

    Moves added / fixed vs original:
      _full_reassign    – uses TRUE cost delta (includes g fixed costs),
                          not just variable cost comparison.
      _swap_open_closed – exhaustive over all (j_close, j_open) pairs
                          instead of 10 random attempts.
      _remove_dc        – routes customers to cheapest alternative
                          accounting for g fixed cost.
      _add_dc           – opens a DC and moves customers that benefit
                          (c+g comparison, not c only).
      _partial_shift    – NEW: moves a fraction of customer demand to
                          another DC, enabling customer demand *splitting*
                          which is legal in TSFCTP and often optimal.
      _two_customer_swap – NEW: swaps the DC assignments of two customers;
                          catches cases neither individual move finds.
    """

    def __init__(self, data):
        self.data = data

    def improve(self, sol, max_no_improve=5):
        sol.compute_cost()
        no_improve = 0
        while no_improve < max_no_improve:
            improved = (
                self._full_reassign(sol)    or
                self._swap_open_closed(sol) or
                self._remove_dc(sol)        or
                self._add_dc(sol)           or
                self._partial_shift(sol)    or
                self._two_customer_swap(sol)
            )
            if improved:
                no_improve = 0
            else:
                no_improve += 1
        return sol

    # ----------------------------------------------------------
    # Move 1: reassign ALL of customer k from j_from to j_to
    # ----------------------------------------------------------
    def _full_reassign(self, sol):
        d = self.data
        best_delta = 0
        best_params = None

        for k in range(d.K):
            for j_from in range(d.J):
                qty = sol.y[j_from, k]
                if qty == 0:
                    continue
                for j_to in range(d.J):
                    if j_to == j_from:
                        continue
                    # Variable cost change
                    delta = (d.c[j_to, k] - d.c[j_from, k]) * qty
                    # Fixed cost: pay g[j_to,k] if j_to->k not yet open
                    delta += d.g[j_to, k] if sol.y[j_to, k] == 0 else 0
                    # Fixed cost: save g[j_from,k] if j_from->k arc is emptied
                    if sol.y[j_from, k] == qty:   # arc will close
                        delta -= d.g[j_from, k]
                    if delta < best_delta:
                        best_delta = delta
                        best_params = (k, j_from, j_to, qty)

        if best_params is None:
            return False
        k, j_from, j_to, qty = best_params
        trial = sol.copy()
        trial.y[j_to, k] += qty
        trial.y[j_from, k] = 0
        if not repair(trial, d):
            return False
        trial.compute_cost()
        if trial.cost < sol.cost:
            sol.x = trial.x.copy(); sol.y = trial.y.copy()
            sol.cost = trial.cost;  sol.open_dc = trial.open_dc.copy()
            return True
        return False

    # ----------------------------------------------------------
    # Move 2: swap one open DC for one closed DC (exhaustive)
    # ----------------------------------------------------------
    def _swap_open_closed(self, sol):
        d = self.data
        closed = [j for j in range(d.J) if j not in sol.open_dc]
        best_cost = sol.cost
        best_trial = None

        for j_close in list(sol.open_dc):
            for j_open in closed:
                trial = sol.copy()
                for k in range(d.K):
                    if trial.y[j_close, k] > 0:
                        trial.y[j_open,  k] += trial.y[j_close, k]
                        trial.y[j_close, k] = 0
                if repair(trial, d):
                    trial.compute_cost()
                    if trial.cost < best_cost:
                        best_cost = trial.cost
                        best_trial = trial

        if best_trial:
            sol.x = best_trial.x.copy(); sol.y = best_trial.y.copy()
            sol.cost = best_trial.cost;  sol.open_dc = best_trial.open_dc.copy()
            return True
        return False

    # ----------------------------------------------------------
    # Move 3: close one DC, reroute customers to cheapest alternative
    # ----------------------------------------------------------
    def _remove_dc(self, sol):
        d = self.data
        if len(sol.open_dc) <= 1:
            return False
        best_cost = sol.cost
        best_trial = None

        for j in list(sol.open_dc):
            trial = sol.copy()
            alts = [jj for jj in trial.open_dc if jj != j]
            for k in range(d.K):
                if trial.y[j, k] > 0:
                    best_j = min(
                        alts,
                        key=lambda jj: (
                            d.c[jj, k] * trial.y[j, k] +
                            (d.g[jj, k] if trial.y[jj, k] == 0 else 0)
                        )
                    )
                    trial.y[best_j, k] += trial.y[j, k]
                    trial.y[j, k] = 0
            if repair(trial, d):
                trial.compute_cost()
                if trial.cost < best_cost:
                    best_cost = trial.cost
                    best_trial = trial

        if best_trial:
            sol.x = best_trial.x.copy(); sol.y = best_trial.y.copy()
            sol.cost = best_trial.cost;  sol.open_dc = best_trial.open_dc.copy()
            return True
        return False

    # ----------------------------------------------------------
    # Move 4: open a new DC and move customers that benefit
    # ----------------------------------------------------------
    def _add_dc(self, sol):
        d = self.data
        closed = [j for j in range(d.J) if j not in sol.open_dc]
        best_cost = sol.cost
        best_trial = None

        for j_new in closed:
            trial = sol.copy()
            moved = False
            for k in range(d.K):
                curr_j = next((j for j in range(d.J) if trial.y[j, k] > 0), None)
                if curr_j is None:
                    continue
                old_cost = d.c[curr_j, k] * d.d[k] + d.g[curr_j, k]
                new_cost = d.c[j_new,  k] * d.d[k] + d.g[j_new,  k]
                if new_cost < old_cost:
                    trial.y[j_new,  k] += trial.y[curr_j, k]
                    trial.y[curr_j, k] = 0
                    moved = True
            if moved and repair(trial, d):
                trial.compute_cost()
                if trial.cost < best_cost:
                    best_cost = trial.cost
                    best_trial = trial

        if best_trial:
            sol.x = best_trial.x.copy(); sol.y = best_trial.y.copy()
            sol.cost = best_trial.cost;  sol.open_dc = best_trial.open_dc.copy()
            return True
        return False

    # ----------------------------------------------------------
    # Move 5 (NEW): move a fraction of customer k to another DC
    #   This enables demand splitting, which is legal in TSFCTP
    #   and can be necessary for optimality.
    # ----------------------------------------------------------
    def _partial_shift(self, sol):
        d = self.data
        best_cost = sol.cost
        best_trial = None

        for k in range(d.K):
            for j_from in range(d.J):
                qty_total = sol.y[j_from, k]
                if qty_total < 2:
                    continue
                for j_to in range(d.J):
                    if j_to == j_from:
                        continue
                    # Only try if variable cost is cheaper at j_to
                    if d.c[j_to, k] >= d.c[j_from, k]:
                        continue
                    # Try three split points: 25%, 50%, 75%
                    for qty in {qty_total // 4, qty_total // 2, 3 * qty_total // 4}:
                        if qty <= 0:
                            continue
                        trial = sol.copy()
                        trial.y[j_to,   k] += qty
                        trial.y[j_from, k] -= qty
                        if repair(trial, d):
                            trial.compute_cost()
                            if trial.cost < best_cost:
                                best_cost = trial.cost
                                best_trial = trial

        if best_trial:
            sol.x = best_trial.x.copy(); sol.y = best_trial.y.copy()
            sol.cost = best_trial.cost;  sol.open_dc = best_trial.open_dc.copy()
            return True
        return False

    # ----------------------------------------------------------
    # Move 6 (NEW): swap the DC assignments of two customers
    # ----------------------------------------------------------
    def _two_customer_swap(self, sol):
        d = self.data
        best_cost = sol.cost
        best_trial = None

        # Build single-DC assignment list (customers served by exactly one DC)
        asgn = [
            (k, next((j for j in range(d.J) if sol.y[j, k] > 0), -1))
            for k in range(d.K)
        ]

        for i1 in range(d.K):
            k1, j1 = asgn[i1]
            if j1 < 0:
                continue
            for i2 in range(i1 + 1, d.K):
                k2, j2 = asgn[i2]
                if j2 < 0 or j1 == j2:
                    continue
                # Variable-cost delta (ignore fixed cost g here as a quick filter)
                delta = (
                    (d.c[j2, k1] - d.c[j1, k1]) * sol.y[j1, k1] +
                    (d.c[j1, k2] - d.c[j2, k2]) * sol.y[j2, k2]
                )
                if delta >= 0:
                    continue
                trial = sol.copy()
                trial.y[j2, k1] += trial.y[j1, k1]; trial.y[j1, k1] = 0
                trial.y[j1, k2] += trial.y[j2, k2]; trial.y[j2, k2] = 0
                if repair(trial, d):
                    trial.compute_cost()
                    if trial.cost < best_cost:
                        best_cost = trial.cost
                        best_trial = trial

        if best_trial:
            sol.x = best_trial.x.copy(); sol.y = best_trial.y.copy()
            sol.cost = best_trial.cost;  sol.open_dc = best_trial.open_dc.copy()
            return True
        return False


# ============================================================
# ELITE POOL  (with diversity filter)
# ============================================================

class ElitePool:
    """
    Keeps the best solutions found, with a diversity filter:
    solutions with the same open-DC signature and nearly identical
    cost are deduplicated to keep the pool varied.
    """

    def __init__(self, size=12):
        self.size = size
        self.pool = []

    def add(self, sol):
        sig = frozenset(sol.open_dc)
        for p in self.pool:
            if frozenset(p.open_dc) == sig and abs(p.cost - sol.cost) < 100:
                return   # duplicate – skip
        self.pool.append(sol.copy())
        self.pool.sort(key=lambda s: s.cost)
        self.pool = self.pool[:self.size]

    def best(self):
        return self.pool[0].copy() if self.pool else None

    def random(self):
        return random.choice(self.pool).copy() if self.pool else None


# ============================================================
# PATH RELINKING
# ============================================================

class PathRelinking:
    """
    Combines two elite solutions by exploring the union and
    intersection of their open-DC sets, then re-assigns customers
    optimally and applies local search.
    """

    def __init__(self, data):
        self.data = data
        self.ls = LocalSearch(data)

    def relink(self, a, b):
        d = self.data
        best = None
        best_cost = float('inf')

        for dc_set in [a.open_dc | b.open_dc, a.open_dc & b.open_dc]:
            if not dc_set:
                continue
            trial = Solution(d)
            # Greedy customer assignment within dc_set
            for k in range(d.K):
                best_j = min(dc_set, key=lambda j: d.c[j, k] * d.d[k] + d.g[j, k])
                trial.y[best_j, k] = d.d[k]
            if not repair(trial, d):
                continue
            trial.compute_cost()
            trial = self.ls.improve(trial)
            if trial.is_valid() and trial.cost < best_cost:
                best_cost = trial.cost
                best = trial

        return best


# ============================================================
# GRASP
# ============================================================

class GRASP:
    """
    Main GRASP loop.

    Changes vs original:
    - Uses improved Construction (no magic 0.97 bias).
    - Uses improved LocalSearch (6 moves, true cost deltas).
    - ElitePool has diversity filter.
    - PathRelinking fires every 8 iterations (same cadence).
    - alpha sampled uniformly in [0.05, 0.50] per iteration.
    """

    def __init__(self, data, iters=300):
        self.data = data
        self.iters = iters

    def run(self):
        best = None
        best_cost = float('inf')

        ls    = LocalSearch(self.data)
        pr    = PathRelinking(self.data)
        elite = ElitePool(12)
        start = time.time()

        for it in range(self.iters):
            alpha = random.uniform(0.05, 0.50)
            sol = Construction(self.data, alpha).build()
            sol = ls.improve(sol)
            sol.compute_cost()
            elite.add(sol)

            if sol.cost < best_cost:
                best = sol.copy()
                best_cost = sol.cost
                print(f"  [{it}] construct = {best_cost}", flush=True)

            # Path relinking every 8 iterations
            if it % 8 == 0 and len(elite.pool) >= 2:
                cand = pr.relink(elite.best(), elite.random())
                if cand and cand.is_valid():
                    elite.add(cand)
                    if cand.cost < best_cost:
                        best = cand.copy()
                        best_cost = cand.cost
                        print(f"  [{it}] PR        = {best_cost}", flush=True)

        elapsed = time.time() - start
        print(f"\nDONE: {best_cost}  time: {elapsed:.2f}s")
        return best, best_cost, elapsed


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python grasp_tsfctp_improved.py <instance_file> [iters]")
        sys.exit(1)

    filename = sys.argv[1]
    iters    = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    data = read_instance(filename)
    grasp = GRASP(data, iters=iters)

    random.seed(42)
    best, best_cost, elapsed = grasp.run()

    print(f"\nOpen DCs : {sorted(best.open_dc)}")
    print(f"Valid    : {best.is_valid()}")
    var_b = int(np.sum(best.x * data.b))
    var_c = int(np.sum(best.y * data.c))
    fix_f = int(np.sum((best.x > 0) * data.f))
    fix_g = int(np.sum((best.y > 0) * data.g))
    print(f"Cost breakdown:")
    print(f"  supplier->DC variable (b): {var_b}")
    print(f"  DC->customer variable (c): {var_c}")
    print(f"  supplier->DC fixed    (f): {fix_f}")
    print(f"  DC->customer fixed    (g): {fix_g}")
    print(f"  TOTAL                    : {var_b+var_c+fix_f+fix_g}")