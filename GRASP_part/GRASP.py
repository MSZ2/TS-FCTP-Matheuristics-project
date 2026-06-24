import numpy as np
import random
import time

# ============================================================
# Solution representation
# ============================================================

class Solution:
    def __init__(self, data):
        self.data = data
        self.x = np.zeros((data.I, data.J), dtype=int)
        self.y = np.zeros((data.J, data.K), dtype=int)
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
        self.cost = int(np.sum(self.x * d.b) +
                        np.sum(self.y * d.c) +
                        np.sum((self.x > 0) * d.f) +
                        np.sum((self.y > 0) * d.g))
        self.open_dc = {j for j in range(d.J)
                        if np.sum(self.x[:, j]) > 0 or np.sum(self.y[j, :]) > 0}
        return self.cost

    def is_valid(self):
        d = self.data
        return (all(np.sum(self.y[:, k]) == d.d[k] for k in range(d.K)) and
                all(np.sum(self.x[i, :]) <= d.s[i] for i in range(d.I)) and
                all(np.sum(self.x[:, j]) == np.sum(self.y[j, :]) for j in range(d.J)))


# ============================================================
# Greedy repair of first-stage flows given second-stage flows
# ============================================================

def repair(sol, data):
    """Assign suppliers (x) to meet the DC demands given by y."""
    sol.x.fill(0)
    rem_s = data.s.copy()
    for j in range(data.J):
        demand = int(sol.y[j].sum())
        if demand == 0:
            continue
        rem = demand
        # score combines variable cost + amortised fixed cost
        score = data.b[:, j] + data.f[:, j] / demand
        # sort suppliers by score, breaking ties with remaining capacity
        order = np.lexsort((-rem_s, score))
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
            return False
    return True


# ============================================================
# GRASP construction
# ============================================================

class Construction:
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
                        base = d.b[i, j] + d.c[j, k]
                        if sol.x[i, j] == 0:
                            base += d.f[i, j] / max(need, 1)
                        if sol.y[j, k] == 0:
                            base += d.g[j, k] / max(need, 1)
                        if j in sol.open_dc:        # favour already opened DCs
                            base *= 0.95
                        cand.append((base, i, j, q))
                if not cand:
                    return sol
                cand.sort(key=lambda x: x[0])
                cmin, cmax = cand[0][0], cand[-1][0]
                limit = cmin + self.alpha * (cmax - cmin + 1e-9)
                rcl = [x for x in cand if x[0] <= limit]
                _, i, j, q = random.choice(rcl)
                sol.x[i, j] += q
                sol.y[j, k] += q
                rem_s[i] -= q
                need -= q
        sol.compute_cost()
        return sol


# ============================================================
# Local search with DC‑structure moves (implicit limits, first‑improvement)
# ============================================================

class LocalSearch:
    def __init__(self, data):
        self.data = data

    def improve(self, sol, max_no_improve=6):
        """
        VND with first‑improvement and bounded candidate evaluations.
        All neighbourhoods are checked in order; on success we reset.
        """
        sol.compute_cost()
        no_improve = 0
        while no_improve < max_no_improve:
            improved = False

            if self._reassign_customer(sol):
                improved = True
                no_improve = 0
                continue

            if self._add_dc(sol):
                improved = True
                no_improve = 0
                continue

            if self._remove_dc(sol):
                improved = True
                no_improve = 0
                continue

            if self._swap_dc(sol):
                improved = True
                no_improve = 0
                continue

            no_improve += 1
        return sol

    # -----------------------------------------------------------------
    def _reassign_customer(self, sol):
        """
        Move a whole customer to a cheaper DC.
        Uses first‑improvement and limits the number of target DCs
        tried per customer to 10 (random order).
        """
        d = self.data
        # For each customer, try at most 10 random target DCs.
        max_targets = min(d.J, 10)

        for k in range(d.K):
            # Find the DC currently serving this customer (assume single source)
            j_from = next((j for j in range(d.J) if sol.y[j, k] > 0), None)
            if j_from is None:
                continue
            qty = sol.y[j_from, k]
            if qty == 0:
                continue

            # Random order of target DCs
            targets = list(range(d.J))
            random.shuffle(targets)
            attempts = 0

            for j_to in targets:
                if j_to == j_from:
                    continue
                if attempts >= max_targets:
                    break
                attempts += 1

                old_full = d.c[j_from, k] * d.d[k] + d.g[j_from, k]
                new_full = d.c[j_to,   k] * d.d[k] + d.g[j_to,   k]
                if new_full < old_full:
                    trial = sol.copy()
                    trial.y[j_to, k] += qty
                    trial.y[j_from, k] = 0
                    if repair(trial, d):
                        trial.compute_cost()
                        if trial.cost < sol.cost:
                            # Accept first improvement
                            sol.x = trial.x.copy()
                            sol.y = trial.y.copy()
                            sol.cost = trial.cost
                            sol.open_dc = trial.open_dc.copy()
                            return True
            # No improvement for this customer -> try next
        return False

    # -----------------------------------------------------------------
    def _add_dc(self, sol):
        """
        Try opening a closed DC (first‑improvement, at most 15 candidates).
        """
        d = self.data
        closed = [j for j in range(d.J) if j not in sol.open_dc]
        if not closed:
            return False

        random.shuffle(closed)
        max_candidates = min(len(closed), 15)

        for j_new in closed[:max_candidates]:
            trial = sol.copy()
            moved = False
            for k in range(d.K):
                curr_j = next((j for j in range(d.J) if trial.y[j, k] > 0), None)
                if curr_j is None:
                    continue
                old_full = d.c[curr_j, k] * d.d[k] + d.g[curr_j, k]
                new_full = d.c[j_new,  k] * d.d[k] + d.g[j_new,  k]
                if new_full < old_full:
                    trial.y[j_new,  k] += trial.y[curr_j, k]
                    trial.y[curr_j, k] = 0
                    moved = True
            if moved and repair(trial, d):
                trial.compute_cost()
                if trial.cost < sol.cost:
                    sol.x = trial.x.copy()
                    sol.y = trial.y.copy()
                    sol.cost = trial.cost
                    sol.open_dc = trial.open_dc.copy()
                    return True
        return False

    # -----------------------------------------------------------------
    def _remove_dc(self, sol):
        """
        Try closing an open DC (first‑improvement, at most 5 candidates).
        """
        d = self.data
        open_dcs = list(sol.open_dc)
        if len(open_dcs) <= 1:
            return False

        random.shuffle(open_dcs)
        max_candidates = min(len(open_dcs), 5)

        for j in open_dcs[:max_candidates]:
            trial = sol.copy()
            alts = [jj for jj in trial.open_dc if jj != j]
            if not alts:
                continue
            for k in range(d.K):
                if trial.y[j, k] > 0:
                    best_j = min(alts,
                                 key=lambda jj: d.c[jj, k] * trial.y[j, k] +
                                               (d.g[jj, k] if trial.y[jj, k] == 0 else 0))
                    trial.y[best_j, k] += trial.y[j, k]
                    trial.y[j, k] = 0
            if repair(trial, d):
                trial.compute_cost()
                if trial.cost < sol.cost:
                    sol.x = trial.x.copy()
                    sol.y = trial.y.copy()
                    sol.cost = trial.cost
                    sol.open_dc = trial.open_dc.copy()
                    return True
        return False

    # -----------------------------------------------------------------
    def _swap_dc(self, sol):
        """
        Swap one open DC with one closed DC (first‑improvement, at most 30 random pairs).
        """
        d = self.data
        open_list = list(sol.open_dc)
        closed = [j for j in range(d.J) if j not in sol.open_dc]
        if not open_list or not closed:
            return False

        # Generate at most 30 random (open, closed) pairs
        pairs = []
        n_pairs = min(30, len(open_list) * len(closed))
        for _ in range(n_pairs):
            o = random.choice(open_list)
            c = random.choice(closed)
            pairs.append((o, c))

        for j_close, j_open in pairs:
            trial = sol.copy()
            for k in range(d.K):
                if trial.y[j_close, k] > 0:
                    trial.y[j_open,  k] += trial.y[j_close, k]
                    trial.y[j_close, k] = 0
            if repair(trial, d):
                trial.compute_cost()
                if trial.cost < sol.cost:
                    sol.x = trial.x.copy()
                    sol.y = trial.y.copy()
                    sol.cost = trial.cost
                    sol.open_dc = trial.open_dc.copy()
                    return True
        return False


# ============================================================
# Elite pool with diversity
# ============================================================

class ElitePool:
    def __init__(self, size=12):
        self.size = size
        self.pool = []

    def add(self, sol):
        sig = frozenset(sol.open_dc)
        for p in self.pool:
            if frozenset(p.open_dc) == sig and abs(p.cost - sol.cost) < 50:
                return
        self.pool.append(sol.copy())
        self.pool.sort(key=lambda s: s.cost)
        self.pool = self.pool[:self.size]

    def best(self):
        return self.pool[0].copy() if self.pool else None

    def random(self):
        return random.choice(self.pool).copy() if self.pool else None


# ============================================================
# Path relinking
# ============================================================

class PathRelinking:
    def __init__(self, data):
        self.data = data
        self.ls = LocalSearch(data)

    def relink(self, a, b):
        d = self.data
        best = None
        best_cost = float('inf')
        # evaluate union and intersection of DC sets
        for dc_set in (a.open_dc | b.open_dc, a.open_dc & b.open_dc):
            if not dc_set:
                continue
            trial = Solution(d)
            for k in range(d.K):
                best_j = min(dc_set, key=lambda j: d.c[j, k] * d.d[k] + d.g[j, k])
                trial.y[best_j, k] = d.d[k]
            if repair(trial, d):
                trial.compute_cost()
                trial = self.ls.improve(trial)
                if trial.cost < best_cost:
                    best_cost = trial.cost
                    best = trial.copy()
        return best if best else a.copy()


# ============================================================
# GRASP main algorithm
# ============================================================

class GRASP:
    def __init__(self, data, iters=300):
        self.data = data
        self.iters = iters

    def run(self):
        ls = LocalSearch(self.data)
        pr = PathRelinking(self.data)
        elite = ElitePool(12)
        best = None
        best_cost = float('inf')
        last_improvement = 0
        start = time.time()

        for it in range(self.iters):
            # reactive alpha: increase randomness when stuck
            if it - last_improvement > 15:
                alpha = random.uniform(0.4, 0.7)
            else:
                alpha = random.uniform(0.05, 0.5)

            sol = Construction(self.data, alpha).build()
            sol = ls.improve(sol)
            sol.compute_cost()
            elite.add(sol)

            if sol.cost < best_cost:
                best = sol.copy()
                best_cost = sol.cost
                last_improvement = it
                print(f'[{it}] best = {best_cost}')

            # path relinking every 8 iterations
            if it % 8 == 0 and len(elite.pool) >= 2:
                cand = pr.relink(elite.best(), elite.random())
                if cand and cand.cost < best_cost:
                    best = cand.copy()
                    best_cost = cand.cost
                    last_improvement = it
                    print(f'[{it}] PR = {best_cost}')

            # shake (ruin-and-recreate) when stuck for too long
            if it - last_improvement > 20 and best:
                d = self.data
                if len(best.open_dc) > 1:
                    # score DCs by average variable cost per unit
                    dc_scores = []
                    for j in best.open_dc:
                        demand = best.y[j, :].sum()
                        avg_var = 0 if demand == 0 else (best.x[:, j] * d.b[:, j]).sum() / demand
                        dc_scores.append((avg_var, j))
                    dc_scores.sort(reverse=True)
                    n_remove = max(1, int(len(dc_scores) * 0.5))
                    remove_set = {j for _, j in dc_scores[:n_remove]}

                    trial = Solution(d)
                    # keep flows of non‑removed DCs
                    for k in range(d.K):
                        for j in best.open_dc:
                            if j not in remove_set and best.y[j, k] > 0:
                                trial.y[j, k] = best.y[j, k]

                    # ensure every customer's demand is fully satisfied
                    allowed_dcs = [j for j in range(d.J) if j not in remove_set]
                    if not allowed_dcs:
                        allowed_dcs = list(best.open_dc)
                    for k in range(d.K):
                        shortfall = d.d[k] - trial.y[:, k].sum()
                        if shortfall > 0:
                            best_j = min(allowed_dcs,
                                         key=lambda j: d.c[j, k] * shortfall +
                                                       (d.g[j, k] if trial.y[j, k] == 0 else 0))
                            trial.y[best_j, k] += shortfall

                    # repair and verify validity before accepting
                    if repair(trial, d) and trial.is_valid():
                        trial.compute_cost()
                        trial = ls.improve(trial)
                        if trial.is_valid() and trial.cost < best_cost:
                            best = trial.copy()
                            best_cost = trial.cost
                            last_improvement = it
                            print(f'[{it}] shake = {best_cost}')
                last_improvement = it   # avoid consecutive shakes

        return best, best_cost, time.time() - start
