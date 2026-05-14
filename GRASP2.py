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
    s: np.ndarray
    d: np.ndarray
    b: np.ndarray
    f: np.ndarray
    c: np.ndarray
    g: np.ndarray

    def validate(self):
        assert np.sum(self.s) >= np.sum(self.d)


# ============================================================
# SOLUTION
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
        self.cost = int(
            np.sum(self.x * self.data.b) +
            np.sum(self.y * self.data.c) +
            np.sum((self.x > 0) * self.data.f) +
            np.sum((self.y > 0) * self.data.g)
        )
        self.open_dc = {j for j in range(self.data.J)
                        if np.sum(self.x[:, j]) > 0 or np.sum(self.y[j, :]) > 0}
        return self.cost

    def is_valid(self):
        for k in range(self.data.K):
            if np.sum(self.y[:, k]) != self.data.d[k]:
                return False

        for i in range(self.data.I):
            if np.sum(self.x[i, :]) > self.data.s[i]:
                return False

        for j in range(self.data.J):
            if np.sum(self.x[:, j]) != np.sum(self.y[j, :]):
                return False

        return True


# ============================================================
# CONSTRUCTION (IMPROVED RCL + DC BIAS)
# ============================================================

class Construction:
    def __init__(self, data, alpha=0.3):
        self.data = data
        self.alpha = alpha

    def build(self):
        sol = Solution(self.data)

        rem_s = self.data.s.copy()
        demand = self.data.d.copy()

        for k in np.argsort(-demand):
            need = demand[k]

            while need > 0:
                cand = []

                for i in range(self.data.I):
                    if rem_s[i] <= 0:
                        continue

                    for j in range(self.data.J):
                        q = min(rem_s[i], need)
                        if q <= 0:
                            continue

                        cost = self.data.b[i, j] + self.data.c[j, k]

                        if sol.x[i, j] == 0:
                            cost += self.data.f[i, j] / q
                        if sol.y[j, k] == 0:
                            cost += self.data.g[j, k] / q

                        # 🔥 STRONG DC REUSE BIAS
                        if j in sol.open_dc:
                            cost *= 0.92

                        cand.append((cost, i, j, q))

                if not cand:
                    return sol

                cand.sort()
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
# LOCAL SEARCH (MULTI-DC MOVE)
# ============================================================

class LocalSearch:
    def __init__(self, data):
        self.data = data

    def improve(self, sol, iters=80):

        for _ in range(iters):
            improved = False

            for k in range(self.data.K):
                j_from = np.argmax(sol.y[:, k])
                if sol.y[j_from, k] == 0:
                    continue

                best_move = None
                best_delta = 0

                for j_to in range(self.data.J):
                    if j_to == j_from:
                        continue

                    delta = sol.y[j_from, k]

                    cost_old = delta * self.data.c[j_from, k]
                    cost_new = delta * self.data.c[j_to, k]

                    if cost_new < cost_old:

                        trial = sol.copy()
                        trial.y[j_from, k] = 0
                        trial.y[j_to, k] += delta

                        if not self._repair_dc(trial):
                            continue

                        trial.compute_cost()

                        if trial.cost < sol.cost:
                            best_move = trial
                            best_delta = trial.cost

                if best_move:
                    sol = best_move
                    improved = True
                    break

            if not improved:
                break

        return sol

    def _repair_dc(self, sol):
        sol.x[:] = 0

        for j in range(self.data.J):
            demand = np.sum(sol.y[j, :])
            if demand == 0:
                continue

            rem = demand

            # 🔥 better ordering (cheap + capacity aware)
            for i in np.argsort(self.data.b[:, j]):

                cap = self.data.s[i] - np.sum(sol.x[i, :])
                if cap <= 0:
                    continue

                take = min(cap, rem)
                sol.x[i, j] += take
                rem -= take

                if rem == 0:
                    break

            if rem > 0:
                return False

        return True


# ============================================================
# ELITE POOL
# ============================================================

class ElitePool:
    def __init__(self, size=10):
        self.size = size
        self.pool = []

    def add(self, sol):
        self.pool.append(sol.copy())
        self.pool.sort(key=lambda s: s.cost)
        self.pool = self.pool[:self.size]

    def best(self):
        return self.pool[0].copy() if self.pool else None

    def random(self):
        return random.choice(self.pool).copy() if self.pool else None


# ============================================================
# PATH RELINKING (FLOW BLENDING - IMPORTANT FIX)
# ============================================================

class PathRelinking:
    def __init__(self, data):
        self.data = data
        self.ls = LocalSearch(data)

    def relink(self, a, b):
        best = a.copy()

        current = a.copy()

        # 🔥 STEP 1: DC structure alignment
        for j in b.open_dc - current.open_dc:
            current.open_dc.add(j)

        for j in current.open_dc - b.open_dc:
            self._close_dc(current, j)

        # 🔥 STEP 2: PARTIAL FLOW BLEND (IMPORTANT FIX)
        for k in range(self.data.K):
            if random.random() < 0.5:
                j_from = np.argmax(current.y[:, k])
                j_to = np.argmax(b.y[:, k])

                if j_from != j_to:
                    delta = current.y[j_from, k] // 2
                    current.y[j_from, k] -= delta
                    current.y[j_to, k] += delta

        if not self.ls._repair_dc(current):
            return best

        current.compute_cost()
        current = self.ls.improve(current)

        if current.cost < best.cost:
            best = current

        return best

    def _close_dc(self, sol, j):
        for k in range(self.data.K):
            if sol.y[j, k] > 0:
                alt = min(
                    [jj for jj in range(self.data.J) if jj != j],
                    key=lambda jj: self.data.c[jj, k]
                )
                sol.y[alt, k] += sol.y[j, k]
                sol.y[j, k] = 0

        sol.open_dc.discard(j)


# ============================================================
# GRASP
# ============================================================

class GRASP:
    def __init__(self, data, iters=200):
        self.data = data
        self.iters = iters

    def run(self):
        best = None
        best_cost = float("inf")

        ls = LocalSearch(self.data)
        pr = PathRelinking(self.data)
        elite = ElitePool(10)

        start = time.time()

        for it in range(self.iters):

            alpha = random.uniform(0.1, 0.6)

            sol = Construction(self.data, alpha).build()
            sol = ls.improve(sol)

            sol.compute_cost()
            elite.add(sol)

            if sol.cost < best_cost:
                best = sol.copy()
                best_cost = sol.cost
                print(f"[{it}] best = {best_cost}")

            if it % 5 == 0 and len(elite.pool) >= 2:
                a = elite.best()
                b = elite.random()

                cand = pr.relink(a, b)
                if cand and cand.cost < best_cost:
                    best = cand.copy()
                    best_cost = cand.cost
                    print(f"[{it}] PR improved = {best_cost}")

        print("time:", time.time() - start)
        return best, best_cost, time.time() - start