import numpy as np
import random
import time
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
        self.cost = int(np.sum(self.x * d.b) + np.sum(self.y * d.c) + np.sum((self.x > 0) * d.f) + np.sum((self.y > 0) * d.g))
        self.open_dc = {j for j in range(d.J) if np.sum(self.x[:, j]) > 0 or np.sum(self.y[j, :]) > 0}
        return self.cost

    def is_valid(self):
        d = self.data
        return all(np.sum(self.y[:, k]) == d.d[k] for k in range(d.K)) and all(np.sum(self.x[i, :]) <= d.s[i] for i in range(d.I)) and all(np.sum(self.x[:, j]) == np.sum(self.y[j, :]) for j in range(d.J))

def repair(sol, data):
    sol.x[:] = 0
    rem_s = data.s.copy()
    for j in range(data.J):
        demand = int(np.sum(sol.y[j, :]))
        if demand == 0:
            continue
        rem = demand
        order = sorted(range(data.I), key=lambda i: (data.b[i, j] + data.f[i, j] / max(demand, 1), -rem_s[i]))
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

class Construction:
    def __init__(self, data, alpha=0.3):
        self.data = data
        self.alpha = alpha

    def build(self):
        d = self.data
        sol = Solution(d)
        rem_s = d.s.copy()
        for k in np.argsort(-d.d):
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
                        if j in sol.open_dc:
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

class LocalSearch:
    def __init__(self, data):
        self.data = data

    def improve(self, sol, max_no_improve=6):
        d = self.data
        sol.compute_cost()
        no_improve = 0
        while no_improve < max_no_improve:
            best_neighbor = None
            best_cost = sol.cost
            for k in range(d.K):
                for j_from in range(d.J):
                    qty = sol.y[j_from, k]
                    if qty == 0:
                        continue
                    for j_to in range(d.J):
                        if j_to == j_from:
                            continue
                        trial = sol.copy()
                        trial.y[j_to, k] += qty
                        trial.y[j_from, k] = 0
                        if repair(trial, d):
                            trial.compute_cost()
                            if trial.cost < best_cost:
                                best_cost = trial.cost
                                best_neighbor = trial
            if best_neighbor:
                sol = best_neighbor
                no_improve = 0
            else:
                no_improve += 1
        return sol

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

class PathRelinking:
    def __init__(self, data):
        self.data = data
        self.ls = LocalSearch(data)

    def relink(self, a, b):
        d = self.data
        current = a.copy()
        best = current.copy()
        best_cost = current.cost
        for j in list(a.open_dc ^ b.open_dc):
            if j in b.open_dc:
                current.open_dc.add(j)
            else:
                current.open_dc.discard(j)
            if not repair(current, d):
                continue
            current.compute_cost()
            current = self.ls.improve(current)
            if current.cost < best_cost:
                best = current.copy()
                best_cost = current.cost
        return best

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
        start = time.time()
        for it in range(self.iters):
            sol = Construction(self.data, random.uniform(0.05, 0.5)).build()
            sol = ls.improve(sol)
            sol.compute_cost()
            elite.add(sol)
            if sol.cost < best_cost:
                best = sol.copy()
                best_cost = sol.cost
                print(f'[{it}] best = {best_cost}')
            if it % 8 == 0 and len(elite.pool) >= 2:
                cand = pr.relink(elite.best(), elite.random())
                if cand and cand.cost < best_cost:
                    best = cand.copy()
                    best_cost = cand.cost
                    print(f'[{it}] PR = {best_cost}')
        return best, best_cost, time.time() - start