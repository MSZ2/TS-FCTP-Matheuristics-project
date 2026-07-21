import numpy as np
import random
import time
import os
import multiprocessing as mp

# ============================================================
# Objective-function call counter (eval_i in the MA report)
# ============================================================
#
# Per-process counter: in serial mode there is only one process, so
# reading it at the end of run() gives the true total. In parallel
# mode each worker process has its own independent counter; workers
# report their own count back per task (see _construct_and_improve)
# and the main process folds it into its own running total, which
# also picks up the compute_cost() calls made directly in the main
# process (path relinking, shake).

_EVAL_COUNT = 0


def _reset_eval_count():
    global _EVAL_COUNT
    _EVAL_COUNT = 0


def _get_eval_count():
    return _EVAL_COUNT


def _add_eval_count(n):
    global _EVAL_COUNT
    _EVAL_COUNT += n


def _bump_eval_count():
    global _EVAL_COUNT
    _EVAL_COUNT += 1


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
        _bump_eval_count()
        d = self.data
        self.cost = int(np.sum(self.x * d.b) +
                        np.sum(self.y * d.c) +
                        np.sum((self.x > 0) * d.f) +
                        np.sum((self.y > 0) * d.g))
        # Vectorised open-DC test: a Python loop with 2 np.sum() calls
        # per DC used to dominate runtime since this runs after every
        # repair() in local search.
        open_mask = (self.x.sum(axis=0) > 0) | (self.y.sum(axis=1) > 0)
        self.open_dc = set(np.flatnonzero(open_mask).tolist())
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
        # Vectorised candidate scoring: the same RCL/alpha logic as
        # before, but the I x J candidate matrix is built and masked
        # with numpy instead of a Python double loop + list of tuples.
        # This is the dominant cost of construction on large instances
        # (I, J in the hundreds), so vectorising it is where most of
        # the speedup comes from. Note: `j in sol.open_dc` in the old
        # loop was always False here (open_dc is only populated by
        # compute_cost(), called after construction finishes), so
        # dropping that dead multiplier changes nothing.
        d = self.data
        sol = Solution(d)
        rem_s = d.s.copy()

        for k in np.argsort(-d.d):          # largest demand first
            k = int(k)
            need = int(d.d[k])
            while need > 0:
                avail = rem_s > 0
                if not avail.any():
                    return sol

                need_div = max(need, 1)
                base = d.b + d.c[:, k][np.newaxis, :]
                base = base.astype(float)
                base += np.where(sol.x == 0, d.f / need_div, 0.0)
                base += np.where((sol.y[:, k] == 0)[np.newaxis, :],
                                  d.g[:, k][np.newaxis, :] / need_div, 0.0)
                base[~avail, :] = np.inf

                cmin = base.min()
                cmax = base[np.isfinite(base)].max()
                limit = cmin + self.alpha * (cmax - cmin + 1e-9)

                idx_i, idx_j = np.where(base <= limit)
                pick = random.randrange(len(idx_i))
                i, j = int(idx_i[pick]), int(idx_j[pick])

                q = min(int(rem_s[i]), need)
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
        Move a whole customer (all of its incoming flow, even when
        currently split across several DCs) to a single cheaper DC.
        Uses first‑improvement and limits the number of target DCs
        tried per customer to 10 (random order).
        """
        d = self.data
        # For each customer, try at most 10 random target DCs.
        max_targets = min(d.J, 10)

        # Cost currently incurred serving each customer, summed over
        # every DC feeding it. A customer's demand can be split across
        # several DCs, so picking a single "owner" (e.g. via argmax on
        # y > 0) would only account for one arc and silently ignore the
        # rest of that customer's flow.
        old_full = (sol.y * d.c).sum(axis=0) + ((sol.y > 0) * d.g).sum(axis=0)

        for k in range(d.K):
            total = int(sol.y[:, k].sum())
            if total == 0:
                continue

            targets = random.sample(range(d.J), min(d.J, max_targets))

            for j_to in targets:
                new_full = d.c[j_to, k] * d.d[k] + d.g[j_to, k]
                if new_full < old_full[k]:
                    trial = sol.copy()
                    trial.y[:, k] = 0
                    trial.y[j_to, k] = total
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

        # Cost currently incurred serving each customer, summed over
        # every DC feeding it (computed once, reused across candidates
        # below). Demand can be split across several DCs, so a single
        # "owner" lookup (e.g. via argmax on y > 0) would only account
        # for one arc and silently ignore the rest of that customer's
        # flow when the move is applied.
        ks_valid = np.flatnonzero(sol.y.sum(axis=0) > 0)
        old_full = (sol.y[:, ks_valid] * d.c[:, ks_valid]).sum(axis=0) + \
                   ((sol.y[:, ks_valid] > 0) * d.g[:, ks_valid]).sum(axis=0)

        for j_new in closed[:max_candidates]:
            new_full = d.c[j_new, ks_valid] * d.d[ks_valid] + d.g[j_new, ks_valid]
            move_mask = new_full < old_full
            if not move_mask.any():
                continue

            trial = sol.copy()
            ks = ks_valid[move_mask]
            trial.y[:, ks] = 0
            trial.y[j_new, ks] = d.d[ks]

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

            # Vectorised "best alternative DC per reassigned customer":
            # replaces an O(len(alts) * K) Python double loop with a
            # small (len(alts) x |served customers|) numpy computation.
            ks = np.flatnonzero(trial.y[j] > 0)
            if len(ks) > 0:
                alts_arr = np.array(alts)
                qty = trial.y[j, ks]
                var_cost = d.c[alts_arr][:, ks] * qty[np.newaxis, :]
                fixed_cost = np.where(trial.y[alts_arr][:, ks] == 0,
                                       d.g[alts_arr][:, ks], 0)
                best_idx = np.argmin(var_cost + fixed_cost, axis=0)
                best_j = alts_arr[best_idx]
                trial.y[best_j, ks] += qty
                trial.y[j, ks] = 0

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
            ks = np.flatnonzero(trial.y[j_close] > 0)
            trial.y[j_open, ks] += trial.y[j_close, ks]
            trial.y[j_close, ks] = 0
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
# Parallel worker: construction + local search for one iteration
# ============================================================
#
# Each iteration's construction+local search only depends on `alpha`
# and shared instance data (never mutated), so a batch of iterations
# can be built independently in worker processes. Only the cheap
# result (flows + cost) is shipped back, not the (much larger,
# read-only) instance data, which each worker keeps a copy of.

_WORKER_DATA = None


def _init_worker(data):
    global _WORKER_DATA
    _WORKER_DATA = data
    # Fork copies the parent's random state; reseed per-process so
    # workers don't all draw the same RCL choices.
    random.seed(os.getpid() ^ int(time.time() * 1e6) & 0xFFFFFFFF)


def _construct_and_improve(alpha):
    # Reset this worker's own counter so each task reports only the
    # evals it personally performed (Pool reuses worker processes
    # across many tasks, so the counter would otherwise keep growing).
    _reset_eval_count()
    sol = Construction(_WORKER_DATA, alpha).build()
    sol = LocalSearch(_WORKER_DATA).improve(sol)
    sol.compute_cost()
    return sol.x, sol.y, sol.cost, sol.open_dc, _get_eval_count()


# ============================================================
# GRASP main algorithm
# ============================================================

class GRASP:
    def __init__(self, data, iters=300, n_workers=None):
        self.data = data
        self.iters = iters
        # Default to all cores but one, capped so tiny instances
        # (few iterations) don't pay pool-startup overhead for nothing.
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)

    def run(self):
        """
        Returns (best, best_cost, total_time, time_to_best, eval_count):
          - total_time    : ttot_i, wall-clock time for this whole run
          - time_to_best  : t_i, wall-clock time when `best` was first found
          - eval_count    : eval_i, number of Solution.compute_cost() calls
        """
        if self.n_workers <= 1:
            return self._run_serial()
        return self._run_parallel()

    # ----------------------------------------------------------
    # Serial fallback (n_workers=1, or explicitly requested)
    # ----------------------------------------------------------
    def _run_serial(self):
        ls = LocalSearch(self.data)
        pr = PathRelinking(self.data)
        elite = ElitePool(12)
        best = None
        best_cost = float('inf')
        last_improvement = 0
        start = time.time()
        time_to_best = 0.0
        _reset_eval_count()

        for it in range(self.iters):
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
                time_to_best = time.time() - start
                print(f'[{it}] best = {best_cost}')

            if it % 8 == 0 and len(elite.pool) >= 2:
                cand = pr.relink(elite.best(), elite.random())
                if cand and cand.cost < best_cost:
                    best = cand.copy()
                    best_cost = cand.cost
                    last_improvement = it
                    time_to_best = time.time() - start
                    print(f'[{it}] PR = {best_cost}')

            if it - last_improvement > 20 and best:
                best, best_cost, last_improvement, time_to_best = self._shake(
                    best, best_cost, it, ls, start, time_to_best)

        total_time = time.time() - start
        return best, best_cost, total_time, time_to_best, _get_eval_count()

    # ----------------------------------------------------------
    # Parallel: build+improve a batch of iterations concurrently,
    # then apply the same sequential bookkeeping (elite pool,
    # best-tracking, path relinking, shake) the serial version uses.
    # ----------------------------------------------------------
    def _run_parallel(self):
        ls = LocalSearch(self.data)
        pr = PathRelinking(self.data)
        elite = ElitePool(12)
        best = None
        best_cost = float('inf')
        last_improvement = 0
        start = time.time()
        time_to_best = 0.0
        _reset_eval_count()

        n_workers = min(self.n_workers, self.iters)
        pool = mp.Pool(n_workers, initializer=_init_worker, initargs=(self.data,))
        try:
            it = 0
            while it < self.iters:
                batch = min(n_workers, self.iters - it)
                alphas = []
                for offset in range(batch):
                    if (it + offset) - last_improvement > 15:
                        alphas.append(random.uniform(0.4, 0.7))
                    else:
                        alphas.append(random.uniform(0.05, 0.5))

                results = pool.map(_construct_and_improve, alphas)

                for x, y, cost, open_dc, evals in results:
                    _add_eval_count(evals)
                    sol = Solution(self.data)
                    sol.x, sol.y, sol.cost, sol.open_dc = x, y, cost, open_dc
                    elite.add(sol)

                    if sol.cost < best_cost:
                        best = sol.copy()
                        best_cost = sol.cost
                        last_improvement = it
                        time_to_best = time.time() - start
                        print(f'[{it}] best = {best_cost}')

                    # path relinking every 8 iterations
                    if it % 8 == 0 and len(elite.pool) >= 2:
                        cand = pr.relink(elite.best(), elite.random())
                        if cand and cand.cost < best_cost:
                            best = cand.copy()
                            best_cost = cand.cost
                            last_improvement = it
                            time_to_best = time.time() - start
                            print(f'[{it}] PR = {best_cost}')

                    # shake (ruin-and-recreate) when stuck for too long
                    if it - last_improvement > 20 and best:
                        best, best_cost, last_improvement, time_to_best = self._shake(
                            best, best_cost, it, ls, start, time_to_best)

                    it += 1
        finally:
            pool.close()
            pool.join()

        total_time = time.time() - start
        return best, best_cost, total_time, time_to_best, _get_eval_count()

    # ----------------------------------------------------------
    # Ruin-and-recreate shake, shared by serial and parallel runs.
    # ----------------------------------------------------------
    def _shake(self, best, best_cost, it, ls, start, time_to_best):
        d = self.data
        last_improvement = it
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
                    time_to_best = time.time() - start
                    print(f'[{it}] shake = {best_cost}')
        return best, best_cost, last_improvement, time_to_best
