import numpy as np
import random
import time
import os
import multiprocessing as mp

# ============================================================
# Objective-function call counter (eval_i in the MA report)
# ============================================================

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

    def eval_cost(self):
        """Compute and store the objective value (does not refresh `open_dc`)."""
        _bump_eval_count()
        d = self.data
        xr, yr = self.x.ravel(), self.y.ravel()
        self.cost = int(xr @ d.b.ravel() + yr @ d.c.ravel() +
                        d.f.ravel()[xr > 0].sum() +
                        d.g.ravel()[yr > 0].sum())
        return self.cost

    def refresh_open_dc(self):
        """Recompute `open_dc`; call after a solution is actually kept."""
        open_mask = (self.x.sum(axis=0) > 0) | (self.y.sum(axis=1) > 0)
        self.open_dc = set(np.flatnonzero(open_mask).tolist())

    def compute_cost(self):
        """Compute the cost and refresh `open_dc`."""
        self.eval_cost()
        self.refresh_open_dc()
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
    """Greedily assign suppliers (x) to meet the DC demands given by y, cheapest supplier first per DC."""
    sol.x.fill(0)
    rem_s = data.s.copy()
    demands = sol.y.sum(axis=1)
    for j in range(data.J):
        rem = int(demands[j])
        if rem == 0:
            continue
        key = data.b[:, j] * rem + data.f[:, j]
        order = np.lexsort((-rem_s, key))
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
# First-stage polish: 4-cycle exchanges on x with y held fixed
# ============================================================

def polish_x(sol, data, max_moves=200, max_arcs=None):
    """Improve the first stage (suppliers -> DCs) with 4-cycle exchanges, y held fixed. Returns True if improved."""
    if max_arcs is None:
        max_arcs = max(600, 2 * (data.I + data.J))
    b, f = data.b, data.f
    improved = False

    for _ in range(max_moves):
        ii, jj = np.nonzero(sol.x)
        n = len(ii)
        if n < 2:
            break
        if n > max_arcs:
            sel = np.array(random.sample(range(n), max_arcs))
            ii, jj, n = ii[sel], jj[sel], max_arcs

        q = sol.x[ii, jj]                              # (n,)
        dmax = np.minimum(q[:, None], q[None, :])      # (n,n)

        # variable part: b[i1,j2] + b[i2,j1] - b[i1,j1] - b[i2,j2]
        cross = b[np.ix_(ii, jj)]                      # cross[a,c] = b[i_a, j_c]
        var = cross + cross.T                          # b[i1,j2] + b[i2,j1]
        diag = b[ii, jj]                               # b[i_a, j_a]
        var = var - diag[:, None] - diag[None, :]

        # fixed part: pay f on newly opened arcs, save f on closed ones
        opened = (sol.x[np.ix_(ii, jj)] == 0)          # is (i_a, j_c) unused?
        fx = f[np.ix_(ii, jj)]
        fixed = np.where(opened, fx, 0) + np.where(opened.T, fx.T, 0)
        closes = (q[:, None] == dmax)
        fixed = fixed - np.where(closes, f[ii, jj][:, None], 0) \
                      - np.where(closes.T, f[ii, jj][None, :], 0)

        delta = dmax * var + fixed

        # invalid pairs: same supplier, same DC, or zero transfer
        bad = (ii[:, None] == ii[None, :]) | (jj[:, None] == jj[None, :]) \
            | (dmax <= 0)
        delta = np.where(bad, 0, delta)

        a, c = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[a, c] >= 0:
            break

        amt = int(dmax[a, c])
        i1, j1, i2, j2 = int(ii[a]), int(jj[a]), int(ii[c]), int(jj[c])
        sol.x[i1, j1] -= amt
        sol.x[i2, j2] -= amt
        sol.x[i1, j2] += amt
        sol.x[i2, j1] += amt
        improved = True

    if improved:
        sol.compute_cost()
    return improved


# ============================================================
# GRASP construction
# ============================================================

def random_dc_mask(J, p_restrict=0.5, min_frac=0.35):
    """Random boolean mask of DCs the construction is allowed to use, or None for all of them."""
    if random.random() >= p_restrict:
        return None
    lo = max(1, int(J * min_frac))
    if lo >= J:
        return None
    k = random.randint(lo, J - 1)
    allowed = np.zeros(J, dtype=bool)
    allowed[random.sample(range(J), k)] = True
    return allowed


class Construction:
    def __init__(self, data, alpha=0.3, allowed_dc=None):
        self.data = data
        self.alpha = alpha
        self.allowed_dc = allowed_dc

    def build(self):
        """Greedy-randomized construction: assign demand largest-first, picking from an RCL of cheap (supplier, DC) pairs."""
        d = self.data
        sol = Solution(d)
        rem_s = d.s.copy()
        blocked = None if self.allowed_dc is None else ~self.allowed_dc

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
                if blocked is not None:
                    base[:, blocked] = np.inf

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
    """VND local search over DC-structure moves. `allowed_dc` restricts which closed DCs a move may open."""

    # ---- per-move screening budgets: how many candidates each move
    # ranks and then fully evaluates per call -----------------------

    # `_reassign_customer`
    MAX_FULL_TARGETS = 6
    N_EXPLORE = 2

    # `_shift_flow`: arcs sampled, and candidates fully evaluated
    SHIFT_MAX_ARCS = 200
    SHIFT_MAX_TRIALS = 40

    # `_add_dc` / `_remove_dc`: DCs tried per call
    ADD_MAX_CANDIDATES = 36
    REMOVE_MAX_CANDIDATES = 20

    # `_swap_dc`: (open, closed) pairs ranked per call, and how many of
    # those are then evaluated in full
    SWAP_MAX_PAIRS = 150
    SWAP_MAX_TRIALS = 12
    SWAP_EXPLORE = 2

    def __init__(self, data, allowed_dc=None):
        self.data = data
        self.allowed_dc = allowed_dc

    def improve(self, sol, max_no_improve=10):
        """VND with first-improvement: cycle through the five neighbourhoods until none improves, then polish x."""
        repair(sol, self.data)
        sol.compute_cost()
        no_improve = 0
        while no_improve < max_no_improve:

            if self._reassign_customer(sol):
                no_improve = 0
                continue

            if self._add_dc(sol):
                no_improve = 0
                continue

            if self._remove_dc(sol):
                no_improve = 0
                continue

            if self._swap_dc(sol):
                no_improve = 0
                continue

            if self._shift_flow(sol):
                no_improve = 0
                continue

            no_improve += 1

        polish_x(sol, self.data)
        return sol

    # -----------------------------------------------------------------
    def _may_open(self, j):
        """May a currently closed DC be opened by a move?"""
        return self.allowed_dc is None or bool(self.allowed_dc[j])

    # -----------------------------------------------------------------
    def _accept(self, sol, trial):
        """Repair x, evaluate, and copy `trial` into `sol` if it is cheaper."""
        if not repair(trial, self.data):
            return False
        trial.eval_cost()
        if trial.cost < sol.cost:
            trial.refresh_open_dc()
            sol.x = trial.x
            sol.y = trial.y
            sol.cost = trial.cost
            sol.open_dc = trial.open_dc
            return True
        return False

    # -----------------------------------------------------------------
    def _reassign_customer(self, sol):
        """Consolidation move: try putting all of a customer's flow on a single other DC."""
        d = self.data
        max_targets = min(d.J, self.MAX_FULL_TARGETS)
        n_explore = self.N_EXPLORE

        # Cost currently incurred serving each customer, summed over
        # every DC feeding it (a customer's demand can be split).
        old_full = (sol.y * d.c).sum(axis=0) + ((sol.y > 0) * d.g).sum(axis=0)

        served = np.flatnonzero(sol.y.sum(axis=0) > 0).tolist()
        random.shuffle(served)

        for k in served:
            total = int(sol.y[:, k].sum())

            # second-stage cost of serving k entirely from each DC
            new_full = d.c[:, k] * total + d.g[:, k]
            order = [j for j in np.argsort(new_full).tolist()
                     if j in sol.open_dc or self._may_open(j)]

            improving = [j for j in order if new_full[j] < old_full[k]][:max_targets]
            rest = [j for j in order if new_full[j] >= old_full[k]]
            targets = improving + random.sample(rest, min(len(rest), n_explore))

            for j_to in targets:
                if sol.y[j_to, k] == total:
                    continue                      # already fully there
                trial = sol.copy()
                trial.y[:, k] = 0
                trial.y[j_to, k] = total
                if self._accept(sol, trial):
                    return True
        return False

    # -----------------------------------------------------------------
    def _shift_flow(self, sol):
        """Split-aware move: shift a partial quantity of one arc (j_from -> k) onto another DC j_to."""
        d = self.data
        max_arcs = self.SHIFT_MAX_ARCS
        max_trials = self.SHIFT_MAX_TRIALS
        S = max(1, int(d.s.min()))

        jf, kk = np.nonzero(sol.y)
        if len(jf) == 0:
            return False
        if len(jf) > max_arcs:
            sel = np.array(random.sample(range(len(jf)), max_arcs))
            jf, kk = jf[sel], kk[sel]

        q_arc = sol.y[jf, kk].astype(np.int64)          # (A,)
        t = sol.y.sum(axis=1).astype(np.int64)          # (J,) DC throughput

        A, J = len(jf), d.J
        # quantity variants, shape (A, nq, J) after broadcasting
        q_full = np.broadcast_to(q_arc[:, None, None], (A, 1, J))
        q_half = np.broadcast_to((q_arc // 2)[:, None, None], (A, 1, J))
        q_af = np.broadcast_to((t[jf] % S)[:, None, None], (A, 1, J))
        q_at = np.broadcast_to(((-t) % S)[None, None, :], (A, 1, J))
        q = np.concatenate([q_full, q_half, q_af, q_at], axis=1)
        q = np.minimum(q, q_arc[:, None, None])          # never exceed the arc

        c_from = d.c[jf, kk][:, None, None]              # (A,1,1)
        c_to = d.c[:, kk].T[:, None, :]                  # (A,1,J)
        g_to = d.g[:, kk].T[:, None, :]                  # (A,1,J)
        g_from = d.g[jf, kk][:, None, None]              # (A,1,1)
        new_arc = (sol.y[:, kk].T == 0)[:, None, :]      # target arc unused?

        delta = q * (c_to - c_from) \
            + np.where(new_arc, g_to, 0) \
            - np.where(q == q_arc[:, None, None], g_from, 0)

        # invalid candidates: zero quantity, or target == source DC
        same = (np.arange(J)[None, None, :] == jf[:, None, None])
        blocked = (q <= 0) | same
        if self.allowed_dc is not None:
            # a closed, blacklisted DC must stay closed
            shut = ~self.allowed_dc & (sol.y.sum(axis=1) == 0)
            blocked = blocked | shut[None, None, :]
        delta = np.where(blocked, np.inf, delta.astype(float))

        flat = delta.ravel()
        n = min(max_trials, int(np.isfinite(flat).sum()))
        if n == 0:
            return False
        cand = np.argpartition(flat, n - 1)[:n]
        cand = cand[np.argsort(flat[cand])]

        for idx in cand:
            a, v, j_to = np.unravel_index(idx, delta.shape)
            amount = int(q[a, v, j_to])
            j_from, k = int(jf[a]), int(kk[a])
            trial = sol.copy()
            trial.y[j_from, k] -= amount
            trial.y[j_to, k] += amount
            if self._accept(sol, trial):
                return True
        return False

    # -----------------------------------------------------------------
    def _add_dc(self, sol):
        """Try opening a closed DC, migrating flow to it arc by arc."""
        d = self.data
        closed = [j for j in range(d.J)
                  if j not in sol.open_dc and self._may_open(j)]
        if not closed:
            return False

        random.shuffle(closed)
        max_candidates = min(len(closed), self.ADD_MAX_CANDIDATES)

        jf, kk = np.nonzero(sol.y)
        if len(jf) == 0:
            return False
        q = sol.y[jf, kk]

        for j_new in closed[:max_candidates]:
            delta = q * (d.c[j_new, kk] - d.c[jf, kk]) \
                - d.g[jf, kk] + d.g[j_new, kk]

            take = np.flatnonzero(delta < 0)
            order = np.argsort(delta)[:3]

            for arcs in (take, order):
                if len(arcs) == 0:
                    continue
                trial = sol.copy()
                np.add.at(trial.y, (j_new, kk[arcs]), q[arcs])
                trial.y[jf[arcs], kk[arcs]] = 0
                if self._accept(sol, trial):
                    return True
        return False

    # -----------------------------------------------------------------
    def _remove_dc(self, sol):
        """Try closing an open DC, reassigning its flow to the cheapest remaining alternative."""
        d = self.data
        open_dcs = list(sol.open_dc)
        if len(open_dcs) <= 1:
            return False

        random.shuffle(open_dcs)
        max_candidates = min(len(open_dcs), self.REMOVE_MAX_CANDIDATES)

        for j in open_dcs[:max_candidates]:
            trial = sol.copy()
            alts = [jj for jj in trial.open_dc if jj != j]
            if not alts:
                continue

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

            if self._accept(sol, trial):
                return True
        return False

    # -----------------------------------------------------------------
    def _swap_dc(self, sol):
        """Try swapping one open DC for one closed DC, reassigning freed flow to the cheapest alternative."""
        d = self.data
        open_list = list(sol.open_dc)
        closed = [j for j in range(d.J)
                  if j not in sol.open_dc and self._may_open(j)]
        if not open_list or not closed:
            return False
        closed_arr = np.array(closed)

        scored = []
        for j_close in open_list:
            ks = np.flatnonzero(sol.y[j_close] > 0)
            if len(ks) == 0:
                continue
            qty = sol.y[j_close, ks]
            base = d.c[:, ks] * qty + np.where(sol.y[:, ks] == 0, d.g[:, ks], 0)
            others = [j for j in open_list if j != j_close]
            best_other = (base[np.array(others)].min(axis=0) if others
                          else np.full(len(ks), np.inf))
            new_cost = np.minimum(base[closed_arr], best_other[None, :]).sum(axis=1)
            old_cost = (d.c[j_close, ks] * qty).sum() + d.g[j_close, ks].sum()
            for idx, j_open in enumerate(closed):
                scored.append((float(new_cost[idx] - old_cost), j_close, j_open))
        if not scored:
            return False

        scored.sort(key=lambda t: t[0])
        pairs = [(jc, jo) for _, jc, jo in scored[:self.SWAP_MAX_TRIALS]]
        rest = scored[self.SWAP_MAX_TRIALS:self.SWAP_MAX_PAIRS]
        if rest:
            pairs += [(jc, jo) for _, jc, jo in
                      random.sample(rest, min(len(rest), self.SWAP_EXPLORE))]

        for j_close, j_open in pairs:
            trial = sol.copy()
            ks = np.flatnonzero(trial.y[j_close] > 0)
            if len(ks) == 0:
                continue
            alts = np.array([j_open] + [j for j in open_list if j != j_close])
            qty = trial.y[j_close, ks]
            var_cost = d.c[alts][:, ks] * qty[np.newaxis, :]
            fixed_cost = np.where(trial.y[alts][:, ks] == 0,
                                  d.g[alts][:, ks], 0)
            best_j = alts[np.argmin(var_cost + fixed_cost, axis=0)]
            trial.y[j_close, ks] = 0
            np.add.at(trial.y, (best_j, ks), qty)
            if self._accept(sol, trial):
                return True
        return False


# ============================================================
# Elite pool with diversity
# ============================================================

class ElitePool:
    """Keeps the top `size` solutions, enforcing a minimum Jaccard distance between their DC sets for diversity."""

    def __init__(self, size=12, min_dist=0.2):
        self.size = size
        self.min_dist = min_dist
        self.pool = []

    @staticmethod
    def _dist(a, b):
        """Jaccard distance between two solutions' open-DC sets."""
        sa, sb = a.open_dc, b.open_dc
        union = len(sa | sb)
        return 1.0 if union == 0 else 1.0 - len(sa & sb) / union

    def add(self, sol):
        """Insert `sol` if it beats its nearest neighbour or the pool has room; evict the worst member when full."""
        near, near_d = None, 2.0
        for p in self.pool:
            dd = self._dist(p, sol)
            if dd < near_d:
                near, near_d = p, dd

        full = len(self.pool) >= self.size
        thr = self.min_dist if full else 0.0

        if near is not None and near_d <= thr:
            if sol.cost < near.cost:
                self.pool.remove(near)
            else:
                return
        elif full:
            worst = max(self.pool, key=lambda s: s.cost)
            if sol.cost >= worst.cost:
                return
            self.pool.remove(worst)

        self.pool.append(sol.copy())
        self.pool.sort(key=lambda s: s.cost)

    def spread(self):
        """Mean pairwise Jaccard distance -- diagnostic only."""
        n = len(self.pool)
        if n < 2:
            return 0.0
        tot = sum(self._dist(self.pool[i], self.pool[j])
                  for i in range(n) for j in range(i + 1, n))
        return tot / (n * (n - 1) / 2)

    def best(self):
        return self.pool[0].copy() if self.pool else None

    def random(self):
        return random.choice(self.pool).copy() if self.pool else None


# ============================================================
# Path relinking
# ============================================================

class PathRelinking:
    """Column-exchange path relinking between two elite solutions."""

    def __init__(self, data):
        self.data = data
        self.ls = LocalSearch(data)

    def _walk(self, a, b, max_steps, max_cand):
        """Move `a` towards `b` one customer column at a time."""
        d = self.data
        cur = a.copy()
        diff = [k for k in range(d.K)
                if not np.array_equal(cur.y[:, k], b.y[:, k])]
        best, best_cost = None, float('inf')

        steps = 0
        while diff and steps < max_steps:
            cands = random.sample(diff, min(len(diff), max_cand))
            step_best, step_k = None, None
            for k in cands:
                trial = cur.copy()
                trial.y[:, k] = b.y[:, k]
                if not repair(trial, d):
                    continue
                trial.compute_cost()
                if step_best is None or trial.cost < step_best.cost:
                    step_best, step_k = trial, k
            if step_best is None:
                break
            cur, diff = step_best, [k for k in diff if k != step_k]
            steps += 1
            if cur.cost < best_cost:
                best, best_cost = cur.copy(), cur.cost
        return best

    def relink(self, a, b, max_steps=15, max_cand=8):
        """Walk from a to b and from b to a, keep the best point seen, and run local search on it."""
        best = None
        for src, dst in ((a, b), (b, a)):
            cand = self._walk(src, dst, max_steps, max_cand)
            if cand is not None and (best is None or cand.cost < best.cost):
                best = cand
        if best is None:
            return a.copy()
        return self.ls.improve(best)


# ============================================================
# Parallel worker: construction + local search for one iteration
# ============================================================

_WORKER_DATA = None


def _init_worker(data):
    """Give each worker process its own copy of the instance data and RNG seed."""
    global _WORKER_DATA
    _WORKER_DATA = data
    random.seed(os.getpid() ^ int(time.time() * 1e6) & 0xFFFFFFFF)


def perturb(sol, data, max_close=None):
    """ILS kick: close a random handful of open DCs and dump their arcs onto random other DCs."""
    open_dcs = list(sol.open_dc)
    if len(open_dcs) < 2:
        return
    if max_close is None:
        max_close = max(1, len(open_dcs) // 3)
    n_close = random.randint(1, max_close)
    victims = random.sample(open_dcs, min(n_close, len(open_dcs) - 1))
    survivors = [j for j in range(data.J) if j not in victims]

    for j in victims:
        ks = np.flatnonzero(sol.y[j] > 0)
        for k in ks:
            j_to = random.choice(survivors)
            sol.y[j_to, k] += sol.y[j, k]
            sol.y[j, k] = 0


def elite_dc_mask(sol, J, n_extra=1):
    """Mask confining an ILS re-descent to the seed's own DC configuration, plus `n_extra` random closed DCs."""
    mask = np.zeros(J, dtype=bool)
    mask[list(sol.open_dc)] = True
    closed = np.flatnonzero(~mask)
    if n_extra and len(closed):
        for j in random.sample(list(closed), min(n_extra, len(closed))):
            mask[j] = True
    return mask


_WORKER_PR = None


def _construct_and_improve(task):
    """
    One parallel task; returns (x, y, cost, open_dc, evals).

      * ('ls', alpha, None)    -> ordinary GRASP restart (construction + VND).
      * ('ls', alpha, seed_y)  -> ILS iteration: perturb an elite solution and re-descend.
      * ('pr', xa, ya, xb, yb) -> path relinking between two elite solutions.
    """
    _reset_eval_count()
    d = _WORKER_DATA

    if task[0] == 'pr':
        global _WORKER_PR
        if _WORKER_PR is None:
            _WORKER_PR = PathRelinking(d)
        _, xa, ya, xb, yb = task
        a, b = Solution(d), Solution(d)
        a.x, a.y = xa.copy(), ya.copy()
        b.x, b.y = xb.copy(), yb.copy()
        a.compute_cost()
        b.compute_cost()
        cand = _WORKER_PR.relink(a, b)
        cand.compute_cost()
        return cand.x, cand.y, cand.cost, cand.open_dc, _get_eval_count()

    _, alpha, seed_y = task

    if seed_y is not None:
        sol = Solution(d)
        sol.y = seed_y.copy()
        sol.compute_cost()
        mask = elite_dc_mask(sol, d.J)
        perturb(sol, d)
        sol = LocalSearch(d, mask).improve(sol)
        sol.compute_cost()
        return sol.x, sol.y, sol.cost, sol.open_dc, _get_eval_count()

    mask = random_dc_mask(d.J)
    sol = Construction(d, alpha, mask).build()
    sol = LocalSearch(d, mask).improve(sol)
    sol.compute_cost()
    return sol.x, sol.y, sol.cost, sol.open_dc, _get_eval_count()


# ============================================================
# GRASP main algorithm
# ============================================================

class GRASP:
    def __init__(self, data, iters=300, n_workers=None, p_seed=0.5):
        self.data = data
        self.iters = iters
        self.p_seed = p_seed  # fraction of iterations that perturb an elite solution (ILS) instead of restarting
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)

    def run(self):
        """Run GRASP; returns (best, best_cost, total_time, time_to_best, eval_count)."""
        if self.n_workers <= 1:
            return self._run_serial()
        return self._run_parallel()

    def _run_serial(self):
        """Serial GRASP loop (n_workers=1): restart or ILS re-descend each iteration, with periodic PR and shake."""
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

            if elite.pool and random.random() < self.p_seed:
                sol = Solution(self.data)
                sol.y = random.choice(elite.pool).y.copy()
                sol.compute_cost()
                mask = elite_dc_mask(sol, self.data.J)
                perturb(sol, self.data)
                sol = LocalSearch(self.data, mask).improve(sol)
            else:
                mask = random_dc_mask(self.data.J)
                sol = Construction(self.data, alpha, mask).build()
                sol = LocalSearch(self.data, mask).improve(sol)
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
                if cand:
                    elite.add(cand)
                    if cand.cost < best_cost:
                        best = cand.copy()
                        best_cost = cand.cost
                        last_improvement = it
                        time_to_best = time.time() - start
                        print(f'[{it}] PR = {best_cost}')

            if it - last_improvement > 20 and best:
                best, best_cost, last_improvement, time_to_best = self._shake(
                    best, best_cost, it, ls, start, time_to_best, elite)

        total_time = time.time() - start
        return best, best_cost, total_time, time_to_best, _get_eval_count()

    def _run_parallel(self):
        """Parallel GRASP: build+improve a batch of iterations concurrently, then apply the serial bookkeeping."""
        ls = LocalSearch(self.data)
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
                tasks = []
                for offset in range(batch):
                    if (it + offset) - last_improvement > 15:
                        alpha = random.uniform(0.4, 0.7)
                    else:
                        alpha = random.uniform(0.05, 0.5)
                    seed = None
                    if elite.pool and random.random() < self.p_seed:
                        seed = random.choice(elite.pool).y
                    tasks.append(('ls', alpha, seed))

                n_pr = (sum(1 for off in range(batch) if (it + off) % 8 == 0)
                        if len(elite.pool) >= 2 else 0)
                for _ in range(n_pr):
                    pa, pb = elite.best(), elite.random()
                    tasks.append(('pr', pa.x, pa.y, pb.x, pb.y))

                results = pool.map(_construct_and_improve, tasks)
                pr_results = iter(results[batch:])

                for x, y, cost, open_dc, evals in results[:batch]:
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

                    if it % 8 == 0:
                        pr_res = next(pr_results, None)
                        if pr_res is not None:
                            px, py, pcost, popen, pevals = pr_res
                            _add_eval_count(pevals)
                            cand = Solution(self.data)
                            cand.x, cand.y = px, py
                            cand.cost, cand.open_dc = pcost, popen
                            elite.add(cand)
                            if cand.cost < best_cost:
                                best = cand.copy()
                                best_cost = cand.cost
                                last_improvement = it
                                time_to_best = time.time() - start
                                print(f'[{it}] PR = {best_cost}')

                    if it - last_improvement > 20 and best:
                        best, best_cost, last_improvement, time_to_best = self._shake(
                            best, best_cost, it, ls, start, time_to_best, elite)

                    it += 1
        finally:
            pool.close()
            pool.join()

        total_time = time.time() - start
        return best, best_cost, total_time, time_to_best, _get_eval_count()

    def _shake(self, best, best_cost, it, ls, start, time_to_best, elite=None):
        """Ruin-and-recreate: close the costliest half of the open DCs and reconstruct, then re-improve."""
        d = self.data
        last_improvement = it

        src = best

        if len(src.open_dc) > 1:
            # score DCs by average variable cost per unit
            dc_scores = []
            for j in src.open_dc:
                demand = src.y[j, :].sum()
                avg_var = 0 if demand == 0 else (src.x[:, j] * d.b[:, j]).sum() / demand
                dc_scores.append((avg_var, j))
            dc_scores.sort(reverse=True)
            n_remove = max(1, int(len(dc_scores) * 0.5))
            remove_set = {j for _, j in dc_scores[:n_remove]}

            trial = Solution(d)
            # keep flows of non‑removed DCs
            for k in range(d.K):
                for j in src.open_dc:
                    if j not in remove_set and src.y[j, k] > 0:
                        trial.y[j, k] = src.y[j, k]

            # ensure every customer's demand is fully satisfied
            allowed_dcs = [j for j in range(d.J) if j not in remove_set]
            if not allowed_dcs:
                allowed_dcs = list(src.open_dc)
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
                if trial.is_valid():
                    if elite is not None:
                        elite.add(trial)
                    if trial.cost < best_cost:
                        best = trial.copy()
                        best_cost = trial.cost
                        time_to_best = time.time() - start
                        print(f'[{it}] shake = {best_cost}')
        return best, best_cost, last_improvement, time_to_best
