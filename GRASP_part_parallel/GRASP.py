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

    def eval_cost(self):
        """
        Objective value only -- `open_dc` is deliberately NOT refreshed.

        This is the screening path: local search evaluates ~2000 trials
        per iteration and rejects almost all of them, and a rejected
        trial's `open_dc` is never read. Building it cost 27-29 % of the
        old `compute_cost` (a J-wide pair of axis-sums plus a Python
        `set`), so it now happens only on acceptance, via
        `refresh_open_dc()`.

        Flat dot products instead of `np.sum(x * b)`: the latter
        materialises an I x J temporary per term, which on the small
        instances is pure numpy dispatch overhead. Measured on small_1
        the whole call goes 13.30 us -> 3.99 us; the value is identical
        (integer dot products, no floating point involved).
        """
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
        """Cost + `open_dc`, for callers that need a fully consistent solution."""
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
    """
    Assign suppliers (x) to meet the DC demands given by y.

    Same greedy as before: DCs in index order, and for each DC the
    suppliers cheapest-first (ties broken by largest remaining capacity),
    each giving everything it has left until the DC's demand is met.

    This runs on every one of the ~2000 trials per local-search
    iteration and was 53-60 % of total runtime, so the two per-DC numpy
    allocations are what matter -- not the inner loop, which almost
    always breaks after one or two suppliers (s[i] = 80 covers most
    single-DC demands). Two changes, both order-preserving:

      * the J separate `sol.y[j].sum()` calls collapse into one axis-sum;
      * the sort key `b[:,j] + f[:,j]/rem` becomes the integer
        `b[:,j]*rem + f[:,j]`, which is the same key scaled by rem > 0.
        Same ranking, no float division, no float temporary -- and it
        cannot lose precision the way the division could.

    Measured 1.4x (small_1) to 2.2x (medium_1) faster, with identical
    `x` on 1000 sampled local-search states. Fully vectorising the inner
    loop (cumsum + searchsorted over all suppliers) was also tried and
    was *slower* -- it does O(I) work where the loop's early break does
    O(1-2).
    """
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

def polish_x(sol, data, max_moves=200, max_arcs=600):
    """
    Improve the first stage (suppliers -> DCs) without touching y.

    `repair()` is a greedy that walks DCs in index order, so DC 0 always
    gets first pick of the cheapest suppliers; its output is not even
    locally optimal w.r.t. the simplest transportation move. This does
    the missing local search:

        x[i1,j1] -= d ;  x[i1,j2] += d
        x[i2,j2] -= d ;  x[i2,j1] += d

    which preserves both the row sums (supplier usage) and the column
    sums (DC throughput), so feasibility and `open_dc` are untouched.

    Candidates are pairs of *currently used* arcs (x > 0) with distinct
    suppliers and distinct DCs -- there are only O(I + J) used arcs, so
    the whole neighbourhood is an n x n numpy computation. The delta is
    evaluated in closed form (no compute_cost() per candidate):

        D = d*(b[i1,j2] + b[i2,j1] - b[i1,j1] - b[i2,j2])
          + f[i1,j2]*[x[i1,j2] = 0] + f[i2,j1]*[x[i2,j1] = 0]
          - f[i1,j1]*[x[i1,j1] = d] - f[i2,j2]*[x[i2,j2] = d]

    Only d = min(x[i1,j1], x[i2,j2]) is tried: D is linear in d plus
    fixed terms that fire only when an arc closes, which needs d = dmax.
    If the variable term is negative, bigger d is better (dmax); if it
    is positive, the only possible gain is the fixed saving (also dmax).

    Best-improvement, at most `max_moves` moves. Returns True if the
    solution got cheaper.
    """
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
    """
    Boolean mask of DCs the construction is allowed to use, or None for
    "all of them".

    Diversification at the DC level. `alpha` only randomises *which
    cheap (supplier, DC) pair* is picked, not *which set of DCs* the
    solution ends up with: the generator sets f = b*r with r in [10,15],
    so the RCL score `b + f/need` is dominated by `b` and the same few
    cheap DCs win every restart regardless of alpha. Measured on
    small_5 (J = 12): 80 restarts produced only 9 distinct DC sets out
    of 2^12 possible, and the elite pool collapsed to near-clones
    (mean Jaccard distance 0.07).

    Blacklisting a random subset of DCs per restart forces the greedy
    into a different region. Feasibility is never at risk: there is no
    DC capacity, so any non-empty allowed set can absorb all demand.
    Local search may still open blacklisted DCs afterwards -- the mask
    only biases the starting point.
    """
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
    """
    `allowed_dc` is an optional boolean mask restricting which DCs the
    DC-opening moves may use. Without it, blacklisting DCs in the
    construction is pointless: `_add_dc` / `_swap_dc` simply re-open
    them and every restart converges back to the same DC set (measured
    on small_5: 80 restarts -> 9 distinct DC sets with or without a
    construction-only mask). Keeping the restriction alive through the
    VND is what actually makes a restart explore a different region.
    """

    # How many second-stage-improving targets `_reassign_customer` puts
    # through a full (repair + eval_cost) evaluation. The move was 58 %
    # of total runtime because it fully evaluated every one of up to
    # min(J, 36) ranked targets; `_shift_flow` already screens the same
    # way (`max_trials`). A class attribute so experiments can sweep it
    # without editing the move.
    MAX_FULL_TARGETS = 6

    def __init__(self, data, allowed_dc=None):
        self.data = data
        self.allowed_dc = allowed_dc

    def improve(self, sol, max_no_improve=10):
        """
        VND with first‑improvement and bounded candidate evaluations.
        All neighbourhoods are checked in order; on success we reset.

        Order matters: the four coarse (whole-customer / whole-DC) moves
        run first because they are cheap, and `_shift_flow` -- the only
        move that can *create* a split by moving part of an arc -- runs
        last, i.e. it fine-tunes the split structure once the coarse
        moves are exhausted.

        Two-level evaluation. Inside the VND every candidate is scored
        with plain `repair()`, so all comparisons share one basis. The
        incoming solution is re-`repair()`ed for the same reason: an
        already-polished `x` (arriving from PR or shake) would make the
        incumbent look cheaper than every trial and freeze the search.
        `polish_x` then runs once, at convergence, so everything leaving
        this method carries a genuinely optimised first stage -- which
        keeps the elite pool, best-tracking, PR and shake consistent
        with each other, since all of them only ever see solutions that
        came out of here.
        """
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
        """
        Repair x, evaluate, and copy `trial` into `sol` if it is better.

        `eval_cost()` rather than `compute_cost()`: `open_dc` is only
        refreshed on the accepting branch, since a rejected trial's is
        never read. Every move reads `sol.open_dc`, so the kept solution
        must still carry a correct one.
        """
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
        """
        Consolidation move: put *all* of a customer's flow on one DC.

        This is deliberately a single-source move -- it is the move that
        *undoes* a split. It is no longer the only customer-level move
        (see `_shift_flow`, which does the opposite), so restricting it
        to single-sourcing no longer biases the search.

        The second-stage cost is used to *rank* target DCs, not to veto
        them: the previous hard filter `new_full < old_full[k]` made the
        move blind to consolidations that pay off only in the first
        stage. A small random exploration quota is evaluated on top of
        the second-stage-improving targets.

        Only the `MAX_FULL_TARGETS` best-ranked improving targets are
        evaluated in full, since `new_full` already orders them by exact
        second-stage cost -- the same two-level screening `_shift_flow`
        uses. Evaluating all min(J, 36) of them made this move 58 % of
        total runtime.
        """
        d = self.data
        max_targets = min(d.J, self.MAX_FULL_TARGETS)
        n_explore = 2

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
        """
        Split-aware move: shift a *partial* quantity of one arc
        (j_from -> k) onto another DC j_to.

        This is the move that lets the search reach solutions in which a
        customer is served by several DCs at once. Splitting is never
        profitable in the second stage alone (c*q is linear and g is a
        pure surcharge), so it only ever pays off through the first
        stage: every supplier holds only s[i] = 80 units, so the
        marginal inbound cost of a DC *rises* with its throughput once
        its cheap suppliers are exhausted. Spreading a customer over two
        DCs can therefore be strictly cheaper overall.

        Candidate quantities per arc (q = y[j_from, k]):
          * q            -- move the whole arc
          * q // 2       -- halve the arc
          * t[j_from] % S -- make the *source* DC's throughput a multiple
                             of the supplier capacity S = min(s), which
                             can free a whole supplier arc in stage 1
          * (-t[j_to]) % S -- same, for the *target* DC

        All (arc, quantity, target) candidates are scored by their exact
        second-stage delta, and the `max_trials` least-penalising ones
        are evaluated in full (repair + compute_cost, first improvement).
        Positive-delta candidates are explicitly kept: they are the ones
        whose gain lives in the first stage.
        """
        d = self.data
        max_arcs = 200
        max_trials = 40
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
        """
        Try opening a closed DC, migrating flow to it **arc by arc**.

        Previously this move wiped every affected customer's whole
        column (`y[:, ks] = 0`) and dumped `d.d[k]` onto the new DC, so
        opening a DC always destroyed existing splits and produced a
        single-source assignment for every customer it touched. Now each
        existing arc (j, k) is evaluated on its own: a customer served
        by two DCs may migrate only one of its two arcs, and customers
        keep whatever split they had for the arcs that stay put.

        Two trials per candidate DC:
          1. migrate every arc with a negative second-stage delta;
          2. migrate only the 3 least-penalising arcs -- an exploratory
             trial that can open a DC whose payoff is in the first stage
             (relieving an overloaded DC's expensive suppliers).
        """
        d = self.data
        closed = [j for j in range(d.J)
                  if j not in sol.open_dc and self._may_open(j)]
        if not closed:
            return False

        random.shuffle(closed)
        max_candidates = min(len(closed), 36)

        jf, kk = np.nonzero(sol.y)
        if len(jf) == 0:
            return False
        q = sol.y[jf, kk]

        for j_new in closed[:max_candidates]:
            # exact second-stage delta of migrating arc (jf, kk) to j_new
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
        """
        Try closing an open DC (first‑improvement, at most 20 candidates).
        """
        d = self.data
        open_dcs = list(sol.open_dc)
        if len(open_dcs) <= 1:
            return False

        random.shuffle(open_dcs)
        max_candidates = min(len(open_dcs), 20)

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

            if self._accept(sol, trial):
                return True
        return False

    # -----------------------------------------------------------------
    def _swap_dc(self, sol):
        """
        Swap one open DC with one closed DC (first‑improvement, at most
        150 pairs).

        The freed arcs are redistributed the same way `_remove_dc` does
        it -- each customer goes to whichever of {the newly opened DC} u
        {the other open DCs} is cheapest for it. Dumping the whole DC on
        the new one, as before, made this move strictly coarser than
        `_remove_dc` for no reason.
        """
        d = self.data
        open_list = list(sol.open_dc)
        closed = [j for j in range(d.J)
                  if j not in sol.open_dc and self._may_open(j)]
        if not open_list or not closed:
            return False

        # Sample pairs without replacement; when the cap covers the whole
        # product, use the full (shuffled) Cartesian product instead of
        # drawing 150 times with repetition.
        n_pairs = min(150, len(open_list) * len(closed))
        if n_pairs == len(open_list) * len(closed):
            pairs = [(o, c) for o in open_list for c in closed]
            random.shuffle(pairs)
        else:
            pairs = random.sample([(o, c) for o in open_list for c in closed],
                                  n_pairs)

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
    """
    Elite pool with an explicit diversity criterion.

    The old `add()` only rejected an exact duplicate (same DC set *and*
    cost within 50) and otherwise kept the 12 cheapest solutions seen.
    That makes the pool a clone set, which starves path relinking: on
    small_5 the 12 members had a mean pairwise Jaccard distance between
    their DC sets of 0.07, i.e. they were essentially the same solution.

    The update rule now follows the standard elite-set policy: a
    candidate that is too close to an existing member competes only
    against *that* member, so a cheap solution can never crowd out the
    pool's diversity -- it can only replace its own nearest neighbour.
    """

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
        # nearest current member
        near, near_d = None, 2.0
        for p in self.pool:
            dd = self._dist(p, sol)
            if dd < near_d:
                near, near_d = p, dd

        full = len(self.pool) >= self.size
        # While the pool has room, only exact clones are turned away --
        # applying the distance rule from the start starves path
        # relinking (it left the pool at 3-4 of 12 members).
        thr = self.min_dist if full else 0.0

        if near is not None and near_d <= thr:
            # too similar to `near`: only allowed in by beating it
            if sol.cost < near.cost:
                self.pool.remove(near)
            else:
                return
        elif full:
            # far enough from everyone, but the pool is full: displace
            # the most expensive member, and only if we are cheaper
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
    """
    Column-exchange path relinking.

    The old version ignored the flows of both parents entirely: it took
    the union / intersection of their DC *sets* and then rebuilt `y`
    from scratch with `trial.y[best_j, k] = d.d[k]`, i.e. every customer
    single-sourced at its cheapest DC in that set. Whatever split
    structure the two elite solutions had discovered was thrown away,
    and every solution PR ever returned was single-source.

    The new version walks an actual path between the two parents in
    solution space. The key observation: in any feasible solution the
    column `y[:, k]` sums to `d[k]`, so replacing the guest's column for
    one customer with the host's column keeps the second stage feasible
    exactly (only `x` has to be repaired). One customer column is
    exchanged per step -- greedily, the cheapest of a sampled subset --
    so intermediate solutions inherit real split structure from both
    parents instead of being re-derived from a DC set.
    """

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
            # the path is allowed to go uphill; we keep its best point
            if cur.cost < best_cost:
                best, best_cost = cur.copy(), cur.cost
        return best

    def relink(self, a, b, max_steps=15, max_cand=8):
        # relink in both directions -- the two paths are different
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


def perturb(sol, data, max_close=None):
    """
    ILS kick: close a random handful of open DCs and dump their arcs on
    *random* other DCs.

    Deliberately random rather than greedy -- a cost-driven relocation
    is just a local-search move and the VND would undo it immediately.
    The point is to land somewhere else and let the VND re-descend.
    """
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
    """
    Mask confining an ILS re-descent to the seed's own DC configuration,
    plus `n_extra` randomly chosen closed DCs.

    The restart branch already runs its local search under the
    construction's blacklist, for the reason spelled out there: a mask
    that only covers the construction is useless, because `_add_dc`
    immediately re-opens the blacklisted DCs and the restart lands back
    in the same basin. The ILS branch used to skip the mask entirely and
    so had exactly that problem -- every perturbation of an elite
    solution drifted back into the generic basin, and the elite pool's
    DC configuration (the one thing path relinking works to find) was
    never actually intensified.

    Measured on small_1 (1000 iterations, 3 runs): without the mask the
    optimum 150568 is never reached (best 150906, 0/3); with it, 3/3, at
    the same wall-clock. For scale: restricting construction + local
    search to the single best DC subset reaches the optimum roughly once
    per 40-120 restarts, so what was missing was not *finding* the right
    DC set but *re-descending inside it*.

    `n_extra` is the escape hatch. With 0 the DC set can only shrink to a
    subset of the seed's under ILS, and new configurations have to come
    from the restart branch, path relinking or shake -- fine on small_1,
    where the restart branch finds the best subset on its own, but on the
    larger instances (J = 40-320) it cannot, so the pool would hold
    near-miss configurations with no way for ILS to walk out of them.
    One extra DC makes each descent a 1-neighbourhood step in DC-set
    space, at ~10 % more time per iteration.

    Note the mask only blocks *opening a currently closed* DC (see
    `_may_open` / the `shut` term in `_shift_flow`). `perturb` runs
    before the local search and can dump flow on DCs outside the mask,
    so the trajectory does leave the elite configuration -- it just
    cannot return to those DCs once the search closes them. Perturb
    pushes out, the mask ratchets back.
    """
    mask = np.zeros(J, dtype=bool)
    mask[list(sol.open_dc)] = True
    closed = np.flatnonzero(~mask)
    if n_extra and len(closed):
        for j in random.sample(list(closed), min(n_extra, len(closed))):
            mask[j] = True
    return mask


def _construct_and_improve(task):
    """
    One parallel iteration. `task` is (alpha, seed_y):

      * seed_y is None -> ordinary GRASP restart (construction + VND),
        under a random DC blacklist.
      * seed_y is an elite solution's y -> ILS iteration: perturb it and
        re-descend under `elite_dc_mask`, i.e. confined to the seed's own
        DC configuration.

    The second mode is the only intensification the algorithm has:
    without it every worker starts from scratch and the diverse elite
    pool is consumed by nothing but path relinking every 8 iterations.
    """
    alpha, seed_y = task
    # Reset this worker's own counter so each task reports only the
    # evals it personally performed (Pool reuses worker processes
    # across many tasks, so the counter would otherwise keep growing).
    _reset_eval_count()
    d = _WORKER_DATA

    if seed_y is not None:
        sol = Solution(d)
        sol.y = seed_y.copy()
        sol.compute_cost()
        # drawn before `perturb`, so it describes the elite configuration
        # we mean to intensify, not the kicked one
        mask = elite_dc_mask(sol, d.J)
        perturb(sol, d)
        sol = LocalSearch(d, mask).improve(sol)
        sol.compute_cost()
        return sol.x, sol.y, sol.cost, sol.open_dc, _get_eval_count()

    # The DC blacklist is drawn here, in the worker, so each worker's own
    # RNG diversifies it independently instead of all of them sharing one
    # mask shipped from the main process.
    # The mask must stay active through local search too -- restricting
    # only the construction is useless, `_add_dc` just re-opens the
    # blacklisted DCs and the restart lands in the same basin.
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
        # fraction of iterations that perturb an elite solution (ILS)
        # instead of building a fresh one
        self.p_seed = p_seed
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

            if elite.pool and random.random() < self.p_seed:
                # ILS iteration: perturb an elite solution and re-descend,
                # confined to its DC configuration (see `elite_dc_mask`)
                sol = Solution(self.data)
                sol.y = random.choice(elite.pool).y.copy()
                sol.compute_cost()
                mask = elite_dc_mask(sol, self.data.J)
                perturb(sol, self.data)
                sol = LocalSearch(self.data, mask).improve(sol)
            else:
                mask = random_dc_mask(self.data.J)
                sol = Construction(self.data, alpha, mask).build()
                # masked LS for this restart; the shared unmasked `ls`
                # stays for path relinking and shake
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
                    # PR output used to be discarded unless it beat the
                    # incumbent; it is a fully improved solution and
                    # belongs in the pool either way.
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
                tasks = []
                for offset in range(batch):
                    if (it + offset) - last_improvement > 15:
                        alpha = random.uniform(0.4, 0.7)
                    else:
                        alpha = random.uniform(0.05, 0.5)
                    # Half the batch intensifies around the elite pool
                    # (ILS), half keeps generating fresh restarts.
                    seed = None
                    if elite.pool and random.random() < self.p_seed:
                        seed = random.choice(elite.pool).y
                    tasks.append((alpha, seed))

                results = pool.map(_construct_and_improve, tasks)

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
                        if cand:
                            elite.add(cand)
                            if cand.cost < best_cost:
                                best = cand.copy()
                                best_cost = cand.cost
                                last_improvement = it
                                time_to_best = time.time() - start
                                print(f'[{it}] PR = {best_cost}')

                    # shake (ruin-and-recreate) when stuck for too long
                    if it - last_improvement > 20 and best:
                        best, best_cost, last_improvement, time_to_best = self._shake(
                            best, best_cost, it, ls, start, time_to_best, elite)

                    it += 1
        finally:
            pool.close()
            pool.join()

        total_time = time.time() - start
        return best, best_cost, total_time, time_to_best, _get_eval_count()

    # ----------------------------------------------------------
    # Ruin-and-recreate shake, shared by serial and parallel runs.
    # ----------------------------------------------------------
    def _shake(self, best, best_cost, it, ls, start, time_to_best, elite=None):
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
                    # the shaken solution is fully improved -- keep it in
                    # the pool even when it does not beat the incumbent
                    if elite is not None:
                        elite.add(trial)
                    if trial.cost < best_cost:
                        best = trial.copy()
                        best_cost = trial.cost
                        time_to_best = time.time() - start
                        print(f'[{it}] shake = {best_cost}')
        return best, best_cost, last_improvement, time_to_best
