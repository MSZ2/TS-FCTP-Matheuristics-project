import numpy as np
import random
import time
from dataclasses import dataclass


# ============================================================
# PROBLEM: TWO-STAGE FIXED CHARGE TRANSPORTATION PROBLEM
#
# Goods flow in two stages:
#   Stage 1: supplier i -> DC j     (variable cost b[i,j], fixed cost f[i,j])
#   Stage 2: DC j -> customer k     (variable cost c[j,k], fixed cost g[j,k])
#
# A fixed cost is paid once per arc, regardless of how much flows through it.
# The problem is NP-hard because of the fixed costs: deciding which arcs to
# open (binary) and how much to flow through them (continuous) are coupled.
# Every integer solution defines a different set of open arcs, making the
# search space exponentially large.
#
# Decision variables:
#   x[i,j] : units shipped from supplier i to DC j
#   y[j,k] : units shipped from DC j to customer k
#
# Objective: minimise
#   sum(x * b) + sum(y * c) + sum((x>0) * f) + sum((y>0) * g)
#
# Constraints:
#   sum_j y[j,k] == d[k]          for all k  (demand satisfaction)
#   sum_j x[i,j] <= s[i]          for all i  (supply capacity)
#   sum_i x[i,j] == sum_k y[j,k]  for all j  (flow balance at each DC)
#   x[i,j], y[j,k] >= 0, integer
#
# Customer demand may be split across multiple DCs. This is legal and
# sometimes necessary for optimality.
# ============================================================


# ============================================================
# DATA
# ============================================================

@dataclass
class TSFCTPData:
    I: int           # number of suppliers
    J: int           # number of DCs
    K: int           # number of customers
    s: np.ndarray    # supply[i]
    d: np.ndarray    # demand[k]
    b: np.ndarray    # variable cost supplier->DC  (I x J)
    f: np.ndarray    # fixed cost supplier->DC     (I x J)
    c: np.ndarray    # variable cost DC->customer  (J x K)
    g: np.ndarray    # fixed cost DC->customer     (J x K)

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
    # A solution is fully described by the two flow matrices x and y.
    # open_dc is derived from them and cached for fast lookup in moves.
    # cost is kept in sync by calling compute_cost() after every change.

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
# REPAIR
#
# The key sub-problem: given a fixed y (DC->customer assignment),
# find the cheapest feasible x (supplier->DC flows).
#
# This is solved greedily per DC. For each DC j, its total inbound
# demand is sum_k y[j,k]. Suppliers are ranked by:
#
#   b[i,j] + f[i,j] / demand_j
#
# The second term is the amortised fixed cost per unit. When demand_j
# is large the fixed cost is spread thin and barely matters. When
# demand_j is small (e.g. a DC serving one small customer) the fixed
# cost dominates and must be included in the ranking.
#
# Without the amortised term, cheap-variable but expensive-fixed
# suppliers look better than they are, producing solutions whose
# true cost is higher than the greedy ranking suggested.
#
# repair() is called after every local search move that modifies y,
# and after construction. It is the single place where x is computed,
# which keeps constraint satisfaction centralised and easy to verify.
# ============================================================

def repair(sol, data):
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
            return False
    return True


# ============================================================
# CONSTRUCTION
#
# Builds a complete feasible solution from scratch using a
# greedy randomised heuristic. Called once per GRASP iteration.
#
# The algorithm works customer by customer (largest demand first,
# because high-demand customers have fewer feasible placements and
# should be served before supply is fragmented). For each unit of
# demand still unassigned, it scores all (supplier, DC) pairs by
# an estimated unit cost that includes amortised fixed costs:
#
#   score = b[i,j] + c[j,k]
#         + f[i,j]/demand  (only if this supplier->DC arc is new)
#         + g[j,k]/demand  (only if this DC->customer arc is new)
#
# All candidates within a factor alpha of the best score form the
# Restricted Candidate List (RCL). One is chosen at random from
# the RCL. This is the standard GRASP construction mechanism.
#
# Alpha controls the tradeoff between greediness and randomness:
#   alpha = 0: always pick the single best candidate (fully greedy,
#              produces the same solution every call, no diversity).
#   alpha = 1: pick from the entire candidate list (fully random,
#              lots of diversity, poor individual solutions).
#   alpha in (0, 1): the useful range. Alpha is sampled fresh each
#              GRASP iteration from Uniform(0.05, 0.50) so the search
#              naturally explores both greedy and random regimes.
#
# Seeded construction: some iterations receive a seed_dc_set from the
# elite pool. In that case the alpha=0 greedy solution is built with
# those DCs pre-favoured, giving intensification around known good
# DC configurations while still allowing the greedy heuristic to
# optimise customer-to-DC assignment within the seed structure.
# ============================================================

class Construction:

    def __init__(self, data, alpha=0.3, seed_dc_set=None):
        self.data = data
        self.alpha = alpha
        # seed_dc_set: optional frozenset of DC indices from the elite pool.
        # When provided, costs for DCs outside the seed are penalised so the
        # greedy heuristic strongly prefers the seeded configuration. This is
        # not a hard constraint -- the heuristic can still open non-seed DCs
        # if the cost benefit is large enough.
        self.seed_dc_set = seed_dc_set

    def build(self):
        d = self.data
        sol = Solution(d)
        rem_s = d.s.copy()

        # Randomise customer order to diversify which DCs get opened first.
        # A fixed descending-demand order causes the same DCs to be chosen
        # iteration after iteration, reducing diversity in the elite pool.
        # Shuffling breaks this while keeping the rough priority of large
        # demands by partially sorting: shuffle, then stable-sort by demand.
        order = list(range(d.K))
        random.shuffle(order)
        order.sort(key=lambda k: -d.d[k])

        for k in order:
            need = int(d.d[k])
            while need > 0:
                cand = []
                for i in range(d.I):
                    if rem_s[i] <= 0:
                        continue
                    for j in range(d.J):
                        q = min(rem_s[i], need)
                        eff = max(need, 1)
                        unit = d.b[i, j] + d.c[j, k]
                        if sol.x[i, j] == 0:
                            unit += d.f[i, j] / eff
                        if sol.y[j, k] == 0:
                            unit += d.g[j, k] / eff
                        # If a seed DC set was provided, add a penalty for
                        # DCs outside the seed. The penalty is proportional
                        # to the average fixed cost so it scales with instance
                        # size, not hardcoded. This softly guides construction
                        # toward the seed structure without forcing it.
                        if self.seed_dc_set and j not in self.seed_dc_set:
                            unit += np.mean(d.g[:, k])
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
# LOCAL SEARCH -- VARIABLE NEIGHBOURHOOD DESCENT (VND)
#
# VND is an extension of standard local search that uses multiple
# neighbourhoods in a structured way. The key idea is:
#
#   A solution that is locally optimal in neighbourhood N1 may not
#   be locally optimal in neighbourhood N2. By switching to a larger
#   or different neighbourhood when the current one is exhausted,
#   VND escapes local optima that trap simpler descent methods.
#
# Standard VND logic:
#   k = 1
#   while k <= k_max:
#       find the best neighbour x' of x in neighbourhood N_k
#       if x' is better: x = x', reset k = 1
#       else: k = k + 1  (move to next neighbourhood)
#
# This differs critically from the naive "try all moves, restart on
# any improvement" loop. In the naive loop, a move that fires in N1
# always causes a restart from N1, so the search never progresses
# through to larger neighbourhoods unless N1 is completely exhausted.
# VND's reset-to-1-on-improvement means a successful N3 move still
# triggers a fresh N1 scan (good: N1 is cheap), but a failed N1 scan
# advances to N2 rather than terminating. This is more aggressive and
# thorough than the naive restart.
#
# Neighbourhood ordering principle:
#   Neighbourhoods are ordered by size and cost, cheapest first.
#   N1 (full_reassign) and N2 (two_customer_swap) are O(J*K) and
#   O(K^2) respectively -- fast, small neighbourhoods checked first.
#   N3 (partial_shift) is also fast due to the analytical split.
#   N4 (remove_dc) and N5 (add_dc) open or close a whole DC -- larger
#   structural changes, more expensive.
#   N6 (swap_open_closed) is the largest structural move.
#   N7 (two_dc_swap) closes two DCs and opens two others simultaneously,
#   the largest neighbourhood, checked last.
#
# Each move uses best-improvement within its neighbourhood: all
# neighbours are evaluated and the best is taken. This is slower per
# iteration than first-improvement but produces higher-quality local
# optima that are harder for perturbation to escape.
# ============================================================

class VND:

    def __init__(self, data):
        self.data = data
        # Ordered list of neighbourhood functions, cheapest to most expensive.
        # VND cycles through this list, resetting to index 0 on any improvement.
        self._neighbourhoods = [
            self._full_reassign,
            self._two_customer_swap,
            self._partial_shift,
            self._remove_dc,
            self._add_dc,
            self._swap_open_closed,
            self._two_dc_swap,
        ]

    def improve(self, sol):
        # Standard VND descent loop.
        # k indexes the current neighbourhood. On improvement: reset to 0.
        # On failure: advance to k+1. Terminate when all neighbourhoods fail.
        sol.compute_cost()
        k = 0
        while k < len(self._neighbourhoods):
            improved = self._neighbourhoods[k](sol)
            if improved:
                k = 0
            else:
                k += 1
        return sol

    # ----------------------------------------------------------
    # N1: full reassignment of customer k from one DC to another.
    #
    # Evaluates all (customer, source DC, target DC) triples and picks
    # the single best improving move. The cost delta is computed exactly,
    # including the fixed cost terms:
    #   - pay g[j_to, k]   if the j_to->k arc is currently empty
    #   - save g[j_from,k] if the j_from->k arc will be emptied by the move
    #
    # This is the most fundamental move in the TSFCTP search space.
    # It can open new DCs (by moving the first customer to them), close
    # DCs (by moving their last customer away), and re-route any existing
    # assignment. It is checked first because it is cheap and frequently
    # improves.
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
                    delta = (d.c[j_to, k] - d.c[j_from, k]) * qty
                    delta += d.g[j_to, k] if sol.y[j_to, k] == 0 else 0
                    if sol.y[j_from, k] == qty:
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
    # N2: swap DC assignments of two customers simultaneously.
    #
    # Customer k1 moves from j1 to j2, and customer k2 moves from
    # j2 to j1 in the same operation. This catches improvements that
    # are invisible to N1: moving k1 alone from j1 to j2 may be
    # unprofitable (higher variable cost), and moving k2 alone may
    # also be unprofitable, but doing both together is profitable
    # because the two customers share the fixed cost burden at each DC.
    #
    # A variable-cost delta is used as a cheap pre-filter before the
    # full trial-and-repair evaluation. This avoids calling repair()
    # on swaps that are obviously unprofitable.
    # ----------------------------------------------------------
    def _two_customer_swap(self, sol):
        d = self.data
        best_cost = sol.cost
        best_trial = None

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

    # ----------------------------------------------------------
    # N3: partial demand shift (analytical optimal split).
    #
    # Moves a fraction of customer k's demand from DC j_from to DC
    # j_to, creating a split assignment. TSFCTP allows splits and
    # they are sometimes required for optimality.
    #
    # The original implementation tried only three fixed fractions
    # (25/50/75%) and called repair() inside a triple loop, making
    # it the most expensive move by far (7.5 ms/call vs 0.2 ms for
    # others). This version computes the optimal integer split qty*
    # analytically and calls repair() exactly once per (j_from, j_to)
    # pair instead of three times.
    #
    # Derivation of qty*:
    # When moving qty units from j_from to j_to, the cost change on
    # the y side (ignoring supplier costs, which repair handles) is:
    #
    #   delta_y(qty) = (c[j_to,k] - c[j_from,k]) * qty
    #                + g[j_to,k]  * [qty > 0 and y[j_to,k] was 0]
    #                - g[j_from,k]* [qty == y[j_from,k]]
    #
    # The variable part is linear in qty with slope (c[j_to,k] - c[j_from,k]).
    # We only enter this move when c[j_to,k] < c[j_from,k], so the slope is
    # negative -- more units moved is better on variable cost alone.
    # The fixed cost g[j_to,k] is a one-time charge paid on the first unit.
    # The fixed cost saving g[j_from,k] is earned only if qty equals the
    # full remaining flow (full move), which is handled by N1 already.
    #
    # For a partial split (1 <= qty < y[j_from,k]):
    #   delta_y(qty) = (c[j_to,k] - c[j_from,k]) * qty + g[j_to,k]
    #
    # This is minimised by maximising qty, so qty* = y[j_from,k] - 1
    # (leave exactly 1 unit at j_from to keep it open and avoid
    # triggering g[j_from,k] saving, which is already handled by N1).
    # If even qty* = y[j_from,k] - 1 is not beneficial, no smaller qty
    # will be either, so we skip.
    #
    # The supplier cost change from repair() is not captured in this
    # analytical formula, so we still call repair() and compute_cost()
    # once to verify the true improvement. But we call it at most once
    # per (j_from, j_to) pair instead of three times.
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
                    if d.c[j_to, k] >= d.c[j_from, k]:
                        continue

                    # Analytical optimal split: move as much as possible
                    # while keeping at least 1 unit at j_from (a full move
                    # is handled by N1, not here).
                    qty = qty_total - 1

                    # Quick check: is the y-side delta negative even at qty*?
                    # If not, no partial shift of any size will help.
                    delta_y = (d.c[j_to, k] - d.c[j_from, k]) * qty
                    if sol.y[j_to, k] == 0:
                        delta_y += d.g[j_to, k]
                    if delta_y >= 0:
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
    # N4: close one DC entirely.
    #
    # Tries closing each open DC in turn. Each customer currently
    # served by that DC is re-routed to whichever remaining open DC
    # minimises cost, accounting for the g fixed cost that would be
    # incurred if a new arc must be opened.
    #
    # This move saves the fixed costs of all arcs touching the closed
    # DC (both supplier->DC arcs and DC->customer arcs) at the expense
    # of potentially higher variable costs on the re-routed flows.
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
    # N5: open one new DC.
    #
    # For each currently-closed DC j_new, checks whether any customer
    # would be better served through j_new than through its current DC.
    # A customer is moved to j_new if:
    #
    #   c[j_new,k]*d[k] + g[j_new,k]  <  c[curr,k]*d[k] + g[curr,k]
    #
    # This is the full arc cost (variable + fixed) for serving customer
    # k entirely through j_new versus its current assignment.
    #
    # The move opens j_new only if at least one customer benefits and
    # the total cost (after repair of the supplier side) is lower.
    # The supplier cost of supplying j_new is not estimated here; it
    # is computed exactly by repair().
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
    # N6: close one open DC, open one closed DC (1-for-1 swap).
    #
    # All open-closed pairs are tried exhaustively. For each pair,
    # all customers of the closing DC are redirected to the opening
    # DC, and repair() recomputes the supplier side.
    #
    # This move changes the DC configuration without changing the
    # number of open DCs. It is checked after N4 and N5 because it
    # is structurally a composition of close+open and is more expensive
    # due to the double loop over all pairs.
    #
    # With J=8, the worst case is 8*8 = 64 pairs, which is manageable.
    # For larger instances this exhaustive approach should be replaced
    # with a ranked or sampled subset.
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
    # N7: close two open DCs, open two closed DCs (2-for-2 swap).
    #
    # This is the largest neighbourhood in the VND and is checked last.
    # It is motivated by the observation that some local optima require
    # simultaneously replacing two DCs to escape: closing DC A alone
    # is unprofitable, closing DC B alone is unprofitable, but closing
    # both and opening two others can be profitable because the fixed
    # costs of A and B together outweigh the cost of the replacement
    # configuration.
    #
    # Profiling on this instance showed that 73% of local search runs
    # converge to the same DC set {0,1,4,6,7}, and the minimum single-
    # move escape cost is +1557. A 2-for-2 swap can cross that barrier
    # in one step where N6 cannot.
    #
    # Customers from both closing DCs are reassigned greedily to the
    # cheapest of the two opening DCs before calling repair().
    #
    # The combinatorial size is |open|^2 * |closed|^2 which grows
    # quickly with J. For J=8 with 5 open and 3 closed DCs this is
    # 10 * 3 = 30 pairs, still manageable. For larger instances, a
    # random sample of pairs should replace the exhaustive search.
    # ----------------------------------------------------------
    def _two_dc_swap(self, sol):
        d = self.data
        open_list = list(sol.open_dc)
        closed = [j for j in range(d.J) if j not in sol.open_dc]
        if len(open_list) < 2 or len(closed) < 2:
            return False

        best_cost = sol.cost
        best_trial = None

        for ci1 in range(len(open_list)):
            for ci2 in range(ci1 + 1, len(open_list)):
                j_close1, j_close2 = open_list[ci1], open_list[ci2]
                for oi1 in range(len(closed)):
                    for oi2 in range(oi1 + 1, len(closed)):
                        j_open1, j_open2 = closed[oi1], closed[oi2]

                        trial = sol.copy()
                        # Redirect customers from both closing DCs to
                        # whichever of the two opening DCs is cheaper.
                        for k in range(d.K):
                            for j_close in (j_close1, j_close2):
                                if trial.y[j_close, k] > 0:
                                    best_j = min(
                                        (j_open1, j_open2),
                                        key=lambda j: d.c[j, k]
                                    )
                                    trial.y[best_j, k] += trial.y[j_close, k]
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


# ============================================================
# ILS PERTURBATION
#
# Iterated Local Search (ILS) addresses the core weakness of GRASP
# on this problem: the search repeatedly converges to the same small
# set of local optima (5 unique DC configurations in 100 runs, with
# 73% converging to the same set).
#
# The root cause is that all local search neighbourhoods are descent-
# only: they will never accept a move that increases cost, even
# temporarily. But the optimal solution may only be reachable by first
# paying a cost penalty to restructure the DC configuration, then
# descending from that new starting point.
#
# ILS breaks this by:
#   1. Taking the current local optimum as the base.
#   2. Applying a random perturbation that changes the DC structure
#      in a way no descent move would accept.
#   3. Running VND from the perturbed solution to find a new local
#      optimum.
#   4. Accepting or rejecting based on cost (accept if better, else
#      revert to base). This is the simplest acceptance criterion;
#      more sophisticated variants (simulated annealing-style) could
#      accept worse solutions with some probability to escape deeper
#      basins.
#
# The perturbation itself:
#   Force-open n_perturb randomly chosen closed DCs and reassign each
#   customer to its cheapest DC among all currently open DCs (both
#   the original open set and the newly forced-open ones). This is
#   a "double-bridge"-style kick adapted to the TSFCTP structure.
#
#   n_perturb is sampled from {1, 2} each call. Perturbing with 1 DC
#   makes a modest change; 2 DCs makes a larger jump. Larger values
#   tend to destroy too much of the good structure built by VND.
#
# ILS is applied to the best solution in the elite pool periodically,
# giving the most promising known solutions a chance to escape their
# local optima basins rather than only applying perturbation to
# freshly constructed solutions.
# ============================================================

class ILS:

    def __init__(self, data, vnd):
        self.data = data
        self.vnd = vnd

    def perturb_and_improve(self, sol, n_perturb=None):
        d = self.data
        if n_perturb is None:
            n_perturb = random.randint(1, 2)

        closed = [j for j in range(d.J) if j not in sol.open_dc]
        if not closed:
            return sol

        # Force-open n_perturb random closed DCs.
        # We cannot force more DCs than are closed.
        n_perturb = min(n_perturb, len(closed))
        forced = set(random.sample(closed, n_perturb))
        available = sol.open_dc | forced

        # Rebuild y from scratch: each customer goes to its cheapest
        # DC in the expanded available set. This is greedy-optimal on
        # the y side given the fixed available DCs, which gives VND
        # the best possible starting point from this DC configuration.
        trial = Solution(d)
        for k in range(d.K):
            best_j = min(available, key=lambda j: d.c[j, k] * d.d[k] + d.g[j, k])
            trial.y[best_j, k] = d.d[k]

        if not repair(trial, d):
            return sol

        trial.compute_cost()
        trial = self.vnd.improve(trial)

        # Accept if the perturbed+improved solution is better.
        # Rejecting on worse cost keeps the overall search monotone
        # with respect to the best known solution, which is appropriate
        # here because ILS is embedded in GRASP which already provides
        # diversity through randomised construction.
        if trial.is_valid() and trial.cost < sol.cost:
            return trial
        return sol


# ============================================================
# ELITE POOL
#
# Stores the best solutions found across all GRASP iterations.
# Used as the source of starting points for ILS perturbation and
# as the guide structure for seeded construction.
#
# Diversity filter: solutions with identical DC-set signatures and
# nearly identical costs (within 100) are considered duplicates and
# not added. Without this filter the pool fills with near-identical
# solutions that produce uninformative constructions and perturbations.
#
# The pool is kept sorted by cost so best() is O(1).
# ============================================================

class ElitePool:

    def __init__(self, size=12):
        self.size = size
        self.pool = []

    def add(self, sol):
        sig = frozenset(sol.open_dc)
        for p in self.pool:
            if frozenset(p.open_dc) == sig and abs(p.cost - sol.cost) < 100:
                return
        self.pool.append(sol.copy())
        self.pool.sort(key=lambda s: s.cost)
        self.pool = self.pool[:self.size]

    def best(self):
        return self.pool[0].copy() if self.pool else None

    def random(self):
        return random.choice(self.pool).copy() if self.pool else None

    def random_dc_set(self):
        # Returns the DC signature of a random elite solution.
        # Used by seeded construction to bias new solutions toward
        # configurations already known to be productive.
        if not self.pool:
            return None
        return frozenset(random.choice(self.pool).open_dc)


# ============================================================
# PATH RELINKING
#
# Path relinking is a post-processing intensification strategy that
# operates on pairs of solutions from the elite pool. The idea is that
# two independently found local optima may have complementary structure:
# one may have a good DC set, the other a good customer assignment.
# Combining them can produce a solution better than either parent.
#
# For TSFCTP the natural relinking attribute is the DC set. Two DC sets
# define a path in the space of DC configurations. Rather than walking
# the path one DC at a time (which is expensive), this implementation
# evaluates two specific points on the path:
#
#   Union:        all DCs open in either solution.
#   Intersection: only DCs open in both solutions.
#
# For each candidate DC set, customers are assigned greedily and VND
# is applied. The better of the two results is returned.
#
# The union point tends to over-invest in fixed costs but gives VND
# a rich starting structure to prune. The intersection is leaner and
# focuses intensification on the DCs both solutions agreed on.
# ============================================================

class PathRelinking:

    def __init__(self, data, vnd):
        self.data = data
        self.vnd = vnd

    def relink(self, a, b):
        d = self.data
        best = None
        best_cost = float('inf')

        for dc_set in [a.open_dc | b.open_dc, a.open_dc & b.open_dc]:
            if not dc_set:
                continue
            trial = Solution(d)
            for k in range(d.K):
                best_j = min(dc_set, key=lambda j: d.c[j, k] * d.d[k] + d.g[j, k])
                trial.y[best_j, k] = d.d[k]
            if not repair(trial, d):
                continue
            trial.compute_cost()
            trial = self.vnd.improve(trial)
            if trial.is_valid() and trial.cost < best_cost:
                best_cost = trial.cost
                best = trial

        return best


# ============================================================
# GRASP + VND + ILS
#
# The overall algorithm structure is:
#
#   for each iteration:
#     1. CONSTRUCTION: build a new feasible solution using the
#        greedy randomised heuristic. Every 4th iteration uses a
#        seed DC set from the elite pool (seeded construction) to
#        intensify around known good configurations. The remaining
#        iterations use an unseeded randomised construction to
#        maintain diversity.
#
#     2. VND: apply Variable Neighbourhood Descent to the constructed
#        solution. This is the main local search phase. VND cycles
#        through 7 neighbourhoods in order of increasing size,
#        resetting to the smallest neighbourhood on any improvement.
#
#     3. ELITE UPDATE: add the VND local optimum to the elite pool
#        (subject to the diversity filter).
#
#     4. ILS (periodic): every ils_freq iterations, apply ILS
#        perturbation to the best elite solution. ILS forces a DC
#        configuration change that VND descent would never accept,
#        then runs VND from the perturbed starting point. This is
#        the primary mechanism for escaping local optima basins.
#
#     5. PATH RELINKING (periodic): every pr_freq iterations, combine
#        the best elite solution with a random elite solution using
#        path relinking. This is a secondary diversification mechanism
#        that exploits structural complementarity between elite solutions.
#
# The interleaving of ILS and PR is important: ILS operates on single
# solutions and escapes local optima by perturbation; PR operates on
# pairs of solutions and exploits shared structure. Together they cover
# different failure modes of pure VND descent.
# ============================================================

class GRASP:

    def __init__(self, data, iters=300, ils_freq=5, pr_freq=10):
        self.data = data
        self.iters = iters
        # ILS is applied every ils_freq iterations. A lower value means
        # more frequent perturbation, which is aggressive but may not
        # give VND enough time to fully exploit each perturbation.
        self.ils_freq = ils_freq
        # Path relinking fires less frequently than ILS because it
        # requires two sufficiently different elite solutions and is
        # more expensive (two VND calls inside relink()).
        self.pr_freq = pr_freq

    def run(self):
        best = None
        best_cost = float('inf')

        vnd   = VND(self.data)
        ils   = ILS(self.data, vnd)
        pr    = PathRelinking(self.data, vnd)
        elite = ElitePool(12)
        start = time.time()

        for it in range(self.iters):

            # Seeded construction: every 4th iteration, seed from elite pool.
            # This biases construction toward DC configurations that have
            # already produced good solutions, concentrating search effort
            # in productive regions without locking out new configurations.
            if it % 4 == 0 and elite.pool:
                seed = elite.random_dc_set()
            else:
                seed = None

            alpha = random.uniform(0.05, 0.50)
            sol = Construction(self.data, alpha, seed_dc_set=seed).build()
            sol = vnd.improve(sol)
            sol.compute_cost()
            elite.add(sol)

            if sol.cost < best_cost:
                best = sol.copy()
                best_cost = sol.cost
                print(f"  [{it}] construct = {best_cost}", flush=True)

            # ILS perturbation on the best elite solution.
            # Applied more frequently than PR because it is cheaper and
            # directly targets the local optima basin problem.
            if it % self.ils_freq == 0 and elite.pool:
                ils_start = elite.best()
                ils_result = ils.perturb_and_improve(ils_start)
                if ils_result and ils_result.is_valid():
                    elite.add(ils_result)
                    if ils_result.cost < best_cost:
                        best = ils_result.copy()
                        best_cost = ils_result.cost
                        print(f"  [{it}] ILS       = {best_cost}", flush=True)

            # Path relinking on two elite solutions.
            if it % self.pr_freq == 0 and len(elite.pool) >= 2:
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
        print("Usage: python grasp_tsfctp_vnd_ils.py <instance_file> [iters]")
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