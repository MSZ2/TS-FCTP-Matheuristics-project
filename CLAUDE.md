# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing matheuristics for the **Two-Stage Fixed-Charge Transportation Problem (TS-FCTP)**. The problem models a supply chain with three tiers: Suppliers → Distribution Centers (DCs) → Customers, with both variable and fixed costs on each arc.

Two solver approaches live in separate directories:
- **`CPLEX_part/`** — exact MIP solver using IBM ILOG CPLEX (C++)
- **`GRASP_part_parallel/`** — metaheuristic solver using GRASP with path relinking (Python), vectorized (numpy) and parallelized (`multiprocessing`) across independent iterations

## Running the Code

### GRASP (Python)

Activate the venv first:
```bash
source venv/bin/activate
```

Single run on one instance:
```bash
cd GRASP_part_parallel
python main.py ../generated_instances/small_1.txt --mode single --iters 100
```

Multi-run experiment (saves report to `reports/`):
```bash
python main.py ../generated_instances/small_1.txt --mode multi --iters 100 --runs 10
```

Control the number of parallel workers (defaults to `cpu_count() - 1`; `--workers 1` forces a purely serial run):
```bash
python main.py ../generated_instances/large_1.txt --mode single --iters 50 --workers 8
```

Batch run across all medium and large instances:
```bash
cd GRASP_part_parallel
bash main_runner.sh
```

### CPLEX (C++)

Compile (requires IBM CPLEX installed):
```bash
cd CPLEX_part
g++ -O2 -o solve_tsfctp solve_tsfctp.cpp -I/path/to/cplex/include -L/path/to/cplex/lib -lcplex -lm -lpthread -ldl
```

Run:
```bash
./solve_tsfctp ../generated_instances/small_1.txt
```

Generate instances:
```bash
g++ -O2 -o generator generator.cpp
./generator   # writes to ../generated_instances/
```

## Architecture

### Problem Data (`GRASP_part_parallel/utils.py`)

`TSFCTPData` is a dataclass holding the full instance: dimensions `(I, J, K)`, supply `s[I]`, demand `d[K]`, and four cost matrices — variable `b[I,J]`/`c[J,K]` and fixed `f[I,J]`/`g[J,K]`. `read_instance()` parses the flat-text instance format.

### Solution Representation

`Solution` stores two flow matrices: `x[I,J]` (supplier→DC) and `y[J,K]` (DC→customer), plus `open_dc` (set of active DCs). Validity requires demand satisfaction, supply limits, and DC flow balance. `compute_cost()` sums variable flows × unit costs plus fixed costs for nonzero arcs.

### GRASP Algorithm (`GRASP_part_parallel/GRASP.py`)

`GRASP.run()` picks `_run_serial()` or `_run_parallel()` based on `n_workers` (`n_workers=1` is bit-for-bit the original serial loop; otherwise construction+local search for a batch of `n_workers` iterations runs in a `multiprocessing.Pool`, while elite pool / best-tracking / path relinking / shake stay sequential in the main process, applied after each batch in the same order as the serial loop).

The algorithm structure per iteration (construction + local search vectorized with numpy; see `IMPROVEMENT.md` for the optimization history):
1. **Construction** (`Construction.build`) — greedy-randomized: assigns demand largest-first, builds a Restricted Candidate List (RCL) with `alpha` controlling randomness, uses reactive alpha (higher when stuck).
2. **Local search** (`LocalSearch.improve`) — VND with four neighborhoods: reassign customer to different DC, add/remove a DC, swap open↔closed DCs. All use first-improvement with randomized candidate ordering.
3. **Repair** (`repair`) — after any y-flow change, recomputes x-flows greedily to minimize supplier cost.
4. **Elite pool** (`ElitePool`) — keeps top-12 diverse solutions by DC configuration signature.
5. **Path relinking** (`PathRelinking.relink`) — every 8 iterations, combines two elite solutions by exploring the union and intersection of their DC sets.
6. **Shake** — when stuck >20 iterations, removes the costliest 50% of DCs and reconstructs.

Note on determinism: only Python's `random` module is seeded (never `np.random`), so serial runs (`n_workers=1`) are bit-for-bit reproducible under a given seed; parallel runs are not (each worker process reseeds from `os.getpid() ^ time`), by design, to diversify the search per worker.

### Experiment Infrastructure (`GRASP_part_parallel/utils.py`)

`GRASPExperiment` handles multi-run experiments: deterministic seeds (`42 + run*100`), gap calculation against CPLEX optimal (loaded from `CPLEX_part/results/`), and report generation to `reports/GRASP_report_{instance}.txt`.

### Instance Format

Plain text, space-separated integers in order: `I J K`, then `s[I]`, `d[K]`, `b[I][J]`, `f[I][J]`, `c[J][K]`, `g[J][K]`. Fixed supply is always `I × 80`; demand is balanced to equal total supply.

### Instance Sizes

- **small**: I=4–18, J=8–36, K=16–80 (20 instances)
- **medium**: I=20–60, J=40–120, K=80–240 (20 instances)
- **large**: I=70–160, J=140–320, K=280–800 (20 instances)

CPLEX results (optimal or best-feasible within 2-hour limit) are in `CPLEX_part/results/results_{size}.txt`.

## Key Design Decisions

- `x`-flows are never directly manipulated during search — only `y` (DC→customer) flows are changed, then `repair()` recomputes `x` optimally given the new `y`. This keeps the search space smaller.
- The venv at `venv/` has only `numpy` installed; no other Python dependencies.
