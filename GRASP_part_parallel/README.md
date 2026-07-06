# GRASP_part_parallel

Vektorizovana i paralelizovana verzija GRASP solvera za **Two-Stage Fixed-Charge
Transportation Problem (TS-FCTP)**. Funkcionalno je API-kompatibilna sa
`GRASP_part/GRASP.py`, ali je construction faza vektorizovana preko numpy-ja,
local search je vektorizovan, i pojedinačan run se izvršava preko batch-eva
paralelnih iteracija (`multiprocessing`). Detalji i benchmark rezultati su u
[IMPROVEMENT.md](IMPROVEMENT.md).

## Sadržaj foldera

| Fajl | Uloga |
|---|---|
| `GRASP.py` | Ceo GRASP algoritam: `Solution`, `repair`, `Construction`, `LocalSearch`, `ElitePool`, `PathRelinking`, `GRASP` (serijski + paralelni run). |
| `utils.py` | Parsiranje instanci (`TSFCTPData`, `read_instance`) i infrastruktura za eksperimente (`GRASPExperiment`). Nepromenjen u odnosu na `GRASP_part`. |
| `main.py` | CLI ulazna tačka — pokreće single ili multi-run eksperiment. |
| `main_runner.sh` | Batch skripta koja pokreće multi-run eksperiment nad svim medium/large instancama. |

## Pokretanje

Aktiviraj venv iz root-a projekta:

```bash
source venv/bin/activate
```

Pojedinačan run:

```bash
cd GRASP_part_parallel
python main.py ../generated_instances/small_1.txt --mode single --iters 100
```

Multi-run eksperiment (izveštaj u `reports/`):

```bash
python main.py ../generated_instances/medium_1.txt --mode multi --iters 100 --runs 10
```

Kontrola broja paralelnih worker-a:

```bash
python main.py ../generated_instances/large_1.txt --mode single --iters 50 --workers 8
python main.py ../generated_instances/large_1.txt --mode single --iters 50 --workers 1   # čisto serijski
```

Bez `--workers`, podrazumevano se koristi `cpu_count() - 1` procesa.

Batch nad svim medium/large instancama:

```bash
bash main_runner.sh
```

## Format instance

Isti kao u `GRASP_part` — plain text, celi brojevi u redosledu:
`I J K`, pa `s[I]`, `d[K]`, `b[I][J]`, `f[I][J]`, `c[J][K]`, `g[J][K]`.
Vidi `CLAUDE.md` u root-u projekta za detalje o veličinama instanci.

## Objašnjenje modula

### `utils.py`

- **`TSFCTPData`** — dataclass sa dimenzijama `(I, J, K)`, ponudom `s`, tražnjom `d`
  i četiri cost matrice: `b`, `f` (supplier→DC var/fixed) i `c`, `g` (DC→customer var/fixed).
- **`read_instance(filename)`** — parsira flat-text fajl instance u `TSFCTPData`.
- **`get_results_filename(instance_path)`** — mapira ime instance (`small_*`,
  `medium_*`, `large_*`) na odgovarajući CPLEX rezultat fajl. Putanja se računa
  relativno na lokaciju `utils.py`, ne na CWD, pa radi bez obzira odakle se
  `main.py` pokreće.
- **`load_optimal_for_instance(instance_path)`** — učitava optimalnu (ili best-feasible)
  vrednost i vreme iz CPLEX rezultata, ako postoji (poredi po `os.path.basename`,
  pa ne zavisi od toga da li je putanja data sa ili bez `../` prefiksa).
- **`GRASPExperiment`** — orkestrira multi-run eksperimente, prati sve što je
  potrebno za MA izveštaj (`sol_i`, `t_i`, `ttot_i`, `eval_i`, `gap_i`, agregati):
  - `single_run(run_id, seed)` — jedan seed-ovan GRASP run; raspakuje
    `(best, best_cost, total_time, time_to_best, eval_count)` iz `GRASP.run()`.
  - `run_multiple(num_runs, verbose)` — pokreće `num_runs` runova sa determinističkim
    seed-ovima (`42 + run*100`), vraća agregatnu statistiku.
  - `_compute_stats(results)` — računa `Bestsol` (optimum ako je poznat i bolji
    od MA rezultata, inače najbolji MA rezultat), `gap_i` po run-u, srednje
    vrednosti `t`, `ttot`, `eval`, `agap` i `σ`.
  - `print_report(stats)` — gradi JEDAN tekst izveštaja i taj isti tekst i
    ispisuje na konzolu i upisuje u `reports/GRASP_report_{instance}.txt`
    (fajl i terminal su uvek identični). Izveštaj sadrži dimenzije instance,
    platformu (`platform.platform()`, broj CPU-ova), optimum/best-known +
    njegovo vreme, `Bestsol`, tabelu po run-u, i agregatne srednje vrednosti.

### `GRASP.py`

- **`Solution`** — čuva `x[I,J]` (supplier→DC) i `y[J,K]` (DC→customer) tokove flow-a.
  - `copy()` — duboka kopija (potrebna jer se local search grana probom kroz "trial" kopije).
  - `compute_cost()` — računa ukupan trošak (var + fixed) i skup otvorenih DC-ova
    (`open_dc`), potpuno vektorizovano preko numpy-ja.
  - `is_valid()` — proverava da li tražnja/ponuda/balans DC-ova važe.

- **`repair(sol, data)`** — pošto se tokom pretrage menja samo `y`, ova funkcija
  pohlepno rekonstruiše `x` od nule tako da svaki DC dobije tačno onoliko robe od
  dobavljača koliko `y` traži, minimizujući trošak dobavljača (`b + f/demand`) po DC-u,
  uz `lexsort` po (score, preostali kapacitet). Vraća `False` ako neki DC ne može
  biti snabdeven (nedostatak kapaciteta).

- **`Construction`** — GRASP pohlepno-randomizovana konstrukcija:
  - `build()` — dodeljuje tražnju kupaca od najveće ka najmanjoj; za svaku jedinicu
    tražnje gradi Restricted Candidate List (RCL) preko vektorizovane (I×J) cost
    matrice (supplier×DC), sa pragom kontrolisanim parametrom `alpha`, i bira
    nasumičnog kandidata iz RCL-a.

- **`LocalSearch`** — VND (Variable Neighborhood Descent) sa first-improvement,
  4 susedstva, svaka koristi vektorizovanu (numpy) evaluaciju kandidata umesto
  Python petlji:
  - `_reassign_customer` — premešta celog kupca na jeftiniji DC (do 10 nasumičnih
    ciljnih DC-ova po kupcu).
  - `_add_dc` — otvara zatvoreni DC i vektorizovano premešta sve kupce kojima bi
    to bilo jeftinije (do 15 kandidata).
  - `_remove_dc` — zatvara otvoreni DC, vektorizovano bira najjeftiniju alternativu
    za svakog pogođenog kupca (do 5 kandidata).
  - `_swap_dc` — menja jedan otvoreni DC za jedan zatvoreni (do 30 nasumičnih parova).
  - Svaka izmena `y` prati poziv `repair()` + `compute_cost()` da se proveri i
    izmeri stvarni efekat na trošak; prihvata se prvo poboljšanje (first-improvement).

- **`ElitePool`** — čuva top-12 raznovrsnih rešenja (raznovrsnost po signaturi
  otvorenih DC-ova).

- **`PathRelinking`** — kombinuje dva elitna rešenja evaluacijom unije i preseka
  njihovih skupova otvorenih DC-ova.

- **`_init_worker` / `_construct_and_improve`** — worker funkcije za
  `multiprocessing.Pool`: svaki worker proces dobija instancu podataka jednom
  (kroz `initializer`), zatim za dati `alpha` radi construction+local search i
  vraća samo lagan rezultat (`x`, `y`, `cost`, `open_dc`, broj poziva funkcije
  cilja u tom zadatku) — ne šalje nazad cost-matrice, koje su nepromenjene i
  već dostupne u worker procesu.

- **`_reset_eval_count` / `_get_eval_count` / `_add_eval_count` /
  `_bump_eval_count`** — brojač poziva `Solution.compute_cost()` (`eval_i` iz
  MA izveštaja). Po-proces brojač: u serijskom modu se čita direktno na kraju
  `run()`; u paralelnom modu svaki worker vraća svoju deltu, koju glavni
  proces sabira (`_add_eval_count`) sa sopstvenim pozivima (path relinking,
  shake rade u glavnom procesu).

- **`GRASP`** — glavna petlja:
  - `__init__(data, iters, n_workers=None)` — `n_workers=None` bira
    `cpu_count() - 1`; `n_workers=1` forsira čisto serijski kod.
  - `run()` — bira `_run_serial()` ili `_run_parallel()` na osnovu `n_workers`;
    vraća `(best, best_cost, total_time, time_to_best, eval_count)`, gde je
    `total_time` = `ttot_i`, `time_to_best` = `t_i` (wall-clock trenutak kad je
    `best` prvi put pronađen), `eval_count` = `eval_i`.
  - `_run_serial()` — originalna sekvencijalna GRASP petlja (construction →
    local search → elite pool → path relinking svakih 8 iteracija → shake
    kad se ne poboljšava >20 iteracija); beleži `time_to_best` pri svakom
    poboljšanju.
  - `_run_parallel()` — ista logika, ali construction+local search za batch od
    `n_workers` iteracija se izvršava paralelno u worker procesima; elite pool,
    best-tracking, path relinking i shake ostaju sekvencijalni (zavise od
    deljenog stanja) i primenjuju se posle svakog batch-a, redom kao u serijskoj
    verziji.
  - `_shake(best, best_cost, it, ls, start, time_to_best)` — ruin-and-recreate:
    uklanja skuplju polovinu otvorenih DC-ova iz najboljeg rešenja i
    rekonstruiše; ažurira `time_to_best` ako shake pronađe novi najbolji.

### `main.py`

CLI sa `--mode {single,multi}`, `--iters`, `--runs`, `--workers`. U `multi` modu
koristi `GRASPExperiment` iz `utils.py` (izveštaj se snima u `reports/`).
