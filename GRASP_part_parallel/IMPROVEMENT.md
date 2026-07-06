# Improvements over GRASP_part/GRASP.py

Polazna tačka je bila `GRASP_part/GRASP.py`, koji je bio znatno sporiji od
`GRASP_part/GRASP_HARD.py` na medium/large instancama (zbog čega `main.py`
u `GRASP_part` uopšte i koristi `GRASP_HARD.py`, a ne `GRASP.py`). Cilj je bio
ubrzati `GRASP.py` dovoljno da postane brži i/ili kvalitetniji od `GRASP_HARD.py`,
bez menjanja algoritma (isti RCL/alpha, iste VND susedstva, isti path relinking
i shake) — samo bržom implementacijom istih koraka.

Sve izmene su rađene i verifikovane u tri koraka, svaki potvrđen na `small`,
`medium` i `large` instancama (`is_valid() == True` posle svake izmene).

## 1. Paralelizacija: batch nezavisnih iteracija (`multiprocessing`)

**Problem:** GRASP je čisto CPU-bound Python kod, pa `threading` ne pomaže
(GIL). Sama petlja je inherentno sekvencijalna (elite pool, best-tracking,
path relinking i shake zavise od deljenog stanja), ali construction + local
search za jednu iteraciju zavisi samo od `alpha` i nepromenjenih podataka
instance — potpuno je nezavisna od drugih iteracija.

**Rešenje:** `GRASP._run_parallel()` gradi batch od `n_workers` iteracija
odjednom, šalje ih u `multiprocessing.Pool` (podaci instance se šalju workerima
samo jednom, kroz `initializer`, ne pri svakom pozivu), i posle batch-a
sekvencijalno primenjuje elite pool / best-tracking / path relinking / shake —
identičnim redosledom kao originalna petlja, samo primenjenim na već gotove
rezultate. `n_workers=1` koristi potpuno originalni serijski kod
(`_run_serial()`), pa je ponašanje 1:1 isto ako se paralelizam isključi.

Naivna varijanta bi slala celu `Solution` (uključujući referencu na sve
cost-matrice) nazad iz svakog worker-a; umesto toga workeri vraćaju samo
`(x, y, cost, open_dc)` — cost-matrice ostaju samo u worker procesu.

## 2. Vektorizacija `Construction.build()`

**Problem:** za svaku jedinicu tražnje, originalni kod je u čistom Python-u
gradio listu od I×J tuple-ova (`for i in range(I): for j in range(J): ...`),
sortirao je, pa birao RCL. Za veliku instancu (I≈160, J≈320) to je >50 000
Python objekata po pozivu — i to potencijalno više puta po kupcu, ako mu
tražnja premašuje kapacitet jednog dobavljača.

**Rešenje:** ista RCL/alpha logika, ali cost matrica se gradi i maskira
vektorizovano (numpy broadcasting + boolean maska), umesto Python petlje +
`list.sort()`. Nasumičan izbor i dalje ide kroz seed-ovani `random` modul
(determinizam runova ostaje netaknut).

Usput je uočeno da je `if j in sol.open_dc: base *= 0.95` u originalnom kodu
bio mrtav kod — `sol.open_dc` se popunjava tek u `compute_cost()`, koji se
zove tek na kraju `build()`, pa je taj uslov tokom cele konstrukcije uvek bio
`False`. Vektorizovana verzija ga zato izostavlja bez promene ponašanja.

**Efekat (medium instanca, I=20,J=40,K=80):** construction sa ~sekundi po
pozivu na ~**3 ms** po pozivu (>100× brže). Posle ove izmene, construction
praktično nestaje iz profila — local search postaje dominantan trošak.

## 3. Vektorizacija `Solution.compute_cost()` i `LocalSearch`

Profajliranje (`cProfile`) posle koraka 2 je pokazalo da je local search sada
99%+ vremena, i unutar njega dva glavna krivca:

- **`compute_cost()`** je računao `open_dc` kroz Python petlju od J elemenata,
  sa po 2 poziva `np.sum(...)` po DC-u — tj. 2·J numpy poziva po pozivu funkcije.
  Zamenjeno jednim vektorizovanim `x.sum(axis=0)` / `y.sum(axis=1)` pozivom.
  Ovo se poziva posle **svakog** `repair()` u local search-u, pa je imalo
  veliki multiplikativni efekat.

- **Traženje "ko trenutno opslužuje kupca k"** (`next(j for j in range(J) if
  sol.y[j,k] > 0)`) je bio O(J) Python loop pozivan po kupcu, u
  `_reassign_customer` i `_add_dc`. Zamenjeno jednim vektorizovanim
  `np.argmax(sol.y > 0, axis=0)` po celom pozivu funkcije (izračuna se jednom
  za sve kupce odjednom, ne po kupcu).

- **`_add_dc`, `_remove_dc`, `_swap_dc`** — unutrašnje petlje "za svakog kupca
  kog dodirne ovaj kandidat DC, izračunaj/uporedi trošak" su zamenjene numpy
  fensi-indeksiranjem (masks + `argmin` po osi), umesto Python `for k in
  range(K)` petlje po kandidatu.

- **`_reassign_customer`** je za svakog od K kupaca radio
  `random.shuffle(list(range(J)))` da izvuče do 10 ciljnih DC-ova — puni shuffle
  od J elemenata kad je trebalo samo 10. Zamenjeno sa `random.sample(range(J),
  11)`, što je jeftinije za veliko J.

Ove izmene ne menjaju rezultat pretrage (ista first-improvement logika, ista
RCL/threshold pravila) — samo isti posao rade numpy vektorizacijom umesto
Python petljama.

**Efekat (medium instanca, 10 construction+local-search poziva,
mereno `cProfile`):** 12.8 s → 7.3 s (~1.75×), od čega je `compute_cost`
sa 4.0 s kumulativno pao na 0.9 s. Preostali dominantan trošak je sada
`repair()` (~65% vremena) — namerno netaknut (videti "Šta nije menjano" ispod).

## Šta nije menjano

- **`repair()`** — i dalje je pohlepan, sekvencijalan po DC-u (deli ograničen,
  tačno izbalansiran kapacitet dobavljača — ukupna ponuda == ukupna tražnja).
  Inkrementalno prepravljanje (rekompajliranje samo "prljavih" DC-ova umesto
  svih) bi bilo brže, ali bi promenilo redosled kojim se dobavljači dodeljuju,
  što — pošto je kapacitet tačno izbalansiran bez ikakve rezerve — može
  promeniti da li `repair()` uspeva ili ne za pojedine kandidate. To je rizik
  po korektnost/kvalitet koji nije preuzet u ovoj iteraciji.
- Sama struktura algoritma (RCL/alpha, VND redosled susedstava, path relinking
  svakih 8 iteracija, shake posle 20 iteracija bez poboljšanja) — nepromenjena.

## 4. Izveštaj po MA specifikaciji (t_i, eval_i, kompletan fajl==terminal)

Ovaj deo se ne tiče brzine, već **korektnosti/kompletnosti izveštaja** koji
`GRASPExperiment` generiše (potreban za seminarski izveštaj po formatu:
sol_i, t_i, ttot_i, eval_i, agap, σ, Bestsol...).

Uočeni i ispravljeni problemi:

- **`t_i` (vreme do prvog nalaska najboljeg rešenja) je bio lažiran** —
  `utils.py` je stavljao `time_to_best = total_time` (komentar u kodu je
  doslovno govorio "fallback, no internal tracking"). `GRASP.run()` sada
  interno beleži `time.time() - start` u tačnom trenutku kad se `best_cost`
  poboljša (u glavnoj petlji, path relinking-u i shake-u), i vraća ga kao
  deo rezultata: `(best, best_cost, total_time, time_to_best, eval_count)`.
  Ažurirano i u serijskom i u paralelnom režimu.

- **`eval_i` (broj poziva funkcije cilja) uopšte nije postojao.** Dodat
  globalni brojač u `Solution.compute_cost()` (svaki poziv = jedna evaluacija
  cilja). U serijskom režimu se čita direktno na kraju `run()`. U paralelnom
  režimu svaki worker proces ima svoj nezavisan brojač (multiprocessing =
  odvojena memorija); `_construct_and_improve()` ga resetuje na početku svakog
  zadatka i vraća deltu, koju glavni proces sabira sa svojim sopstvenim
  pozivima (path relinking/shake rade u glavnom procesu).

- **Optimalno/best-known rešenje (`optimal_value`, `optimal_time`) se skoro
  nikad nije učitavalo**, iz dva razloga koja su se poklopila:
  1. `get_results_filename()` je gradio putanju `"CPLEX_part/results/..."`
     relativno na CWD, a `main.py` se po `CLAUDE.md` pokreće iz
     `GRASP_part(_parallel)/`, gde ta putanja ne postoji.
  2. `load_optimal_for_instance()` je poredio `parts[0] == instance_path`
     doslovno — fajl čuva `"generated_instances/medium_1.txt"`, a CLI
     argument je tipično `"../generated_instances/medium_1.txt"`, pa se
     stringovi nikad nisu poklapali.

  Posledica: `agap`/`Bestsol` su se u praksi računali u odnosu na sopstveni
  najbolji rezultat (`best_ma`), ne u odnosu na CPLEX optimum — suprotno
  specifikaciji izveštaja. Ispravljeno: putanja do `CPLEX_part/results/` se
  sada računa relativno na lokaciju `utils.py` fajla (radi bez obzira odakle
  se pokreće skripta), a poređenje ide po `os.path.basename(...)` umesto po
  celoj putanji.

- **Fajl i terminal ispis su sad identični.** `print_report()` gradi jedan
  tekst izveštaja (lista linija spojena u string), pa ga i `print()`-uje i
  upisuje u `reports/GRASP_report_{instance}.txt` — nema više razmimoilaženja
  između onoga što se vidi na ekranu i onoga što se čuva.

Izveštaj sada po run-u čuva: `sol_i`, `t_i`, `ttot_i`, `eval_i`, `gap_i`; a
agregatno: dimenzije instance, platformu (`platform.platform()`, broj CPU-ova),
optimalno/best-known rešenje i njegovo vreme, `Bestsol`, srednje `t`, `ttot`,
`eval`, `agap` i `σ`. `cache_i`/`cache%` namerno nisu dodati — nema
memoizacije funkcije cilja u kodu (rešenja su gotovo uvek različite matrice,
pa bi hit-rate bio nizak); mogu se dodati naknadno ako zatreba.

**Napomena:** `cwd`-nezavisna putanja i basename-poređenje su ispravljeni
samo u `GRASP_part_parallel/utils.py`. `GRASP_part/utils.py` (koji koristi
`GRASP_HARD.py`) ima isti bug i nije menjan u ovom prolazu.

## Rezultati: novi GRASP.py vs GRASP_HARD.py

Isti seed (42), poređenje sa `GRASP_HARD.py` (koji `GRASP_part/main.py`
trenutno koristi za medium/large instance, serijski):

| Instanca | Verzija | Iteracije | Vreme | Cost |
|---|---|---|---|---|
| medium_1 (I=20,J=40,K=80) | GRASP_HARD, serijski | 50 | 23.5 s | 441 072 |
| medium_1 | **novi GRASP.py**, serijski | 50 | 30.4 s | 433 086 |
| medium_1 | **novi GRASP.py**, paralelno (8 workera) | 50 | **9.7 s** | **428 682** |
| large_1 (I=70,J=140,K=280) | GRASP_HARD, serijski | 5 | 92.9 s | 1 031 940 |
| large_1 | **novi GRASP.py**, paralelno (12 workera) | 8 | **72.9 s** | **960 626** |

Zaključak: sam serijski `GRASP.py` (vektorizovan) je nešto sporiji od
`GRASP_HARD.py` po iteraciji na medium instanci — jer koristi šire limite
kandidata u local search-u (temeljnija, ali skuplja pretraga) — ali daje bolji
kvalitet. Sa paralelizacijom (8–12 workera na 16-jezgarnoj mašini), novi
`GRASP.py` je i brži i kvalitetniji od `GRASP_HARD.py` i na medium i na large
instancama, uz više odrađenih iteracija u kraćem vremenu.

## Napomena o determinizmu

`GRASPExperiment` seed-uje samo Python-ov `random` modul (`random.seed(seed)`),
ne i numpy-jev RNG — construction i local search namerno koriste isključivo
`random.*` (nikad `np.random.*`) da runovi ostanu reproducibilni pod istim
seed-om. U paralelnom režimu, svaki worker proces se re-seed-uje na
`os.getpid() ^ time`, pa paralelni run **nije** bit-za-bit reproducibilan
između pokretanja (za razliku od serijskog `n_workers=1` moda, koji jeste) —
ovo je namerni kompromis radi raznovrsnosti pretrage po worker-u.
