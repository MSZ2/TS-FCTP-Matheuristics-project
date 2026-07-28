# Ispravka: pogrešna pretpostavka "jedan DC po kupcu" u lokalnoj pretrazi

## Problem (prijavljen u recenziji)

Model dozvoljava da jednog kupca `k` snabdeva više distributivnih centara
odjednom (`y[:, k]` može imati više nenula elemenata — split tražnje). Dva
poteza u `LocalSearch` (`GRASP_part_parallel/GRASP.py`) su to ignorisala i
pretpostavljala da svaki kupac ima tačno jednog "vlasnika" (jedan DC koji ga
snabdeva):

```python
has_owner = sol.y.any(axis=0)
owner = np.where(has_owner, np.argmax(sol.y > 0, axis=0), -1)
```

`np.argmax(sol.y > 0, axis=0)` vraća indeks **prvog** DC-a (najmanji indeks)
koji ima `y[j, k] > 0`, ne "vlasnika" u smislu "jedini DC koji ga snabdeva".
Kad je kupac `k` bio snabdeven sa dva ili više DC-a:

- `_reassign_customer()` — komentar je govorio "Move a whole customer to a
  cheaper DC", ali je kod premeštao samo tok sa tog jednog pronađenog DC-a
  (`qty = sol.y[j_from, k]`) na novi DC, dok su ostali delovi toka istog
  kupca ostajali netaknuti na starim DC-ovima. Kupac je i dalje ostajao
  split, samo malo drugačije.
- `_add_dc()` — ista greška pri proceni da li treba otvoriti novi DC:
  procena uštede (`old_full`) i sama zamena su računate kao da je ceo
  `d.d[k]` dolazio sa jednog DC-a, pa se ponovo premeštao samo prvi
  pronađeni deo toka.

Posledica: čitava klasa poteza — "konsolidovati split kupca na jedan
(jeftiniji) DC" — u praksi nikad nije bila isprobana. Pretraga je ostajala
zaglavljena u konfiguracijama sa nepotrebno split-ovanim kupcima, što
objašnjava zašto se nije dostizalo optimalno/best-known rešenje.

## Ispravka

U oba poteza, umesto lookup-a "vlasnika", trošak i tok se računaju
sumiranjem preko **svih** DC-ova koji trenutno snabdevaju kupca, a pri
prihvatanju poteza se ceo red `y[:, k]` (odnosno kolone za pogođene kupce)
najpre nulira, pa se cela količina upisuje na ciljni/novi DC:

```python
# umesto owner-lookup-a:
old_full = (sol.y * d.c).sum(axis=0) + ((sol.y > 0) * d.g).sum(axis=0)
...
trial.y[:, k] = 0
trial.y[j_to, k] = total          # total = sol.y[:, k].sum()
```

Ovim se kupac stvarno premešta u celosti, bez obzira da li je pre poteza bio
snabdeven sa jednog ili više DC-a. Analogna izmena je urađena i u
`_add_dc()` (`trial.y[:, ks] = 0; trial.y[j_new, ks] = d.d[ks]`).

`_remove_dc()` i `_swap_dc()` nisu imali ovu grešku — oni rade po
**DC-u** (uzimaju sve kupce koje trenutno snabdeva taj konkretan DC,
`ks = np.flatnonzero(trial.y[j] > 0)`), ne po "vlasniku" kupca, pa već
ispravno pomeraju ceo tok tog DC-a bez obzira na split.

## Provera

- `Solution.is_valid()` prolazi nepromenjeno na small/medium instancama
  nakon izmene (demand/supply/balans DC-ova ostaju zadovoljeni — potez samo
  premešta postojeći tok, ne menja ukupne količine).
- Direktan test `_reassign_customer()` na `medium_1.txt` (seed=1): od 80
  kupaca, 17 je nakon construction-a bilo split-ovano; sam taj jedan potez
  je u 30 uzastopnih pozivа spustio trošak sa **1 198 965** na **918 548**
  (~23%), dok je pre ispravke ostajao potpuno neaktivan na istom stanju.
  Pun VND (`LocalSearch.improve`) na istom stanju spušta na **450 199**.

## Napomena o postojećim izveštajima

Svih 60 fajlova u `GRASP_part_parallel/reports/` je generisano **pre** ove
ispravke — reflektuju stari (buggy) algoritam i treba ih regenerisati.

## Kako regenerisati rezultate

Ceo batch (small/medium/large, `main_runner.sh` trenutno koristi 3 run-a i
500/200/100 iteracija respektivno — po potrebi izmeni te brojeve direktno u
skripti):

```bash
cd GRASP_part_parallel
source ../venv/bin/activate
bash main_runner.sh
```

Ili pojedinačna instanca sa proizvoljnim brojem run-ova/iteracija (i,
opciono, broja paralelnih worker-a — default je `cpu_count() - 1`):

```bash
cd GRASP_part_parallel
source ../venv/bin/activate
python main.py ../generated_instances/medium_1.txt --mode multi --iters 200 --runs 3
python main.py ../generated_instances/large_1.txt  --mode multi --iters 100 --runs 3 --workers 8
```

Izveštaj se i ispisuje na ekran i upisuje u
`reports/GRASP_report_{instance}.txt` (preko `GRASPExperiment.print_report`
iz `utils.py`).

# Ispravka: preterano uska komšijstva u lokalnoj pretrazi na malim instancama

## Problem

Sve četiri komšijske strukture u `LocalSearch` (`GRASP_part_parallel/GRASP.py`)
ograničavaju broj kandidata koje first-improvement proverava po pozivu, na
fiksne konstante:

- `_reassign_customer`: najviše 10 ciljnih DC-ova (od mogućih `J-1`)
- `_add_dc`: najviše 15 zatvorenih DC-ova
- `_remove_dc`: najviše 5 otvorenih DC-ova
- `_swap_dc`: najviše 30 nasumičnih (otvoren, zatvoren) parova

Te konstante su birane da velike instance (J do 320) budu brze, ali kod malih
instanci (J do 36) nepotrebno sasecaju pretragu iako bi iscrpna pretraga bila
jeftina — npr. za `small_14` (J=24) `_reassign_customer` je gledao samo
10 od 23 mogućih ciljnih DC-ova po kupcu. Posledica: gap prema CPLEX
optimalnom rešenju je na "težim" malim instancama (small_12–small_20, sa
više DC-ova/kupaca) rastao i do ~17%, jer je lokalna pretraga propuštala
poteze koje bi lako našla bez ograničenja.

Čist procenat od `J` nije odgovarajuća mera za skaliranje ovih ograničenja:
odnos "trenutni cap / J" u postojećem kodu već nije konstantan — ide od
~100% (J=8, gde je `min(J,10)=8`, već iscrpno) do ~3% (J=320). To je inverzna
relacija (`min(J, konstanta)`), ne linearna. Fiksni procenat kalibrisan za
velike instance (npr. 3%) dao bi apsurdno mali cap na malim instancama, a
procenat kalibrisan da bude generozan na malim instancama (npr. 70–100%) bi
eksplodirao broj kandidata na velikim.

## Ispravka

Zadržan je postojeći obrazac `min(pool_size, konstanta)` (samoskalirajući —
iscrpan kad je `pool_size ≤ konstanta`, ograničen iznad toga), ali su
konstante podignute da pokriju ceo opseg malih instanci (max J=36):

- `_reassign_customer`: `min(J, 10)` → `min(J, 36)`
- `_add_dc`: `min(len(closed), 15)` → `min(len(closed), 36)`
- `_remove_dc`: `min(len(open_dcs), 5)` → `min(len(open_dcs), 20)`
- `_swap_dc`: `min(30, open·closed)` → `min(150, open·closed)`, i generisanje
  parova promenjeno da bude bez duplikata: kad cap pokriva ceo proizvod
  `open × closed`, koristi se kompletan (promešan) Dekartov proizvod umesto
  nasumičnog uzorkovanja sa ponavljanjem.

Efekat: na malim instancama (J≤36) sve četiri komšijske strukture postaju
iscrpne ili skoro iscrpne. Na medium (J=40–120) i large (J=140–320) se
ponašanje praktično ne menja — plafon od 36 (odnosno 20/150) je i dalje
manji od tipičnog `pool_size` na tim veličinama, osim na samoj donjoj granici
medium opsega (J≈40) gde je efekat blag.

## Napomena

Ova izmena dolazi nakon ispravke "jedan DC po kupcu" iznad, pa svi postojeći
izveštaji u `reports/` treba ponovo regenerisati i zbog ove promene (isti
postupak kao gore).
