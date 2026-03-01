import numpy as np
import random
import time
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import sys

# ============================================================================
# KLASA ZA PODATKE
# ============================================================================

@dataclass
class TSFCTPData:
    I: int  # broj izvora (sources)
    J: int  # broj distributivnih centara (DCs)
    K: int  # broj kupaca (customers)
    s: np.ndarray  # supply of sources [I] - INTEGER
    d: np.ndarray  # demand of customers [K] - INTEGER
    b: np.ndarray  # variable cost i->j [I, J] - INTEGER
    f: np.ndarray  # fixed cost i->j [I, J] - INTEGER
    c: np.ndarray  # variable cost j->k [J, K] - INTEGER
    g: np.ndarray  # fixed cost j->k [J, K] - INTEGER
    
    def validate(self):
        """Provera konzistentnosti podataka"""
        assert len(self.s) == self.I, "Supply vektor mora imati I elemenata"
        assert len(self.d) == self.K, "Demand vektor mora imati K elemenata"
        assert self.b.shape == (self.I, self.J), "b matrica mora biti I x J"
        assert self.f.shape == (self.I, self.J), "f matrica mora biti I x J"
        assert self.c.shape == (self.J, self.K), "c matrica mora biti J x K"
        assert self.g.shape == (self.J, self.K), "g matrica mora biti J x K"
        assert np.sum(self.s) >= np.sum(self.d), "Ukupna ponuda mora biti >= ukupnoj potraznji"
        assert np.all(self.s == self.s.astype(int)), "s mora biti ceo broj"
        assert np.all(self.d == self.d.astype(int)), "d mora biti ceo broj"


# ============================================================================
# FUNKCIJA ZA CITANJE INSTANCI
# ============================================================================

def read_instance(filename):
    """Cita instancu problema iz datoteke"""
    with open(filename, 'r') as f:
        nums = [int(x) for line in f for x in line.split()]

    it = iter(nums)

    # Dimenzije
    I, J, K = next(it), next(it), next(it)

    # Supply i demand
    s = np.array([next(it) for _ in range(I)], dtype=int)
    d = np.array([next(it) for _ in range(K)], dtype=int)

    # Matrice
    b = np.array([[next(it) for _ in range(J)] for _ in range(I)], dtype=int)
    f = np.array([[next(it) for _ in range(J)] for _ in range(I)], dtype=int)
    c = np.array([[next(it) for _ in range(K)] for _ in range(J)], dtype=int)
    g = np.array([[next(it) for _ in range(K)] for _ in range(J)], dtype=int)

    print(f"I={I}, J={J}, K={K}, supply={np.sum(s)}, demand={np.sum(d)}")

    data = TSFCTPData(I, J, K, s, d, b, f, c, g)
    data.validate()

    return data


# ============================================================================
# FUNKCIJA ZA UCITAVANJE OPTIMALNIH REZULTATA
# ============================================================================

def get_results_filename(instance_path):
    """
    Odredjuje koji fajl sa rezultatima treba ucitati na osnovu prefiksa instance.
    
    Primeri:
    - generated_instances/small_1.txt -> results_small.txt
    - generated_instances/medium_5.txt -> results_medium.txt
    - generated_instances/large_3.txt -> results_large.txt
    """
    filename = os.path.basename(instance_path)
    
    if filename.startswith('small_'):
        return "results_small.txt"
    elif filename.startswith('medium_'):
        return "results_medium.txt"
    elif filename.startswith('large_'):
        return "results_large.txt"
    else:
        # Ako ne mozemo da odredimo, vracamo None
        return None


def load_optimal_for_instance(instance_path):
    """
    Ucitava optimalno resenje i vreme za konkretnu instancu.
    Automatski odredjuje koji fajl sa rezultatima treba koristiti.
    """
    results_file = get_results_filename(instance_path)
    
    if results_file is None:
        print(f"Upozorenje: Ne mogu da odredim fajl sa rezultatima za {instance_path}")
        return None, None
    
    try:
        with open(results_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                
                # Prva kolona je putanja do instance
                instance_in_results = parts[0]
                
                # Ako je ovo bas nasa instanca
                if instance_in_results == instance_path:
                    status = parts[1]  # OPTIMAL ili FEASIBLE
                    try:
                        optimal_value = float(parts[2])  # treca kolona
                        optimal_time = float(parts[3])   # cetvrta kolona
                        print(f"Ucitano optimalno resenje za {instance_path}: {optimal_value} (vreme: {optimal_time}s, status: {status})")
                        return optimal_value, optimal_time
                    except (ValueError, IndexError) as e:
                        print(f"Greska pri parsiranju reda: {line}")
                        return None, None
        
        print(f"Nije pronadjeno optimalno resenje za {instance_path} u fajlu {results_file}")
        return None, None
        
    except FileNotFoundError:
        print(f"Fajl {results_file} nije pronadjen.")
        return None, None


# ============================================================================
# FUNKCIJA ZA IME IZVEŠTAJNOG FAJLA
# ============================================================================

def get_report_filename(instance_path):
    """
    Generiše ime izveštajnog fajla na osnovu instance.
    Primer: generated_instances/small_3.txt -> GRASP_report_small_3.txt
    """
    filename = os.path.basename(instance_path)
    name_without_ext = os.path.splitext(filename)[0]  # ukloni .txt
    return f"GRASP_results/GRASP_report_{name_without_ext}.txt"


# ============================================================================
# KLASA ZA RESENJE
# ============================================================================

class TSFCTPSolution:
    def __init__(self, data: TSFCTPData):
        self.data = data
        self.x = np.zeros((data.I, data.J), dtype=int)  # i->j
        self.y = np.zeros((data.J, data.K), dtype=int)  # j->k
        self.total_cost = 0
        self.var_cost = 0
        self.fixed_cost = 0
        
    def calculate_cost(self):
        """Izracunava ukupni trosak ukljucujuci fiksne"""
        self.var_cost = np.sum(self.x * self.data.b) + np.sum(self.y * self.data.c)
        
        self.fixed_cost = 0
        for i in range(self.data.I):
            for j in range(self.data.J):
                if self.x[i, j] > 0:
                    self.fixed_cost += self.data.f[i, j]
        
        for j in range(self.data.J):
            for k in range(self.data.K):
                if self.y[j, k] > 0:
                    self.fixed_cost += self.data.g[j, k]
        
        self.total_cost = self.var_cost + self.fixed_cost
        return self.total_cost
    
    def is_valid(self) -> bool:
        """Proverava da li je resenje validno"""
        # Provera ponude
        for i in range(self.data.I):
            if np.sum(self.x[i, :]) > self.data.s[i]:
                return False
        
        # Provera potraznje - mora biti tacno zadovoljena
        for k in range(self.data.K):
            if np.sum(self.y[:, k]) != self.data.d[k]:
                return False
        
        # Konzistencija DC-ova (ulaz = izlaz)
        for j in range(self.data.J):
            if np.sum(self.x[:, j]) != np.sum(self.y[j, :]):
                return False
        
        # Provera nenegativnosti
        if np.any(self.x < 0) or np.any(self.y < 0):
            return False
        
        return True
    
    def copy(self):
        """Stvara kopiju resenja"""
        new = TSFCTPSolution(self.data)
        new.x = self.x.copy()
        new.y = self.y.copy()
        new.total_cost = self.total_cost
        new.var_cost = self.var_cost
        new.fixed_cost = self.fixed_cost
        return new
    
    def get_active_first_stage(self) -> Set[Tuple[int, int]]:
        """Vraca skup aktivnih veza u prvoj fazi"""
        return {(i, j) for i in range(self.data.I) for j in range(self.data.J) if self.x[i, j] > 0}
    
    def get_active_second_stage(self) -> Set[Tuple[int, int]]:
        """Vraca skup aktivnih veza u drugoj fazi"""
        return {(j, k) for j in range(self.data.J) for k in range(self.data.K) if self.y[j, k] > 0}
    
    def __str__(self):
        """String reprezentacija resenja"""
        s = f"TSFCTP Resenje (trosak: {self.total_cost})\n"
        s += f"  Varijabilni: {self.var_cost}, Fiksni: {self.fixed_cost}\n"
        s += f"  Validno: {self.is_valid()}\n"
        
        # Prva faza - samo aktivne veze
        active_first = self.get_active_first_stage()
        if active_first:
            s += "\nPrva faza (i->j):\n"
            for i, j in sorted(active_first):
                s += f"  {i}->{j}: {self.x[i, j]} (var:{self.data.b[i,j]}, fix:{self.data.f[i,j]})\n"
        
        # Druga faza - samo aktivne veze
        active_second = self.get_active_second_stage()
        if active_second:
            s += "\nDruga faza (j->k):\n"
            for j, k in sorted(active_second):
                s += f"  {j}->{k}: {self.y[j, k]} (var:{self.data.c[j,k]}, fix:{self.data.g[j,k]})\n"
        
        return s


# ============================================================================
# POBOLJSANA KONSTRUKCIJSKA FAZA
# ============================================================================

class ImprovedGRASPConstruction:
    def __init__(self, data: TSFCTPData, alpha: float = 0.2):
        self.data = data
        self.alpha = alpha
    
    def construct(self) -> TSFCTPSolution:
        """Konstruira pocetno resenje koristeci GRASP"""
        solution = TSFCTPSolution(self.data)
        
        # Preostale kolicine
        remaining_supply = self.data.s.copy()
        remaining_demand = self.data.d.copy()
        
        # Vec aktivirane veze
        active_first = set()
        active_second = set()
        
        # Prioritet: kupci s vecom potraznjom prvi (za bolju konsolidaciju)
        customers = list(range(self.data.K))
        customers.sort(key=lambda k: self.data.d[k], reverse=True)
        
        for k in customers:
            need = remaining_demand[k]
            if need == 0:
                continue
            
            while need > 0:
                # Pronadji sve moguce rute s procenom troška
                candidates = self._evaluate_candidates(k, remaining_supply, active_first, active_second)
                
                if not candidates:
                    # Ako nema kandidata s postojećim vezama, dozvoli nove
                    candidates = self._evaluate_candidates(k, remaining_supply, set(), set())
                
                # RCL - restricted candidate list
                min_score = candidates[0][0]
                max_score = candidates[-1][0]
                threshold = min_score + self.alpha * (max_score - min_score)
                
                rcl = [cand for score, cand in candidates if score <= threshold]
                
                # Nasumicni odabir iz RCL
                i, j = random.choice(rcl)
                
                # Odredi kolicinu za slanje
                max_possible = min(remaining_supply[i], need)
                
                # Ako je veza vec aktivna, posalji maksimalno
                if (i, j) in active_first:
                    amount = max_possible
                else:
                    # Inace, posalji vecu kolicinu da opravda fiksni trosak
                    min_amount = max(1, max_possible // 5)
                    amount = random.randint(min_amount, max_possible)
                
                # Dodaj u resenje
                solution.x[i, j] += amount
                solution.y[j, k] += amount
                active_first.add((i, j))
                active_second.add((j, k))
                
                # Azuriraj preostale kolicine
                remaining_supply[i] -= amount
                need -= amount
        
        solution.calculate_cost()
        return solution
    
    def _evaluate_candidates(self, k: int, remaining_supply: np.ndarray,
                            active_first: Set[Tuple[int, int]], 
                            active_second: Set[Tuple[int, int]]) -> List[Tuple[float, Tuple[int, int]]]:
        """
        Evaluacija svih mogucih (i, j) ruta za kupca k
        Vraca listu (score, (i, j)) sortiranu po score-u
        """
        candidates = []
        
        for i in range(self.data.I):
            if remaining_supply[i] <= 0:
                continue
            
            for j in range(self.data.J):
                # Osnovni varijabilni trosak
                var_cost = self.data.b[i, j] + self.data.c[j, k]
                
                # Procena fiksnih troškova
                fixed_penalty = 0.0
                expected_flow = min(remaining_supply[i], self.data.d[k])
                
                if expected_flow > 0:
                    if (i, j) not in active_first:
                        # Amortizacija fiksnog troška kroz očekivani tok
                        fixed_penalty += self.data.f[i, j] / expected_flow
                    
                    if (j, k) not in active_second:
                        fixed_penalty += self.data.g[j, k] / expected_flow
                
                # Ukupni score (manji je bolji)
                score = var_cost + fixed_penalty
                
                # Dodaj mali random faktor za diversifikaciju
                score += random.uniform(0, 0.1)
                
                candidates.append((score, (i, j)))
        
        # Sortiraj po score-u
        candidates.sort(key=lambda x: x[0])
        return candidates


# ============================================================================
# AGRESIVNO LOKALNO PRETRAZIVANJE
# ============================================================================

class AggressiveLocalSearch:
    def __init__(self, data: TSFCTPData):
        self.data = data
    
    def improve(self, solution: TSFCTPSolution) -> TSFCTPSolution:
        """Poboljsava resenje kroz više tipova pretrazivanja"""
        current = solution.copy()
        improved = True
        iteration = 0
        max_iterations = 100
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # 1. Konsolidacija tokova (najvece uštede)
            if self._consolidate_flows(current):
                improved = True
                continue
            
            # 2. Prebacivanje celih DC-ova
            if self._switch_dc(current):
                improved = True
                continue
            
            # 3. Gasenje neprofitabilnih veza
            if self._remove_unprofitable_links(current):
                improved = True
                continue
        
        current.calculate_cost()
        return current
    
    def _consolidate_flows(self, solution: TSFCTPSolution) -> bool:
        """
        Pokušava konsolidirati tokove na manje veza da uštedi fiksne troškove.
        Za svaki DC, ako više izvora šalje u njega, pokušava sve prebaciti na najjeftiniji izvor.
        """
        for j in range(self.data.J):
            # Koji izvori šalju u ovaj DC?
            sources = [(i, solution.x[i, j]) for i in range(self.data.I) if solution.x[i, j] > 0]
            
            if len(sources) <= 1:
                continue
            
            # Sortiraj izvore po varijabilnom trošku (najjeftiniji prvi)
            sources.sort(key=lambda x: self.data.b[x[0], j])
            
            # Pokušaj prebaciti sav tok na najjeftiniji izvor
            best_i = sources[0][0]
            
            # Trenutni fiksni troškovi
            current_fixed = sum(self.data.f[i, j] for i, _ in sources)
            # Novi fiksni trošak (samo jedan)
            new_fixed = self.data.f[best_i, j]
            
            # Ušteda na fiksnim troškovima
            fixed_saving = current_fixed - new_fixed
            
            # Ako je ušteda pozitivna, pokušaj konsolidirati
            if fixed_saving > 0:
                # Zapamti trenutno stanje za slučaj vraćanja
                old_values = [(i, solution.x[i, j]) for i, _ in sources[1:]]
                
                # Prebaci sve na best_i
                for i, flow in sources[1:]:
                    solution.x[best_i, j] += flow
                    solution.x[i, j] = 0
                
                # Proveri da li je rešenje i dalje validno
                if solution.is_valid():
                    return True
                else:
                    # Vrati nazad
                    for i, flow in old_values:
                        solution.x[best_i, j] -= flow
                        solution.x[i, j] = flow
        
        return False
    
    def _switch_dc(self, solution: TSFCTPSolution) -> bool:
        """
        Pokušava prebaciti tokove s jednog DC-a na drugi.
        Računa promenu varijabilnih i fiksnih troškova.
        """
        for j1 in range(self.data.J):
            for j2 in range(self.data.J):
                if j1 == j2:
                    continue
                
                # Za svaki kupac koji prima od j1
                for k in range(self.data.K):
                    flow = solution.y[j1, k]
                    if flow == 0:
                        continue
                    
                    # Izračunaj promenu troška
                    old_var = flow * self.data.c[j1, k]
                    new_var = flow * self.data.c[j2, k]
                    var_diff = new_var - old_var
                    
                    # Promena fiksnih troškova
                    fixed_diff = 0
                    
                    # Ako gasimo vezu j1->k (ako je ovo jedini tok kroz nju)
                    if solution.y[j1, k] > 0 and np.sum(solution.y[j1, :]) == flow:
                        fixed_diff -= self.data.g[j1, k]
                    
                    # Ako palimo novu vezu j2->k
                    if solution.y[j2, k] == 0:
                        fixed_diff += self.data.g[j2, k]
                    
                    total_diff = var_diff + fixed_diff
                    
                    # Ako je promena negativna (ušteda)
                    if total_diff < 0:
                        # Pronađi izvor za ovaj tok
                        for i in range(self.data.I):
                            if solution.x[i, j1] >= flow:
                                # Zapamti trenutno stanje
                                old_x_i_j1 = solution.x[i, j1]
                                old_x_i_j2 = solution.x[i, j2]
                                old_y_j1_k = solution.y[j1, k]
                                old_y_j2_k = solution.y[j2, k]
                                
                                # Prebaci
                                solution.y[j1, k] -= flow
                                solution.y[j2, k] += flow
                                solution.x[i, j1] -= flow
                                solution.x[i, j2] += flow
                                
                                if solution.is_valid():
                                    return True
                                else:
                                    # Vrati nazad
                                    solution.y[j1, k] = old_y_j1_k
                                    solution.y[j2, k] = old_y_j2_k
                                    solution.x[i, j1] = old_x_i_j1
                                    solution.x[i, j2] = old_x_i_j2
                                break
        
        return False
    
    def _remove_unprofitable_links(self, solution: TSFCTPSolution) -> bool:
        """
        Uklanja veze koje su neprofitabilne (mali tok, visok fiksni trošak).
        """
        # Prva faza - veze i->j
        for i in range(self.data.I):
            for j in range(self.data.J):
                if solution.x[i, j] == 0:
                    continue
                
                # Ako je tok mali, a fiksni trošak visok
                if solution.x[i, j] < 3 and self.data.f[i, j] > 100:
                    # Pokušaj prebaciti ovaj tok na drugi DC
                    for j2 in range(self.data.J):
                        if j2 == j:
                            continue
                        
                        # Da li postoji aktivna veza i->j2?
                        if solution.x[i, j2] > 0:
                            # Zapamti trenutno stanje
                            flow = solution.x[i, j]
                            old_x_i_j = solution.x[i, j]
                            old_x_i_j2 = solution.x[i, j2]
                            
                            # Prebaci sav tok
                            solution.x[i, j2] += flow
                            solution.x[i, j] = 0
                            
                            # Trebamo preusmeriti i odgovarajuće y tokove
                            y_changes = []
                            remaining_flow = flow
                            
                            for k in range(self.data.K):
                                if solution.y[j, k] > 0 and remaining_flow > 0:
                                    amount = min(remaining_flow, solution.y[j, k])
                                    y_changes.append((k, solution.y[j, k], solution.y[j2, k]))
                                    solution.y[j, k] -= amount
                                    solution.y[j2, k] += amount
                                    remaining_flow -= amount
                            
                            if solution.is_valid() and remaining_flow == 0:
                                return True
                            else:
                                # Vrati nazad
                                solution.x[i, j] = old_x_i_j
                                solution.x[i, j2] = old_x_i_j2
                                for k, old_y_j_k, old_y_j2_k in y_changes:
                                    solution.y[j, k] = old_y_j_k
                                    solution.y[j2, k] = old_y_j2_k
        
        return False


# ============================================================================
# GLAVNA GRASP KLASA SA PRAĆENJEM METRIKA
# ============================================================================

class GRASPWithOptimalMetrics:
    """
    GRASP klasa koja prikuplja sve potrebne metrike za izveštaj.
    Automatski učitava optimalno rešenje za datu instancu.
    """
    def __init__(self, data: TSFCTPData, 
                 instance_path: str,
                 alpha: float = 0.2,
                 max_iterations_per_run: int = 200,
                 platform_info: str = "AMD Ryzen 5, 16GB RAM"):
        
        self.data = data
        self.instance_path = instance_path
        self.alpha = alpha
        self.max_iterations_per_run = max_iterations_per_run
        self.platform_info = platform_info
        
        # Učitaj optimalno rešenje za ovu instancu
        self.optimal_solution, self.optimal_time = load_optimal_for_instance(instance_path)
        
        # Za čuvanje metrika kroz k izvršavanja
        self.results = []
        
        # Ime izveštajnog fajla
        self.report_filename = get_report_filename(instance_path)
        print(f"Izveštaj će biti sačuvan u: {self.report_filename}")
    
    def single_run(self, run_id: int, seed: int) -> dict:
        """
        Jedno izvršavanje GRASP algoritma sa praćenjem svih metrika.
        """
        random.seed(seed)
        np.random.seed(seed)
        
        start_time = time.time()
        
        best_solution = None
        best_cost = float('inf')
        best_time_found = 0  # ti - početno vreme
        iteration_count = 0
        
        # Različite alpha vrednosti
        alphas = [0.1, 0.2, 0.3, 0.4]
        
        for iteration in range(self.max_iterations_per_run):
            alpha = alphas[iteration % len(alphas)]
            
            # Konstrukcija
            constructor = ImprovedGRASPConstruction(self.data, alpha)
            solution = constructor.construct()
            
            # Lokalno pretraživanje
            local_search = AggressiveLocalSearch(self.data)
            solution = local_search.improve(solution)
            
            if not solution.is_valid():
                continue
            
            cost = solution.calculate_cost()
            iteration_count += 1
            
            # Ažuriraj najbolje rešenje
            if cost < best_cost:
                best_cost = cost
                best_solution = solution.copy()
                best_time_found = time.time() - start_time  # ti - početno vreme
        
        total_time = time.time() - start_time  # ttoti - ukupno vreme
        
        return {
            'run_id': run_id,
            'seed': seed,
            'best_solution': best_solution,
            'best_cost': best_cost,
            'time_to_best': best_time_found,  # ti
            'total_time': total_time,          # ttoti
            'iterations': iteration_count       # iteri
        }
    
    def run_multiple(self, num_runs: int = 10, verbose: bool = True) -> dict:
        """
        Pokreće GRASP algoritam k puta i prikuplja sve metrike.
        """
        self.results = []
        
        for run in range(num_runs):
            seed = 42 + run * 100  # Različiti seed za svako izvršavanje
            
            if verbose:
                print(f"\n--- Izvršavanje {run + 1}/{num_runs} (seed={seed}) ---")
            
            result = self.single_run(run, seed)
            self.results.append(result)
            
            if verbose:
                print(f"    Najbolji trošak: {result['best_cost']}")
                print(f"    Vreme do najboljeg: {result['time_to_best']:.2f}s")
                print(f"    Ukupno vreme: {result['total_time']:.2f}s")
        
        return self._compute_statistics()
    
    def _compute_statistics(self) -> dict:
        """
        Računa sve potrebne statistike prema zahtevu.
        """
        # 1. Odredi Bestsol - prema zahtevu, ako postoji optimalno, koristi njega
        all_costs = [r['best_cost'] for r in self.results]
        best_ma = min(all_costs)
        
        if self.optimal_solution is not None:
            bestsol = self.optimal_solution
            print(f"\nKoristi se optimalno rešenje: {bestsol:.2f} (MA najbolje: {best_ma})")
        else:
            bestsol = best_ma
            print(f"\nNema optimalnog rešenja, koristi se najbolje MA: {bestsol}")
        
        # 2. Izračunaj gap za svako izvršavanje
        gaps = []
        for r in self.results:
            gap = 100 * abs(r['best_cost'] - bestsol) / abs(bestsol)
            r['gap'] = gap
            gaps.append(gap)
        
        # 3. Izračunaj srednje vrednosti
        stats = {
            'num_runs': len(self.results),
            'bestsol': bestsol,
            'optimal_solution': self.optimal_solution,
            'optimal_time': self.optimal_time,
            'platform': self.platform_info,
            'instance_path': self.instance_path,
            
            # Najbolje MA rešenje
            'best_ma_solution': best_ma,
            'best_ma_run': np.argmin(all_costs),
            
            # Srednje vrednosti (obavezno)
            'mean_time_to_best': np.mean([r['time_to_best'] for r in self.results]),  # t
            'mean_total_time': np.mean([r['total_time'] for r in self.results]),      # ttot
            'mean_gap': np.mean(gaps),                                                 # agap
            'std_gap': np.std(gaps),                                                   # σ
            
            # Opciono
            'mean_iterations': np.mean([r['iterations'] for r in self.results]),      # iter
            'all_results': self.results
        }
        
        return stats
    
    def print_report(self, stats: dict):
        """
        Ispisuje izveštaj u traženom formatu i čuva ga u fajl.
        """
        # Generiši sadržaj izveštaja
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("IZVEŠTAJ O IZVRŠAVANJU GRASP METAHEURISTIKE")
        report_lines.append("=" * 80)
        
        # Informacije o instanci
        report_lines.append("\n--- INSTANCA ---")
        report_lines.append(f"Putanja: {self.instance_path}")
        report_lines.append(f"Dimenzije: I={self.data.I}, J={self.data.J}, K={self.data.K}")
        report_lines.append(f"Ukupna ponuda/potražnja: {np.sum(self.data.d)}")
        
        if self.optimal_solution:
            report_lines.append(f"\n--- OPTIMALNO REŠENJE (iz literature) ---")
            report_lines.append(f"Optimalno rešenje: {self.optimal_solution:.2f}")
            if self.optimal_time:
                report_lines.append(f"Vreme za optimalno: {self.optimal_time:.2f}s")
        
        report_lines.append(f"\nNajbolje MA rešenje: {stats['best_ma_solution']}")
        
        report_lines.append(f"\n--- PLATFORMA ---")
        report_lines.append(f"MA izvršavan: {self.platform_info}")
        
        report_lines.append(f"\n--- REZULTATI ({stats['num_runs']} izvršavanja) ---")
        report_lines.append(f"\n{'Run':<6} {'Best Cost':<12} {'Gap (%)':<10} {'t (s)':<10} {'ttot (s)':<10} {'Iter':<8}")
        report_lines.append("-" * 70)
        
        for i, r in enumerate(stats['all_results']):
            report_lines.append(f"{i+1:<6} {r['best_cost']:<12.0f} {r['gap']:<10.2f} "
                  f"{r['time_to_best']:<10.2f} {r['total_time']:<10.2f} {r['iterations']:<8}")
        
        report_lines.append("-" * 70)
        report_lines.append(f"\n--- SREDNJE VREDNOSTI ---")
        report_lines.append(f"srednje početno vreme (t)           = {stats['mean_time_to_best']:.2f} s")
        report_lines.append(f"srednje ukupno vreme (ttot)         = {stats['mean_total_time']:.2f} s")
        report_lines.append(f"srednje odstupanje (agap)           = {stats['mean_gap']:.2f} %")
        report_lines.append(f"standardna devijacija (σ)           = {stats['std_gap']:.2f} %")
        
        if stats['mean_iterations'] > 0:
            report_lines.append(f"srednji broj iteracija (iter)      = {stats['mean_iterations']:.1f}")
        
        report_lines.append("\n" + "=" * 80)
        
        # Ispiši na ekran
        for line in report_lines:
            print(line)
        
        # Sačuvaj u fajl
        with open(self.report_filename, 'w', encoding='utf-8') as f:
            for line in report_lines:
                f.write(line + '\n')
        
        print(f"\nIzveštaj je sačuvan u: {self.report_filename}")


# ============================================================================
# GLAVNI PROGRAM
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Korišćenje: python grasp_report.py <datoteka_instance> [broj_izvršavanja]")
        print("Primer: python grasp_report.py generated_instances/small_3.txt 10")
        sys.exit(1)
    
    filename = sys.argv[1]
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    try:
        # Učitaj instancu
        print("Učitavanje instance...")
        data = read_instance(filename)
        
        print("\n" + "=" * 80)
        print("GRASP METAHEURISTIKA - PRIKUPLJANJE METRIKA")
        print("=" * 80)
        print(f"Problem: {data.I} izvora, {data.J} DC-ova, {data.K} kupaca")
        print(f"Broj izvršavanja: {num_runs}")
        print(f"Maksimum iteracija po izvršavanju: 200")
        print()
        
        # Pokreni GRASP sa prikupljanjem metrika
        grasp = GRASPWithOptimalMetrics(
            data,
            instance_path=filename,
            alpha=0.2,
            max_iterations_per_run=200,
            platform_info="AMD Ryzen 5, 16GB RAM"
        )
        
        stats = grasp.run_multiple(num_runs=num_runs, verbose=True)
        grasp.print_report(stats)
        
    except FileNotFoundError:
        print(f"Greška: Datoteka '{filename}' nije pronađena.")
        sys.exit(1)
    except Exception as e:
        print(f"Greška: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)