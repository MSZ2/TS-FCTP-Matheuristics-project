import random
import time
import numpy as np
import os
from GRASP2 import TSFCTPData, Solution, Construction, LocalSearch

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
# FUNKCIJA ZA POKRETANJE GRASP-A I PRIKUPLJANJE METRIKA
# ============================================================================

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
        constructor = Construction(self.data, alpha)
        solution = constructor.construct()
        
        # Lokalno pretraživanje
        local_search = LocalSearch(self.data)
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