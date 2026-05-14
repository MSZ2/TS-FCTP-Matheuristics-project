import sys
from utils import read_instance
from GRASP2 import GRASP
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Korišćenje: python main.py <datoteka_instance> [broj_izvršavanja]")
        print("Primer: python main.py generated_instances/small_3.txt 10")
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
        print()
        
        # Pokreni GRASP sa prikupljanjem metrika
        grasp = GRASP(
            data,
            iters = num_runs
        )
        
        best, best_cost, run_time = grasp.run()
        print("\n" + "=" * 80)
        print(best)
        print("\n" + "=" * 80)
        print(f"Najbolje rešenje: {best_cost}")
        print(f"Vreme izvršavanja: {run_time:.2f} sekundi")

        
    except FileNotFoundError:
        print(f"Greška: Datoteka '{filename}' nije pronađena.")
        sys.exit(1)
    except Exception as e:
        print(f"Greška: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)