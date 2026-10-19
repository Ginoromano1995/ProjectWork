"""
Esegue l'intera pipeline in sequenza: generazione dati -> clustering -> (opzionale) dashboard.

Uso:
  python run_all.py                 # genera dati e clustering
  python run_all.py --launch        # ... e avvia anche la dashboard
  python run_all.py --n 450 --seed 7 --k 4
"""
import argparse
import subprocess
import sys
from pathlib import Path

from clustering_pipeline import run_pipeline
from data_generator import generate_data

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegue generazione dati, clustering e (opzionale) dashboard.")
    parser.add_argument("--n", type=int, default=500, help="Numero di clienti da generare (300-600)")
    parser.add_argument("--seed", type=int, default=42, help="Seed per la generazione dei dati")
    parser.add_argument("--k", type=int, default=None, help="Forza K (default: scelto con la silhouette tra 3 e 5)")
    parser.add_argument("--launch", action="store_true", help="Avvia la dashboard al termine")
    args = parser.parse_args()

    print("=== 1/2 Generazione dati sintetici ===")
    generate_data(n_samples=args.n, random_state=args.seed)
    print("\n=== 2/2 Pipeline di clustering ===")
    run_pipeline(forced_k=args.k)

    if args.launch:
        print("\nAvvio della dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(BASE_DIR / "dashboard.py")], check=False)
    else:
        print("\nPer aprire la dashboard: streamlit run dashboard.py")
