"""
Avvio automatico del progetto: venv -> dipendenze -> dati e clustering -> dashboard.

Uso:
  python run_all.py                # tutto: se dati e modelli esistono salta il training
  python run_all.py --force        # rigenera dati e modelli
  python run_all.py --no-launch    # non avvia la dashboard
Opzioni: --n 450 --seed 7 --k 4 --no-venv
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Gli script possono stare in src/ (struttura consigliata) oppure accanto a run_all.py
SCRIPTS = ROOT / "src" if (ROOT / "src").exists() else ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Avvio automatico del progetto di clustering RFM.")
    parser.add_argument("--force", action="store_true", help="Rigenera dati e modelli anche se esistono")
    parser.add_argument("--no-launch", action="store_true", help="Non avviare la dashboard")
    parser.add_argument("--no-venv", action="store_true", help="Usa l'interprete corrente senza creare il venv")
    parser.add_argument("--n", type=int, default=500, help="Numero di clienti da generare (300-600)")
    parser.add_argument("--seed", type=int, default=42, help="Seed per la generazione dei dati")
    parser.add_argument("--k", type=int, default=None, help="Forza K (default: scelto con la silhouette tra 3 e 5)")
    args = parser.parse_args()

    print("===================================================")
    print("   AVVIO AUTOMATICO PROGETTO CLUSTERING RFM - PEGASO")
    print("===================================================\n")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    if args.no_venv:
        python = sys.executable
    else:
        # 1. Ambiente virtuale
        venv_dir = ROOT / "venv"
        if not venv_dir.exists():
            print("[+] Creazione dell'ambiente virtuale Python (venv)...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        python = str(venv_dir / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python"))

        # 2. Dipendenze
        print("[+] Verifica e installazione dei pacchetti da requirements.txt...")
        subprocess.run([python, "-m", "pip", "install", "--upgrade", "pip"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([python, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)

    # 3. Dati e modelli
    ready = (ROOT / "data" / "processed_customers.csv").exists() and (ROOT / "models" / "kmeans.pkl").exists()
    if args.force or not ready:
        print("\n[+] Esecuzione pipeline Machine Learning...")
        print("  -> Generazione dataset e profili reali...")
        subprocess.run([python, str(SCRIPTS / "data_generator.py"), "--n", str(args.n), "--seed", str(args.seed)],
                       check=True, cwd=ROOT, env=env)
        print("  -> Pulizia, clustering e validazione...")
        cmd = [python, str(SCRIPTS / "clustering_pipeline.py")]
        if args.k is not None:
            cmd += ["--k", str(args.k)]
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    else:
        print("\n[+] Dataset elaborato e modelli trovati. Fase di training saltata (usa --force per rigenerare).")

    # 4. Dashboard
    if args.no_launch:
        print("\nPipeline completata. Per la dashboard: streamlit run src/dashboard.py")
        return
    print("\n===================================================")
    print("   TUTTO PRONTO! Avvio della Dashboard Streamlit...")
    print("===================================================\n")
    subprocess.run([python, "-m", "streamlit", "run", str(SCRIPTS / "dashboard.py")], cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
