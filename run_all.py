"""
Questo script avvia l'intero progetto tramite le seguenti fasi:
ambiente virtuale -> dipendenze -> dati e clustering -> dashboard.

Utilizzo:
  python run_all.py                # esegue tutte le fasi, saltando il training se dati e modelli sono già presenti
  python run_all.py --force        # rigenera dati e modelli da zero
  python run_all.py --no-launch    # non avvia la dashboard
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # cartella in cui si trova questo script
SCRIPTS = ROOT / "src" if (ROOT / "src").exists() else ROOT


def main() -> None:
    # opzioni da riga di comando, nessuna delle quali è obbligatoria
    parser = argparse.ArgumentParser(description="Avvio automatico del progetto di clustering RFM.")
    parser.add_argument("--force", action="store_true", help="Rigenera dati e modelli anche se esistono")
    parser.add_argument("--no-launch", action="store_true", help="Non avviare la dashboard")
    parser.add_argument("--no-venv", action="store_true", help="Usa l'interprete corrente senza creare il venv")
    parser.add_argument("--n", type=int, default=500, help="Numero di clienti da generare (300-600)")
    parser.add_argument("--seed", type=int, default=42, help="Seed per la generazione dei dati")
    parser.add_argument("--k", type=int, default=None, help="Forza K (default: scelto con la silhouette tra 3 e 5)")
    args = parser.parse_args()

    print("===================================================")
    print("   AVVIO PROJECT WORK PW 28 - PEGASO")
    print("===================================================\n")

    # viene impostata la codifica UTF-8 per l'input/output dei processi figli,
    # in quanto su Windows la mancanza di questa impostazione generava un errore con le lettere accentate
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    # sys.executable corrisponde all'interprete Python con cui viene eseguito questo script
    if args.no_venv:
        python = sys.executable
    else:
        # 1. l'ambiente virtuale viene creato solo se non è già presente
        venv_dir = ROOT / "venv"
        if not venv_dir.exists():
            print("[+] Creazione dell'ambiente virtuale Python (venv)...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        # il percorso dell'interprete all'interno del venv cambia in base al sistema: Scripts\python.exe su Windows, bin/python su Linux e Mac
        python = str(venv_dir / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python"))

        # 2. installazione delle librerie necessarie
        print("[+] Verifica e installazione dei pacchetti da requirements.txt...")
        subprocess.run([python, "-m", "pip", "install", "--upgrade", "pip"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([python, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)

    # 3. il training viene eseguito solo se i file già prodotti non sono presenti (oppure se viene utilizzata l'opzione --force)
    ready = (ROOT / "data" / "processed_customers.csv").exists() and (ROOT / "models" / "kmeans.pkl").exists()
    if args.force or not ready:
        print("\n[+] Esecuzione pipeline...")
        print("  -> Generazione dataset e profili...")
        subprocess.run([python, str(SCRIPTS / "data_generator.py"), "--n", str(args.n), "--seed", str(args.seed)],
                       check=True, cwd=ROOT, env=env)
        print("  -> Pulizia, clustering e validazione...")
        cmd = [python, str(SCRIPTS / "clustering_pipeline.py")]
        if args.k is not None:
            cmd += ["--k", str(args.k)]
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    else:
        print("\n[+] Dataset elaborato e modelli trovati. Fase di training saltata (usa --force per rigenerare).")

    # 4. dashboard: viene avviato Streamlit, che apre automaticamente il browser
    if args.no_launch:
        print("\nPipeline completata. Per la dashboard: streamlit run src/dashboard.py")
        return
    print("\n===================================================")
    print("   TUTTO PRONTO! Avvio della Dashboard Streamlit...")
    print("===================================================\n")
    subprocess.run([python, "-m", "streamlit", "run", str(SCRIPTS / "dashboard.py")], cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
