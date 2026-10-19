import os
import subprocess
import sys

def main():
    print("===================================================")
    print("   AVVIO AUTOMATICO PROGETTO CLUSTERING RFM - PEGASO")
    print("===================================================\n")

    # 1. Creazione ambiente virtuale (venv) se non esiste
    if not os.path.exists("venv"):
        print("[+] Creazione dell'ambiente virtuale Python (venv)...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])

    # Determina i percorsi dei binari in base al Sistema Operativo
    if os.name == "nt":  # Windows
        venv_python = os.path.join("venv", "Scripts", "python.exe")
        venv_pip = os.path.join("venv", "Scripts", "pip.exe")
    else:  # macOS / Linux
        venv_python = os.path.join("venv", "bin", "python")
        venv_pip = os.path.join("venv", "bin", "pip")

    # 2. Aggiornamento pip e installazione dipendenze
    print("[+] Verifica e installazione dei pacchetti da requirements.txt...")
    subprocess.run(
        [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    subprocess.run([venv_pip, "install", "-r", "requirements.txt"])

    # 3. Controllo presenza dati e modelli
    data_exists = os.path.exists("data/processed_customers.csv")
    model_exists = os.path.exists("models/kmeans.pkl")

    if not (data_exists and model_exists):
        print("\n[+] Dati o modelli non trovati. Esecuzione pipeline Machine Learning...")
        
        # 3a. Generazione Dati Sintetici + Anomalie
        print("  -> Generazione dataset e profili reali...")
        subprocess.run([venv_python, "src/data_generator.py"], check=True)
        
        # 3b. Preprocessing, Clustering, Valutazione ARI e Salvataggio Modelli
        print("  -> Esecuzione pipeline di clustering e validazione...")
        subprocess.run([venv_python, "src/clustering_pipeline.py"], check=True)
    else:
        print("\n[+] Dataset elaborato e modelli trovati in 'models/'. Fase di training saltata.")

    print("\n===================================================")
    print("   TUTTO PRONTO! Avvio della Dashboard Streamlit...")
    print("===================================================\n")

    # 4. Avvio Dashboard Streamlit con la sintassi consigliata nel venv
    subprocess.run([venv_python, "-m", "streamlit", "run", "src/dashboard.py"])

if __name__ == "__main__":
    main()