import os
import subprocess
import sys


def main():
  print("===================================================")
  print("   AVVIO AUTOMATICO PROGETTO PW 28 - UNIPEGASO")
  print("===================================================\n")

  # 1. Crea un ambiente virtuale (venv) se non esiste gia
  if not os.path.exists("venv"):
    print("[+] Creazione dell'ambiente virtuale Python (venv)...")
    subprocess.run([sys.executable, "-m", "venv", "venv"])

  # Determina i percorsi corretti di python e pip nel venv in base al sistema operativo
  if os.name == "nt":  # Windows
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    venv_pip = os.path.join("venv", "Scripts", "pip.exe")
  else:  # Mac / Linux
    venv_python = os.path.join("venv", "bin", "python")
    venv_pip = os.path.join("venv", "bin", "pip")

  # 2. Aggiorna pip e installa le dipendenze da requirements.txt
  print("[+] Installazione automatica dei pacchetti richiesti...")
  subprocess.run(
      [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
  )
  subprocess.run([venv_pip, "install", "-r", "requirements.txt"])

  # 3. Esegue la generazione dati e il clustering se il file CSV non esiste
  if not os.path.exists("data/clustered_customers.csv"):
    print("[+] Dataset non trovato. Generazione dati e esecuzione modello ML...")
    subprocess.run([venv_python, "src/generate_data.py"])
    subprocess.run([venv_python, "src/clustering_model.py"])
  else:
    print("[+] Dataset e modello trovati. Saltata la fase di generazione.")

  print("\n===================================================")
  print("   TUTTO PRONTO! Apertura della Dashboard Streamlit...")
  print("===================================================\n")

  # 4. Avvia la dashboard interattiva
  subprocess.run([venv_python, "-m", "streamlit", "run", "app/dashboard.py"])


if __name__ == "__main__":
  main()