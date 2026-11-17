"""
Generatore di dati sintetici.

L'azienda utilizzata è fittizia: una startup digitale che offre servizi
(manutenzione, pulizia, consulenza IT, logistica).

File prodotti (nella cartella data/):
  - raw_customers.csv   : dataset "grezzo", contenente valori mancanti, outlier e duplicati
                          che verranno gestiti nella pipeline. Non contiene il profilo reale.
  - true_profiles.csv   : profilo con cui è stato generato ogni cliente. Viene utilizzato solo per
                          verificare il funzionamento del clustering (ARI) e non come variabile del modello.

"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Cartella principale del progetto: se lo script si trova all'interno di src/ si risale di un livello,
# in modo che le cartelle data/ e models/ rimangano all'esterno di src/
BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == "src":
    BASE_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"

# liste utilizzate per le due variabili categoriche (zona e tipo di servizio preferito)
REGIONS = ["Nord", "Centro", "Sud", "Isole"]
CATEGORIES = ["Manutenzione Casa", "Pulizia", "Consulenza IT", "Logistica Rapida"]

# data di riferimento fissa, in modo che le date di iscrizione siano identiche a ogni esecuzione
REF_DATE = datetime(2026, 10, 1)

# Quattro profili di comportamento. Gli intervalli sono stati fatti sovrapporre in parte
# (e viene aggiunto del rumore gaussiano), altrimenti i cluster risulterebbero poco realistici.
# weight = probabilità che un cliente appartenga al profilo (la somma è 1)
# recency, freq, monetary = intervalli (min, max) da cui viene estratto il valore
PROFILES = [
    {"name": "Campioni", "weight": 0.20, "recency": (1, 30),  "freq": (15, 40), "monetary": (100, 300)},
    {"name": "Fedeli",   "weight": 0.30, "recency": (10, 60), "freq": (8, 20),  "monetary": (50, 150)},
    {"name": "A Rischio", "weight": 0.25, "recency": (45, 120), "freq": (5, 15), "monetary": (40, 120)},
    {"name": "Nuovi",    "weight": 0.25, "recency": (1, 45),  "freq": (1, 5),   "monetary": (10, 50)},
]


def generate_data(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Crea il dataset sintetico, lo salva nella cartella data/ e restituisce il DataFrame grezzo."""
    # generatore casuale con seed: a parità di seed vengono prodotti gli stessi dati (esperimento ripetibile)
    rng = np.random.default_rng(random_state)
    # la cartella data/ viene creata nel caso in cui non esista
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # probabilità dei profili, nello stesso ordine della lista PROFILES
    weights = [p["weight"] for p in PROFILES]
    records = []

    # ogni iterazione del ciclo corrisponde a un cliente
    for i in range(n_samples):
        # il profilo viene scelto in modo casuale, tenendo conto dei pesi
        prof = PROFILES[rng.choice(len(PROFILES), p=weights)]

        # Metriche RFM: valore uniforme all'interno dell'intervallo del profilo, con l'aggiunta di rumore gaussiano.
        # La funzione np.clip mantiene i valori entro limiti reali (ad esempio nessun giorno negativo)
        recency = int(np.clip(rng.uniform(*prof["recency"]) + rng.normal(0, 5), 1, 180))
        frequency = int(np.clip(rng.uniform(*prof["freq"]) + rng.normal(0, 2), 1, 60))
        monetary_avg = round(float(np.clip(rng.uniform(*prof["monetary"]) + rng.normal(0, 10), 5, 500)), 2)
        # il totale speso è pari al numero di richieste per l'importo medio, in modo da mantenere coerenti i valori
        total_spent = round(frequency * monetary_avg, 2)

        # la data di iscrizione deve essere precedente all'ultima richiesta, quindi si parte dalla recency andando indietro nel tempo
        signup_days_ago = recency + int(rng.integers(10, 300))
        signup_date = (REF_DATE - timedelta(days=signup_days_ago)).strftime("%Y-%m-%d")

        # il cliente viene salvato come dizionario, che corrisponde a una riga del csv finale
        records.append({
            "customer_id": f"CUST-{i + 1:04d}",
            "signup_date": signup_date,
            "region": str(rng.choice(REGIONS)),
            "preferred_category": str(rng.choice(CATEGORIES)),
            "recency": recency,
            "frequency": frequency,
            "monetary_avg": monetary_avg,
            "total_spent": total_spent,
            "true_profile": prof["name"],
        })

    # la lista di dizionari viene trasformata in una tabella pandas
    df = pd.DataFrame(records)

    # Il profilo reale viene salvato in un file separato e utilizzato solo per la validazione.
    # Il clustering è non supervisionato, quindi non deve avere accesso a questa informazione
    df[["customer_id", "true_profile"]].to_csv(DATA_DIR / "true_profiles.csv", index=False)

    # il dataset grezzo corrisponde alla stessa tabella privata della colonna del profilo reale
    raw_df = df.drop(columns=["true_profile"]).copy()

    # vengono inserite alcune imperfezioni, come accade nei dati reali, da gestire nella fase di pulizia
    # 1) valori mancanti (circa 2%) in recency e importo medio, colonne effettivamente utilizzate dal modello
    n_missing = max(1, int(n_samples * 0.02))
    # replace=False evita di selezionare due volte la stessa riga
    idx = rng.choice(raw_df.index, size=n_missing, replace=False)
    raw_df.loc[idx, "monetary_avg"] = np.nan
    idx = rng.choice(raw_df.index, size=n_missing, replace=False)
    raw_df.loc[idx, "recency"] = np.nan

    # 2) outlier (circa 1%): importo medio moltiplicato per 5 (ad esempio ordini eccezionali),
    # con ricalcolo di total_spent per mantenerlo coerente
    n_out = max(1, int(n_samples * 0.01))
    idx = rng.choice(raw_df.index, size=n_out, replace=False)
    raw_df.loc[idx, "monetary_avg"] = raw_df.loc[idx, "monetary_avg"] * 5
    raw_df.loc[idx, "total_spent"] = (raw_df.loc[idx, "monetary_avg"] * raw_df.loc[idx, "frequency"]).round(2)

    # 3) duplicati: le prime 3 righe vengono copiate in fondo alla tabella
    raw_df = pd.concat([raw_df, raw_df.iloc[:3]], ignore_index=True)

    # il csv viene salvato (una riga per cliente) e vengono stampati i messaggi di conferma
    raw_df.to_csv(DATA_DIR / "raw_customers.csv", index=False)
    print(f"[OK] Generati {len(raw_df)} record in {DATA_DIR / 'raw_customers.csv'}")
    print(f"[OK] Profili reali (solo per validazione) in {DATA_DIR / 'true_profiles.csv'}")
    return raw_df


# questo blocco viene eseguito solo se il file è lanciato direttamente da terminale
if __name__ == "__main__":
    # tramite argparse è possibile modificare numero di clienti e seed senza intervenire sul codice
    parser = argparse.ArgumentParser(description="Genera il dataset sintetico di clienti.")
    parser.add_argument("--n", type=int, default=500, help="Numero di clienti (300-600 richiesti dalla traccia)")
    parser.add_argument("--seed", type=int, default=42, help="Seed per la riproducibilita'")
    args = parser.parse_args()
    generate_data(n_samples=args.n, random_state=args.seed)
