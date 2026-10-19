"""
Generatore di dati sintetici per il PW 28 - Segmentazione clienti.

Azienda fittizia: startup digitale che eroga servizi on-demand
(manutenzione casa, pulizia, consulenza IT, logistica rapida).

Output (cartella data/):
  - raw_customers.csv   : dataset "grezzo", con valori mancanti, outlier e duplicati
                          (da pulire nella pipeline). NON contiene il profilo reale.
  - true_profiles.csv   : profilo comportamentale usato per generare ogni cliente.
                          Serve SOLO per validare il clustering (ARI), non come feature.

Tutti i dati sono inventati: nessuna informazione personale.
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Radice del progetto: se lo script si trova in src/, dati e modelli stanno nella cartella superiore
BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == "src":
    BASE_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"

REGIONS = ["Nord", "Centro", "Sud", "Isole"]
CATEGORIES = ["Manutenzione Casa", "Pulizia", "Consulenza IT", "Logistica Rapida"]

# Data di riferimento fissa: garantisce la riproducibilita' delle date di iscrizione
REF_DATE = datetime(2026, 10, 1)

# Profili comportamentali. Gli intervalli si sovrappongono volutamente (e si
# aggiunge rumore gaussiano) per evitare cluster artificialmente perfetti.
PROFILES = [
    {"name": "Campioni", "weight": 0.20, "recency": (1, 30),  "freq": (15, 40), "monetary": (100, 300)},
    {"name": "Fedeli",   "weight": 0.30, "recency": (10, 60), "freq": (8, 20),  "monetary": (50, 150)},
    {"name": "A Rischio", "weight": 0.25, "recency": (45, 120), "freq": (5, 15), "monetary": (40, 120)},
    {"name": "Nuovi",    "weight": 0.25, "recency": (1, 45),  "freq": (1, 5),   "monetary": (10, 50)},
]


def generate_data(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Genera il dataset sintetico e lo salva in data/. Restituisce il raw DataFrame."""
    rng = np.random.default_rng(random_state)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    weights = [p["weight"] for p in PROFILES]
    records = []

    for i in range(n_samples):
        prof = PROFILES[rng.choice(len(PROFILES), p=weights)]

        # Metriche RFM: valore uniforme nel range del profilo + rumore gaussiano
        recency = int(np.clip(rng.uniform(*prof["recency"]) + rng.normal(0, 5), 1, 180))
        frequency = int(np.clip(rng.uniform(*prof["freq"]) + rng.normal(0, 2), 1, 60))
        monetary_avg = round(float(np.clip(rng.uniform(*prof["monetary"]) + rng.normal(0, 10), 5, 500)), 2)
        total_spent = round(frequency * monetary_avg, 2)

        # L'iscrizione non puo' essere successiva all'ultima richiesta
        signup_days_ago = recency + int(rng.integers(10, 300))
        signup_date = (REF_DATE - timedelta(days=signup_days_ago)).strftime("%Y-%m-%d")

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

    df = pd.DataFrame(records)

    # Profilo reale: salvato a parte, solo per la validazione
    df[["customer_id", "true_profile"]].to_csv(DATA_DIR / "true_profiles.csv", index=False)

    raw_df = df.drop(columns=["true_profile"]).copy()

    # --- Imperfezioni realistiche, da gestire nella fase di pulizia ---
    # 1) Valori mancanti (~2%) in colonne USATE dal modello (recency e importo medio)
    n_missing = max(1, int(n_samples * 0.02))
    idx = rng.choice(raw_df.index, size=n_missing, replace=False)
    raw_df.loc[idx, "monetary_avg"] = np.nan
    idx = rng.choice(raw_df.index, size=n_missing, replace=False)
    raw_df.loc[idx, "recency"] = np.nan

    # 2) Outlier (~1%): importo medio anomalo (es. ordini eccezionali), total_spent coerente
    n_out = max(1, int(n_samples * 0.01))
    idx = rng.choice(raw_df.index, size=n_out, replace=False)
    raw_df.loc[idx, "monetary_avg"] = raw_df.loc[idx, "monetary_avg"] * 5
    raw_df.loc[idx, "total_spent"] = (raw_df.loc[idx, "monetary_avg"] * raw_df.loc[idx, "frequency"]).round(2)

    # 3) Righe duplicate
    raw_df = pd.concat([raw_df, raw_df.iloc[:3]], ignore_index=True)

    raw_df.to_csv(DATA_DIR / "raw_customers.csv", index=False)
    print(f"[OK] Generati {len(raw_df)} record in {DATA_DIR / 'raw_customers.csv'}")
    print(f"[OK] Profili reali (solo per validazione) in {DATA_DIR / 'true_profiles.csv'}")
    return raw_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera il dataset sintetico di clienti.")
    parser.add_argument("--n", type=int, default=500, help="Numero di clienti (300-600 richiesti dalla traccia)")
    parser.add_argument("--seed", type=int, default=42, help="Seed per la riproducibilita'")
    args = parser.parse_args()
    generate_data(n_samples=args.n, random_state=args.seed)
