import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_data(n_samples=500, random_state=42):
    np.random.seed(random_state)
    os.makedirs("data", exist_ok=True)
    
    # Data di riferimento fissa per riproducibilità
    ref_date = datetime(2026, 10, 1)
    
    # Profili target (con sovrapposizioni realistiche)
    profiles = [
        {"name": "Campioni", "weight": 0.20, "recency": (1, 30), "freq": (15, 40), "monetary": (100, 300)},
        {"name": "Fedeli", "weight": 0.30, "recency": (10, 60), "freq": (8, 20), "monetary": (50, 150)},
        {"name": "A Rischio", "weight": 0.25, "recency": (45, 120), "freq": (5, 15), "monetary": (40, 120)},
        {"name": "Nuovi/Occasionali", "weight": 0.25, "recency": (1, 45), "freq": (1, 5), "monetary": (10, 50)}
    ]
    
    records = []
    regions = ["Nord", "Centro", "Sud", "Isole"]
    categories = ["Elettronica", "Abbigliamento", "Casa", "Sport"]
    
    for i in range(n_samples):
        prof = np.random.choice(profiles, p=[p["weight"] for p in profiles])
        
        # Generazione metriche RFM con rumore gaussiano
        recency = int(np.clip(np.random.uniform(*prof["recency"]) + np.random.normal(0, 5), 1, 180))
        frequency = int(np.clip(np.random.uniform(*prof["freq"]) + np.random.normal(0, 2), 1, 60))
        monetary_avg = round(float(np.clip(np.random.uniform(*prof["monetary"]) + np.random.normal(0, 10), 5, 500)), 2)
        total_spent = round(frequency * monetary_avg, 2)
        
        # Calcolo date reali
        signup_days_ago = recency + np.random.randint(10, 300)
        signup_date = (ref_date - timedelta(days=signup_days_ago)).strftime("%Y-%m-%d")
        
        records.append({
            "customer_id": f"CUST_{i+1000}",
            "signup_date": signup_date,
            "recency": recency,
            "frequency": frequency,
            "monetary_avg": monetary_avg,
            "total_spent": total_spent,
            "region": np.random.choice(regions),
            "top_category": np.random.choice(categories),
            "true_profile": prof["name"]
        })
        
    df = pd.DataFrame(records)
    
    # Salva il profilo reale nascosto per validazione (ARI)
    df[["customer_id", "true_profile"]].to_csv("data/true_profiles.csv", index=False)
    
    # Inserimento di anomalie/mancanti per la fase di pulizia (~2% outlier/null)
    raw_df = df.drop(columns=["true_profile"]).copy()
    
    # Valori mancanti
    null_idx = np.random.choice(raw_df.index, size=int(n_samples * 0.02), replace=False)
    raw_df.loc[null_idx, "monetary_avg"] = np.nan
    
    # Outlier estremi
    outlier_idx = np.random.choice(raw_df.index, size=int(n_samples * 0.01), replace=False)
    raw_df.loc[outlier_idx, "total_spent"] *= 10
    
    # Duplicati
    raw_df = pd.concat([raw_df, raw_df.iloc[:3]], ignore_index=True)
    
    raw_df.to_csv("data/raw_customers.csv", index=False)
    print(f"✅ Generati {len(raw_df)} record in 'data/raw_customers.csv' e profili reali in 'data/true_profiles.csv'")

if __name__ == "__main__":
    generate_data()