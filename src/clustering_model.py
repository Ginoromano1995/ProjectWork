import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_clustering_pipeline(input_path="data/raw_customers.csv", output_path="data/clustered_customers.csv"):
    """
    Pipeline di Machine Learning Non Supervisionato:
    1. Caricamento dati e pre-elaborazione
    2. Standardizzazione delle feature numeriche (RFM)
    3. Valutazione del numero ottimale di cluster (Metodo del Gomito e Indice di Silhouette)
    4. Applicazione di K-Means e salvataggio dei risultati
    """
    
    # 1. Caricamento Dati
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Il file {input_path} non esiste. Esegui prima generate_data.py!")
        
    df = pd.read_csv(input_path)
    print(f"Dataset caricato correttamente: {len(df)} record riscontrati.")
    
    # Selezione delle variabili comportamentali principali (RFM)
    rfm_features = ['recency', 'frequency', 'monetary']
    X = df[rfm_features]
    
    # 2. Normalizzazione dei dati (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Analisi per la scelta del K ottimale (Elbow Method e Silhouette Score)
    k_range = range(2, 7)
    inertias = []
    silhouette_scores = []
    
    print("\n--- Valutazione Configurazioni Cluster ---")
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = kmeans_temp.fit_predict(X_scaled)
        
        inertia = kmeans_temp.inertia_
        sil_score = silhouette_score(X_scaled, labels_temp)
        
        inertias.append(inertia)
        silhouette_scores.append(sil_score)
        
        print(f"K = {k} | Inerzia: {inertia:.2f} | Silhouette Score: {sil_score:.4f}")
        
    # Generazione e salvataggio del grafico di valutazione
    os.makedirs("reports", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.set_xlabel('Numero di Cluster (K)')
    ax1.set_ylabel('Inerzia (Elbow Method)', color='tab:blue')
    ax1.plot(k_range, inertias, marker='o', color='tab:blue', label='Inerzia')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('Silhouette Score', color='tab:orange')
    ax2.plot(k_range, silhouette_scores, marker='s', color='tab:orange', linestyle='--', label='Silhouette')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('Valutazione del Numero di Cluster Ottimale (K)')
    fig.tight_layout()
    plt.savefig("reports/evaluation_k.png")
    plt.close()
    print("\nGrafico di valutazione salvato in: reports/evaluation_k.png")
    
    # 4. Fitting del Modello Finale (Scegliamo K=4 per una segmentazione chiara)
    optimal_k = 4
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = kmeans_final.fit_predict(X_scaled)
    
    # Assegnazione di etichette di marketing descrittive basate sui centroidi
    cluster_mapping = {
        0: "Clienti a Rischio (Churn)",
        1: "Clienti Occasionali",
        2: "Top Spenders (Campioni)",
        3: "Nuovi Clienti"
    }
    # Nota: la mappatura numerica dipenderà dalla posizione dei centroidi
    df['cluster_label'] = df['cluster'].map(lambda c: f"Cluster {c}")
    
    # 5. Salvataggio Dataset Risultante
    df.to_csv(output_path, index=False)
    print(f"Pipeline completata! Dataset con cluster salvato in: {output_path}\n")
    
    # Stampa del profilo medio per ciascun cluster
    print("--- Profilo Medio dei Cluster Individuati ---")
    cluster_summary = df.groupby('cluster')[rfm_features].mean().round(2)
    cluster_summary['count'] = df['cluster'].value_counts()
    print(cluster_summary)

if __name__ == "__main__":
    run_clustering_pipeline()