import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score

def run_pipeline():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # 1. Caricamento Dati
    df = pd.read_csv("data/raw_customers.csv")
    true_profiles_df = pd.read_csv("data/true_profiles.csv")
    
    # 2. Pulizia e Preprocessing
    df = df.drop_duplicates(subset=["customer_id"])
    df["monetary_avg"] = df["monetary_avg"].fillna(df["monetary_avg"].median())
    df["total_spent"] = df["total_spent"].fillna(df["frequency"] * df["monetary_avg"])
    
    # Trattamento outlier via capping al 99° percentile
    q_max = df["total_spent"].quantile(0.99)
    df["total_spent"] = np.clip(df["total_spent"], 0, q_max)
    
    # Features per la segmentazione (RFM)
    features = ["recency", "frequency", "total_spent"]
    X = df[features]
    
    # Standardizzazione Z-score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Selezione Ottimale di K (Gomito e Silhouette)
    k_results = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        k_results.append({"K": k, "Inerzia": round(km.inertia_, 2), "Silhouette": round(sil, 4)})
        
    k_df = pd.DataFrame(k_results)
    k_df.to_csv("reports/k_comparison.csv", index=False)
    
    # 4. Addestramento Modello Finale K-Means (K=4)
    best_k = 4
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    
    # 5. Algoritmo di Confronto: Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=best_k, random_state=42)
    df["gmm_cluster"] = gmm.fit_predict(X_scaled)
    
    # 6. Mappatura Automatica dei Cluster basata sui Centroidi
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    cent_df = pd.DataFrame(centroids, columns=features)
    
    label_mapping = {}
    for i, row in cent_df.iterrows():
        if row["recency"] > cent_df["recency"].mean() and row["frequency"] < cent_df["frequency"].mean():
            label_mapping[i] = "A Rischio"
        elif row["frequency"] > cent_df["frequency"].mean() and row["total_spent"] > cent_df["total_spent"].mean():
            label_mapping[i] = "Campioni"
        elif row["recency"] <= cent_df["recency"].quantile(0.5) and row["frequency"] <= cent_df["frequency"].quantile(0.5):
            label_mapping[i] = "Nuovi"
        else:
            label_mapping[i] = "Fedeli"
            
    df["cluster_label"] = df["cluster"].map(label_mapping)
    
    # 7. Validazione con Profilo Reale (Adjusted Rand Index)
    merged = df.merge(true_profiles_df, on="customer_id")
    ari_kmeans = adjusted_rand_score(merged["true_profile"], merged["cluster_label"])
    ari_gmm = adjusted_rand_score(merged["true_profile"], merged["gmm_cluster"])
    
    metrics_summary = pd.DataFrame([
        {"Modello": "K-Means", "ARI": round(ari_kmeans, 4), "Silhouette": k_df.loc[k_df['K']==4, 'Silhouette'].values[0]},
        {"Modello": "Gaussian Mixture", "ARI": round(ari_gmm, 4), "Silhouette": round(silhouette_score(X_scaled, df['gmm_cluster']), 4)}
    ])
    metrics_summary.to_csv("reports/model_metrics.csv", index=False)
    
    # 8. Salvataggio Artefatti
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(kmeans, "models/kmeans.pkl")
    joblib.dump(label_mapping, "models/label_mapping.pkl")
    df.to_csv("data/processed_customers.csv", index=False)
    
    print("✅ Pipeline completata con successo! Modelli salvati in 'models/' e report in 'reports/'")

if __name__ == "__main__":
    run_pipeline()