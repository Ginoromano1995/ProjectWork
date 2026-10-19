"""
Pipeline di Machine Learning non supervisionato - PW 28 Segmentazione clienti.

Fasi:
  1. Caricamento e pulizia (duplicati, valori mancanti, outlier)
  2. Standardizzazione z-score delle feature RFM
  3. Valutazione di K (metodo del gomito + indice di silhouette) e scelta motivata
  4. K-Means finale + confronto con Gaussian Mixture Model
  5. Etichettatura automatica dei segmenti (unica per costruzione)
  6. Validazione: ARI vs profili reali, matrice di confusione, stabilita' al variare del seed
  7. Salvataggio di modelli, dataset con cluster e report
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # backend senza interfaccia grafica
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Radice del progetto: se lo script si trova in src/, dati e modelli stanno nella cartella superiore
BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == "src":
    BASE_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42

# Feature RFM. Si usa l'importo MEDIO per richiesta (e non il totale): il totale
# e' frequency x importo medio, quindi pesa due volte la frequenza (collinearita').
FEATURES = ["recency", "frequency", "monetary_avg"]

# La traccia chiede 3-5 gruppi: K viene scelto in questo intervallo
K_MIN, K_MAX = 3, 5
K_EVAL_RANGE = range(2, 8)  # K valutati per il grafico del gomito
N_STABILITY_RUNS = 20


# --------------------------------------------------------------------------- #
# 1. Pulizia
# --------------------------------------------------------------------------- #
def clean_data(df: pd.DataFrame):
    """Pulizia di base. Restituisce (df pulito, dizionario con il report di pulizia)."""
    report = {"righe_iniziali": int(len(df))}

    # Duplicati
    df = df.drop_duplicates(subset=["customer_id"]).copy()
    report["duplicati_rimossi"] = report["righe_iniziali"] - int(len(df))

    # Valori mancanti
    report["recency_mancanti"] = int(df["recency"].isna().sum())
    report["monetary_avg_mancanti"] = int(df["monetary_avg"].isna().sum())

    # Importo medio mancante: lo si ricostruisce da total_spent / frequency, se possibile
    rebuild = df["monetary_avg"].isna() & df["total_spent"].notna() & (df["frequency"] > 0)
    df.loc[rebuild, "monetary_avg"] = df.loc[rebuild, "total_spent"] / df.loc[rebuild, "frequency"]
    # Eventuali residui: mediana (robusta agli outlier)
    for col in ["monetary_avg", "recency", "frequency"]:
        df[col] = df[col].fillna(df[col].median())

    # Outlier: capping dell'importo medio al 99° percentile
    cap = float(df["monetary_avg"].quantile(0.99))
    report["monetary_avg_cap_p99"] = round(cap, 2)
    report["outlier_cappati"] = int((df["monetary_avg"] > cap).sum())
    df["monetary_avg"] = df["monetary_avg"].clip(upper=cap).round(2)

    # Coerenza: il totale speso viene ricalcolato dopo la pulizia
    df["total_spent"] = (df["frequency"] * df["monetary_avg"]).round(2)

    report["righe_finali"] = int(len(df))
    return df, report


# --------------------------------------------------------------------------- #
# 3. Valutazione e scelta di K
# --------------------------------------------------------------------------- #
def evaluate_k(X_scaled: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in K_EVAL_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        rows.append({
            "K": k,
            "Inerzia": round(float(km.inertia_), 2),
            "Silhouette": round(float(silhouette_score(X_scaled, labels)), 4),
            "Davies_Bouldin": round(float(davies_bouldin_score(X_scaled, labels)), 4),
            "Nel_range_traccia": K_MIN <= k <= K_MAX,
        })
    return pd.DataFrame(rows)


def plot_k_evaluation(k_df: pd.DataFrame, best_k: int, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_xlabel("Numero di cluster (K)")
    ax1.set_ylabel("Inerzia (metodo del gomito)", color="tab:blue")
    ax1.plot(k_df["K"], k_df["Inerzia"], marker="o", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Indice di silhouette", color="tab:orange")
    ax2.plot(k_df["K"], k_df["Silhouette"], marker="s", linestyle="--", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.axvline(best_k, color="gray", linestyle=":", label=f"K scelto = {best_k}")
    ax1.legend(loc="upper center")
    plt.title("Valutazione del numero di cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def choose_k(k_df: pd.DataFrame, forced_k=None):
    """Sceglie K: silhouette massima nell'intervallo 3-5 (o K imposto dall'utente)."""
    if forced_k is not None:
        return int(forced_k), f"K={forced_k} impostato manualmente (parametro --k)."
    candidates = k_df[k_df["Nel_range_traccia"]]
    best = candidates.loc[candidates["Silhouette"].idxmax()]
    k = int(best["K"])
    motivation = (
        f"K={k} ha la silhouette piu' alta ({best['Silhouette']:.4f}) tra le configurazioni "
        f"con {K_MIN}-{K_MAX} gruppi richieste dalla traccia."
    )
    return k, motivation


# --------------------------------------------------------------------------- #
# 5. Etichettatura automatica dei segmenti
# --------------------------------------------------------------------------- #
def build_label_mapping(centers_scaled: np.ndarray) -> dict:
    """
    Assegna etichette di marketing ai centroidi (nello spazio standardizzato),
    per ranking: ogni etichetta e' usata una sola volta, quindi non si possono
    avere cluster con lo stesso nome.
    """
    c = pd.DataFrame(centers_scaled, columns=FEATURES)
    value = c["frequency"] + c["monetary_avg"] - c["recency"]
    remaining = set(c.index)
    mapping = {}

    def take(idx, label):
        mapping[int(idx)] = label
        remaining.discard(idx)

    take(value.idxmax(), "Campioni")                                       # alto valore, attivi
    take(c.loc[list(remaining), "recency"].idxmax(), "A Rischio")          # inattivi da piu' tempo
    take(c.loc[list(remaining), "frequency"].idxmin(), "Nuovi")            # poche richieste
    # Eventuali cluster residui (K=4 o 5): ordinati per valore
    leftovers = value[list(remaining)].sort_values(ascending=False).index.tolist()
    for idx, label in zip(leftovers, ["Fedeli", "Occasionali"]):
        take(idx, label)

    assert len(set(mapping.values())) == len(mapping) == len(c), "Etichette non uniche"
    return mapping


# --------------------------------------------------------------------------- #
# Pipeline completa
# --------------------------------------------------------------------------- #
def run_pipeline(forced_k=None) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. Caricamento
    raw_path = DATA_DIR / "raw_customers.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} non esiste: esegui prima data_generator.py")
    df_raw = pd.read_csv(raw_path)
    print(f"Dataset caricato: {len(df_raw)} record")

    # 2. Pulizia + standardizzazione
    df, clean_report = clean_data(df_raw)
    print("Report di pulizia:", json.dumps(clean_report, ensure_ascii=False))
    pd.Series(clean_report).to_csv(REPORTS_DIR / "cleaning_report.csv", header=["valore"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    # 3. Scelta di K
    k_df = evaluate_k(X_scaled)
    best_k, motivation = choose_k(k_df, forced_k)
    k_df.to_csv(REPORTS_DIR / "k_comparison.csv", index=False)
    plot_k_evaluation(k_df, best_k, REPORTS_DIR / "evaluation_k.png")
    print("\n--- Confronto configurazioni ---")
    print(k_df.to_string(index=False))
    print(f"\nScelta di K: {motivation}")

    # 4. Modelli finali
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    gmm = GaussianMixture(n_components=best_k, random_state=RANDOM_STATE, n_init=5)
    df["gmm_cluster"] = gmm.fit_predict(X_scaled)

    # 5. Etichette
    label_mapping = build_label_mapping(kmeans.cluster_centers_)
    df["cluster_label"] = df["cluster"].map(label_mapping)

    # PCA 2D: calcolata una sola volta sull'intero dataset (assi stabili nella dashboard)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    df["PCA1"], df["PCA2"] = coords[:, 0], coords[:, 1]

    # 6. Validazione (ARI sugli ID numerici per entrambi i modelli: confronto equo)
    true_df = pd.read_csv(DATA_DIR / "true_profiles.csv")
    merged = df[["customer_id", "cluster", "gmm_cluster", "cluster_label"]].merge(true_df, on="customer_id")
    ari_km = adjusted_rand_score(merged["true_profile"], merged["cluster"])
    ari_gmm = adjusted_rand_score(merged["true_profile"], merged["gmm_cluster"])

    metrics = pd.DataFrame([
        {"Modello": "K-Means", "K": best_k, "Silhouette": round(float(silhouette_score(X_scaled, df["cluster"])), 4),
         "Davies_Bouldin": round(float(davies_bouldin_score(X_scaled, df["cluster"])), 4), "ARI": round(ari_km, 4)},
        {"Modello": "Gaussian Mixture", "K": best_k, "Silhouette": round(float(silhouette_score(X_scaled, df["gmm_cluster"])), 4),
         "Davies_Bouldin": round(float(davies_bouldin_score(X_scaled, df["gmm_cluster"])), 4), "ARI": round(ari_gmm, 4)},
    ])
    metrics.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)

    confusion = pd.crosstab(merged["true_profile"], merged["cluster_label"])
    confusion.to_csv(REPORTS_DIR / "confusion_matrix.csv")

    # Stabilita': ARI tra la partizione di riferimento e quelle ottenute con altri seed
    stab = []
    for seed in range(100, 100 + N_STABILITY_RUNS):
        labels = KMeans(n_clusters=best_k, random_state=seed, n_init=1).fit_predict(X_scaled)
        stab.append(adjusted_rand_score(df["cluster"], labels))
    pd.DataFrame({"seed": range(100, 100 + N_STABILITY_RUNS), "ARI_vs_riferimento": np.round(stab, 4)}).to_csv(
        REPORTS_DIR / "stability.csv", index=False)

    # 7. Salvataggio artefatti
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(kmeans, MODELS_DIR / "kmeans.pkl")
    joblib.dump(pca, MODELS_DIR / "pca.pkl")
    joblib.dump(label_mapping, MODELS_DIR / "label_mapping.pkl")
    meta = {
        "features": FEATURES, "k": best_k, "motivazione_k": motivation, "random_state": RANDOM_STATE,
        "stabilita_ari_media": round(float(np.mean(stab)), 4), "stabilita_ari_std": round(float(np.std(stab)), 4),
    }
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(DATA_DIR / "processed_customers.csv", index=False)

    # Riepilogo a video
    print("\n--- Metriche dei modelli ---")
    print(metrics.to_string(index=False))
    print(f"\nStabilita' K-Means su {N_STABILITY_RUNS} seed: ARI medio = {np.mean(stab):.3f} (± {np.std(stab):.3f})")
    print("\n--- Matrice di confusione (profilo reale vs segmento) ---")
    print(confusion)
    print("\n--- Profilo medio dei segmenti ---")
    summary = df.groupby("cluster_label")[FEATURES + ["total_spent"]].mean().round(2)
    summary["clienti"] = df["cluster_label"].value_counts()
    print(summary)
    print(f"\n[OK] Pipeline completata. Modelli in {MODELS_DIR}, report in {REPORTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegue il clustering K-Means sui clienti.")
    parser.add_argument("--k", type=int, default=None, help=f"Forza un valore di K (default: scelto con silhouette tra {K_MIN} e {K_MAX})")
    args = parser.parse_args()
    run_pipeline(forced_k=args.k)
