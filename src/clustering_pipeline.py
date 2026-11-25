"""
Pipeline di Machine Learning non supervisionato.

Fasi principali:
  1. Caricamento e pulizia dei dati (duplicati, valori mancanti, outlier)
  2. Standardizzazione z-score delle feature RFM
  3. Valutazione di K (metodo del gomito e indice di silhouette) e scelta del valore migliore
  4. K-Means finale e confronto con un Gaussian Mixture Model
  5. Assegnazione automatica dei nomi ai segmenti (ogni nome viene utilizzato in modo univoco)
  6. Validazione: ARI rispetto ai profili reali, matrice di confusione e stabilità al variare del seed
  7. Salvataggio di modelli, dataset con i cluster e report
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # backend senza finestra, in quanto i grafici vengono salvati soltanto come immagine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Cartella principale del progetto: se lo script si trova in src/ si risale di un livello,
# in modo che dati e modelli vengano salvati nella cartella esterna a src/
BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == "src":
    BASE_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# seed fisso, in modo che i risultati siano identici a ogni esecuzione (riproducibilità)
RANDOM_STATE = 42

# Feature RFM. Come terza variabile è stato utilizzato l'importo MEDIO per richiesta e non il totale speso:
# il totale corrisponde a frequenza per importo medio, quindi la frequenza verrebbe considerata due volte
# (le due variabili risulterebbero collineari).
FEATURES = ["recency", "frequency", "monetary_avg"]

# K viene scelto nell'intervallo da 3 a 5 gruppi
K_MIN, K_MAX = 3, 5
K_EVAL_RANGE = range(2, 8)  # valori di K provati per il grafico del gomito (intervallo più ampio in modo da osservare meglio la curva)
N_STABILITY_RUNS = 20  # numero di ripetizioni del K-Means con seed diversi per verificarne la stabilità


# --------------------------------------------------------------------------- #
# 1. Pulizia
# --------------------------------------------------------------------------- #
def clean_data(df: pd.DataFrame):
    """Pulizia di base. Restituisce il df pulito e un dizionario con il report delle operazioni effettuate."""
    report = {"righe_iniziali": int(len(df))}

    # duplicati: due righe con lo stesso customer_id corrispondono allo stesso cliente, quindi ne viene mantenuta una sola
    df = df.drop_duplicates(subset=["customer_id"]).copy()
    report["duplicati_rimossi"] = report["righe_iniziali"] - int(len(df))

    # valori mancanti: vengono prima contati (per il report) e poi sostituiti
    report["recency_mancanti"] = int(df["recency"].isna().sum())
    report["monetary_avg_mancanti"] = int(df["monetary_avg"].isna().sum())

    # se l'importo medio manca viene ricostruito, quando possibile, tramite il rapporto total_spent / frequency
    rebuild = df["monetary_avg"].isna() & df["total_spent"].notna() & (df["frequency"] > 0)
    df.loc[rebuild, "monetary_avg"] = df.loc[rebuild, "total_spent"] / df.loc[rebuild, "frequency"]
    # per i valori rimanenti viene utilizzata la mediana, meno influenzata dagli outlier rispetto alla media
    for col in ["monetary_avg", "recency", "frequency"]:
        df[col] = df[col].fillna(df[col].median())

    # outlier: l'importo medio viene limitato al 99° percentile.
    # I valori superiori alla soglia vengono riportati alla soglia, in modo da non distorcere il K-Means
    cap = float(df["monetary_avg"].quantile(0.99))
    report["monetary_avg_cap_p99"] = round(cap, 2)
    report["outlier_cappati"] = int((df["monetary_avg"] > cap).sum())
    df["monetary_avg"] = df["monetary_avg"].clip(upper=cap).round(2)

    # dopo la pulizia il totale speso viene ricalcolato, per mantenerlo coerente con i nuovi importi
    df["total_spent"] = (df["frequency"] * df["monetary_avg"]).round(2)

    report["righe_finali"] = int(len(df))
    return df, report


# --------------------------------------------------------------------------- #
# 3. Valutazione e scelta di K
# --------------------------------------------------------------------------- #
def evaluate_k(X_scaled: np.ndarray) -> pd.DataFrame:
    """Esegue il K-Means con diversi valori di K e calcola le metriche di qualità per ciascuno."""
    rows = []
    for k in K_EVAL_RANGE:
        # n_init=10: l'algoritmo parte da 10 inizializzazioni diverse e mantiene la migliore
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        # Inerzia: somma delle distanze al quadrato dai centroidi (più è bassa, più i gruppi sono compatti)
        # Silhouette: misura la separazione tra i gruppi, con valori da -1 a 1 (più è alta, meglio è)
        # Davies-Bouldin: rapporto tra dispersione interna e distanza tra i gruppi (più è bassa, meglio è)
        rows.append({
            "K": k,
            "Inerzia": round(float(km.inertia_), 2),
            "Silhouette": round(float(silhouette_score(X_scaled, labels)), 4),
            "Davies_Bouldin": round(float(davies_bouldin_score(X_scaled, labels)), 4),
            "Nel_range_traccia": K_MIN <= k <= K_MAX,
        })
    return pd.DataFrame(rows)


def plot_k_evaluation(k_df: pd.DataFrame, best_k: int, path: Path) -> None:
    """Realizza il grafico del gomito (inerzia) affiancato dalla silhouette e lo salva in formato PNG."""
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_xlabel("Numero di cluster (K)")
    ax1.set_ylabel("Inerzia (metodo del gomito)", color="tab:blue")
    ax1.plot(k_df["K"], k_df["Inerzia"], marker="o", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # secondo asse Y a destra, in quanto inerzia e silhouette hanno scale troppo diverse per condividere lo stesso asse
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
    """Sceglie K: il valore con la silhouette più alta tra 3 e 5 (oppure quello impostato manualmente con --k)."""
    if forced_k is not None:
        return int(forced_k), f"K={forced_k} impostato manualmente (parametro --k)."
    # vengono considerati solo i valori di K tra 3 e 5 e viene scelto quello con la silhouette migliore
    candidates = k_df[k_df["Nel_range_traccia"]]
    best = candidates.loc[candidates["Silhouette"].idxmax()]
    k = int(best["K"])
    motivation = (
        f"K={k} ha la silhouette piu' alta ({best['Silhouette']:.4f}) tra le configurazioni "
        f"con {K_MIN}-{K_MAX} gruppi."
    )
    return k, motivation


# --------------------------------------------------------------------------- #
# 5. Etichettatura automatica dei segmenti
# --------------------------------------------------------------------------- #
def build_label_mapping(centers_scaled: np.ndarray) -> dict:
    """
    Assegna un nome di marketing a ciascun centroide (nello spazio standardizzato)
    tramite una classifica. Ogni etichetta viene utilizzata una sola volta, quindi
    non possono esistere due cluster con lo stesso nome.
    """
    c = pd.DataFrame(centers_scaled, columns=FEATURES)
    # punteggio di "valore" del cluster: alto se il cliente acquista spesso e spende molto, più basso se è inattivo da tempo
    value = c["frequency"] + c["monetary_avg"] - c["recency"]
    remaining = set(c.index)  # cluster a cui non è ancora stata assegnata un'etichetta
    mapping = {}

    def take(idx, label):
        # l'etichetta viene assegnata al cluster, che viene poi rimosso da quelli ancora disponibili
        mapping[int(idx)] = label
        remaining.discard(idx)

    take(value.idxmax(), "Campioni")                                       # punteggio più alto, quindi i clienti migliori
    take(c.loc[list(remaining), "recency"].idxmax(), "A Rischio")          # inattivi da più tempo
    take(c.loc[list(remaining), "frequency"].idxmin(), "Nuovi")            # poche richieste
    # se K è 4 o 5 rimangono dei cluster: vengono ordinati per punteggio e ricevono le etichette restanti
    leftovers = value[list(remaining)].sort_values(ascending=False).index.tolist()
    for idx, label in zip(leftovers, ["Fedeli", "Occasionali"]):
        take(idx, label)

    # controllo di sicurezza: se due cluster ricevessero la stessa etichetta l'esecuzione si interrompe
    assert len(set(mapping.values())) == len(mapping) == len(c), "Etichette non uniche"
    return mapping


# --------------------------------------------------------------------------- #
# Pipeline completa
# --------------------------------------------------------------------------- #
def run_pipeline(forced_k=None) -> None:
    """Esegue tutte le fasi in sequenza, dal csv grezzo fino al salvataggio di modelli e report."""
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. caricamento: senza il csv grezzo non è possibile procedere, quindi viene restituito un errore
    raw_path = DATA_DIR / "raw_customers.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} non esiste: esegui prima data_generator.py")
    df_raw = pd.read_csv(raw_path)
    print(f"Dataset caricato: {len(df_raw)} record")

    # 2. pulizia e successiva standardizzazione
    df, clean_report = clean_data(df_raw)
    print("Report di pulizia:", json.dumps(clean_report, ensure_ascii=False))
    pd.Series(clean_report).to_csv(REPORTS_DIR / "cleaning_report.csv", header=["valore"])

    # Standardizzazione z-score: (valore - media) / deviazione standard.
    # Questo passaggio è necessario perché il K-Means si basa sulle distanze e le feature hanno scale molto diverse
    # (giorni, euro, numero di richieste), quindi senza standardizzazione l'importo influenzerebbe il risultato in modo eccessivo
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    # 3. scelta di K
    k_df = evaluate_k(X_scaled)
    best_k, motivation = choose_k(k_df, forced_k)
    k_df.to_csv(REPORTS_DIR / "k_comparison.csv", index=False)
    plot_k_evaluation(k_df, best_k, REPORTS_DIR / "evaluation_k.png")
    print("\n--- Confronto configurazioni ---")
    print(k_df.to_string(index=False))
    print(f"\nScelta di K: {motivation}")

    # 4. modelli finali
    # K-Means: ogni cliente viene assegnato al centroide più vicino
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Gaussian Mixture: secondo algoritmo, utilizzato solo come confronto
    gmm = GaussianMixture(n_components=best_k, random_state=RANDOM_STATE, n_init=5)
    df["gmm_cluster"] = gmm.fit_predict(X_scaled)

    # 5. etichette
    label_mapping = build_label_mapping(kmeans.cluster_centers_)
    df["cluster_label"] = df["cluster"].map(label_mapping)

    # PCA in 2D, calcolata una sola volta sull'intero dataset in modo che gli assi non cambino nella dashboard
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    df["PCA1"], df["PCA2"] = coords[:, 0], coords[:, 1]

    # 6. validazione (l'ARI viene calcolato sugli ID numerici di entrambi i modelli, in modo che il confronto sia alla pari)
    # L'ARI (Adjusted Rand Index) confronta i cluster individuati con i profili utilizzati per generare i dati:
    # 1 = partizioni identiche, 0 = accordo casuale
    true_df = pd.read_csv(DATA_DIR / "true_profiles.csv")
    merged = df[["customer_id", "cluster", "gmm_cluster", "cluster_label"]].merge(true_df, on="customer_id")
    ari_km = adjusted_rand_score(merged["true_profile"], merged["cluster"])
    ari_gmm = adjusted_rand_score(merged["true_profile"], merged["gmm_cluster"])

    # tabella riassuntiva delle metriche dei due modelli, con una riga per ciascuno
    metrics = pd.DataFrame([
        {"Modello": "K-Means", "K": best_k, "Silhouette": round(float(silhouette_score(X_scaled, df["cluster"])), 4),
         "Davies_Bouldin": round(float(davies_bouldin_score(X_scaled, df["cluster"])), 4), "ARI": round(ari_km, 4)},
        {"Modello": "Gaussian Mixture", "K": best_k, "Silhouette": round(float(silhouette_score(X_scaled, df["gmm_cluster"])), 4),
         "Davies_Bouldin": round(float(davies_bouldin_score(X_scaled, df["gmm_cluster"])), 4), "ARI": round(ari_gmm, 4)},
    ])
    metrics.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)

    # matrice di confusione: righe = profilo reale, colonne = segmento individuato
    confusion = pd.crosstab(merged["true_profile"], merged["cluster_label"])
    confusion.to_csv(REPORTS_DIR / "confusion_matrix.csv")

    # Stabilità: ARI tra la partizione di riferimento e quelle ottenute con altri seed.
    # Se i valori restano vicini a 1 il risultato non dipende dal seed scelto
    stab = []
    for seed in range(100, 100 + N_STABILITY_RUNS):
        labels = KMeans(n_clusters=best_k, random_state=seed, n_init=1).fit_predict(X_scaled)
        stab.append(adjusted_rand_score(df["cluster"], labels))
    pd.DataFrame({"seed": range(100, 100 + N_STABILITY_RUNS), "ARI_vs_riferimento": np.round(stab, 4)}).to_csv(
        REPORTS_DIR / "stability.csv", index=False)

    # 7. salvataggio di tutto ciò che serve alla dashboard, in modo da classificare nuovi clienti senza riaddestrare il modello
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

    # riepilogo a schermo: metriche, stabilità, matrice di confusione e profilo medio dei segmenti
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


# punto di ingresso: da terminale è possibile passare il valore di K tramite --k
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegue il clustering K-Means sui clienti.")
    parser.add_argument("--k", type=int, default=None, help=f"Forza un valore di K (default: scelto con silhouette tra {K_MIN} e {K_MAX})")
    args = parser.parse_args()
    run_pipeline(forced_k=args.k)
