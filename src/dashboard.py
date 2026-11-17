"""
Dashboard interattiva.
Per avviarla:  streamlit run dashboard.py (è necessario aver prima eseguito python run_all.py)
"""
import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# Cartella principale del progetto: se questo file si trova in src/ si risale di un livello,
# in quanto dati e modelli sono salvati all'esterno di src/
# (Path(__file__) restituisce il percorso di questo file, mentre .parent indica la cartella superiore)
BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == "src":
    BASE_DIR = BASE_DIR.parent
DATA_PATH = BASE_DIR / "data" / "processed_customers.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Le feature sono le stesse utilizzate nel training e devono rispettare lo stesso ordine (altrimenti il modello produce risultati errati)
FEATURES = ["recency", "frequency", "monetary_avg"]
REGIONS = ["Nord", "Centro", "Sud", "Isole"]
CATEGORIES = ["Manutenzione Casa", "Pulizia", "Consulenza IT", "Logistica Rapida"]

AXIS_LABELS = {  # nomi degli assi, utilizzati nei menu a tendina e nei grafici
    "recency": "Recency (giorni dall'ultima richiesta)",
    "frequency": "Frequenza (n. richieste)",
    "monetary_avg": "Importo medio per richiesta (€)",
    "total_spent": "Spesa totale (€)",
}

# Assegnazione colori, in modo che ogni segmento abbia sempre lo stesso colore in tutti i grafici
COLOR_MAP = {
    "Campioni": "#2E7D32",
    "Fedeli": "#1976D2",
    "Occasionali": "#00ACC1",
    "Nuovi": "#F9A825",
    "A Rischio": "#C62828",
}

# Per ogni segmento: (interpretazione, azione di marketing consigliata). Vengono mostrate nella prima scheda della dashboard
SEGMENT_INFO = {
    "Campioni": ("Clienti molto attivi, recenti e ad alto valore.",
                 "Programmi VIP, anteprime esclusive, upsell di servizi premium."),
    "Fedeli": ("Clienti regolari con frequenza e spesa medie.",
               "Programma fedeltà, cross-selling tra categorie, richiesta di recensioni."),
    "Occasionali": ("Clienti che usano il servizio saltuariamente.",
                    "Promozioni stagionali e promemoria per aumentare la frequenza."),
    "Nuovi": ("Clienti recenti con poche richieste e importi bassi.",
              "Onboarding guidato, coupon di benvenuto per la seconda richiesta."),
    "A Rischio": ("Clienti con buona spesa storica ma inattivi da molto tempo.",
                  "Campagne win-back, sconto personalizzato, sondaggio sui motivi di abbandono."),
}

# Impostazioni della pagina (deve essere il primo comando di Streamlit, altrimenti viene generato un errore)
st.set_page_config(page_title="Dashboard Segmentazione Clienti - PW 28", page_icon="📊", layout="wide")


# @st.cache_data: il csv viene letto una sola volta e poi mantenuto in memoria in modo da non rallentare la lettura
@st.cache_data
def load_data() -> pd.DataFrame:
    # se il file non è presente significa che la pipeline non è stata eseguita: viene mostrato l'errore e l'esecuzione si ferma
    if not DATA_PATH.exists():
        st.error("Dati non trovati. Esegui prima la pipeline con: python run_all.py")
        st.stop()
    return pd.read_csv(DATA_PATH, parse_dates=["signup_date"])


# @st.cache_resource: funziona come @st.cache_data ma viene utilizzato per oggetti che non sono dati (in questo caso i modelli)
@st.cache_resource
def load_models():
    # viene verificata la presenza di tutti i file dei modelli
    for name in ["scaler.pkl", "kmeans.pkl", "pca.pkl", "label_mapping.pkl"]:
        if not (MODELS_DIR / name).exists():
            st.error("Modelli non trovati. Esegui prima la pipeline con: python run_all.py")
            st.stop()
    # il file meta.json contiene il valore di K scelto, la motivazione e la stabilità
    meta = json.loads((MODELS_DIR / "meta.json").read_text(encoding="utf-8"))
    return (joblib.load(MODELS_DIR / "scaler.pkl"), joblib.load(MODELS_DIR / "kmeans.pkl"),
            joblib.load(MODELS_DIR / "pca.pkl"), joblib.load(MODELS_DIR / "label_mapping.pkl"), meta)


def classify(raw: pd.DataFrame) -> pd.DataFrame:
    """Assegna segmento e coordinate PCA tramite i modelli già addestrati, senza effettuare un nuovo addestramento."""
    # 1) stessa standardizzazione del training: viene usato transform e non fit in modo da utilizzare la media e la deviazione standard già salvate in precedenza
    scaled = scaler.transform(raw[FEATURES])
    out = raw.copy()
    # 2) predict assegna ogni riga al centroide più vicino, poi l'ID del cluster viene trasformato nella sua etichetta
    out["cluster"] = kmeans.predict(scaled)
    out["cluster_label"] = out["cluster"].map(mapping)
    # 3) coordinate bidimensionali per lo scatter della PCA
    coords = pca.transform(scaled)
    out["PCA1"], out["PCA2"] = coords[:, 0], coords[:, 1]
    return out


# Vengono caricati dati e modelli (l'operazione viene eseguita a ogni interazione, velocizzata grazie alla cache)
base_df = load_data()
scaler, kmeans, pca, mapping, meta = load_models()

# Il nome della colonna della categoria viene individuato automaticamente (sia con preferred_category sia con top_category)
CAT_COL = next((c for c in ["preferred_category", "top_category", "categoria"] if c in base_df.columns), "preferred_category")

# In questa lista vengono conservati i clienti aggiunti durante la sessione.
# session_state mantiene i valori tra un click e l'altro, perché Streamlit riesegue l'intero script a ogni interazione
if "new_customers" not in st.session_state:
    st.session_state.new_customers = []

# I dati visualizzati corrispondono a quelli originali più gli eventuali nuovi clienti inseriti dall'utente
df = base_df
if st.session_state.new_customers:
    df = pd.concat([base_df, pd.DataFrame(st.session_state.new_customers)], ignore_index=True)

#  Titolo  #
st.title("📊 Dashboard")
st.markdown(
    "Prototipo di clustering RFM con dashboard per il marketing"
    f"con **K-Means (K = {meta['k']})** su metriche RFM (Recency, Frequency, Monetary). "
)

#  Filtri  #
st.sidebar.header("🔍 Filtri interattivi")

# Filtro per periodo. Con un intervallo di date, date_input restituisce una tupla solo quando
# l'utente ha selezionato sia la data di inizio sia quella di fine, quindi il controllo viene effettuato successivamente nel codice
min_date, max_date = df["signup_date"].min().date(), df["signup_date"].max().date()
dates = st.sidebar.date_input("Periodo di iscrizione", (min_date, max_date), min_value=min_date, max_value=max_date)
if not isinstance(dates, (tuple, list)) or len(dates) != 2:
    st.info("Seleziona sia la data di inizio sia quella di fine nel filtro 'Periodo di iscrizione'.")
    st.stop()
start_date, end_date = dates

selected_region = st.sidebar.selectbox("Zona geografica", ["Tutte"] + sorted(df["region"].unique()))
selected_category = st.sidebar.selectbox("Categoria di servizio preferita", ["Tutte"] + sorted(df[CAT_COL].unique()))

# I filtri vengono applicati uno alla volta: prima il periodo, poi la zona e la categoria
filtered_df = df[(df["signup_date"].dt.date >= start_date) & (df["signup_date"].dt.date <= end_date)]
if selected_region != "Tutte":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]
if selected_category != "Tutte":
    filtered_df = filtered_df[filtered_df[CAT_COL] == selected_category]

# Se i filtri non lasciano nessun cliente viene mostrato un avviso, in modo che i grafici non riportino errori
if filtered_df.empty:
    st.warning("⚠️ Nessun cliente corrisponde ai filtri selezionati.")
    st.stop()

#  KPI  #
# Indicatori riassuntivi sui clienti filtrati, uno per ogni colonna
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Totale clienti", f"{len(filtered_df)}")
c2.metric("Spesa totale media", f"€ {filtered_df['total_spent'].mean():.2f}")
c3.metric("Importo medio/richiesta", f"€ {filtered_df['monetary_avg'].mean():.2f}")
c4.metric("Frequenza media", f"{filtered_df['frequency'].mean():.1f} richieste")
c5.metric("Recency media", f"{filtered_df['recency'].mean():.1f} giorni")

st.markdown("---")

# Le quattro schede della dashboard
tab1, tab2, tab3, tab4 = st.tabs(
    ["📌 Analisi dei segmenti", "🔮 Nuovo cliente", "📁 Classificazione da CSV", "📐 Validazione e modelli"]
)

#  TAB 1: analisi dei segmenti  #
with tab1:
    # l'utente sceglie quali due variabili utilizzare sugli assi dello scatter
    ax_col1, ax_col2 = st.columns(2)
    x_axis = ax_col1.selectbox("Asse X", list(AXIS_LABELS), index=0, format_func=AXIS_LABELS.get)
    y_axis = ax_col2.selectbox("Asse Y", list(AXIS_LABELS), index=3, format_func=AXIS_LABELS.get)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        # scatter interattivo: un colore per ogni segmento
        # la dimensione della bolla rappresenta la frequenza
        fig = px.scatter(
            filtered_df, x=x_axis, y=y_axis, color="cluster_label", size="frequency",
            hover_name="customer_id", hover_data=["region", CAT_COL],
            color_discrete_map=COLOR_MAP, labels={**AXIS_LABELS, "cluster_label": "Segmento"},
            title="Segmenti di clientela (dimensione bolla = frequenza)",
        )
        st.plotly_chart(fig, width="stretch")
    with col_b:
        # grafico a ciambella con la quota di clienti di ciascun segmento
        fig_pie = px.pie(filtered_df, names="cluster_label", hole=0.4, color="cluster_label",
                         color_discrete_map=COLOR_MAP, title="Composizione dei segmenti")
        st.plotly_chart(fig_pie, width="stretch")

    st.subheader("📋 Profilo di ciascun segmento")
    # tabella del profilo: per ogni segmento vengono calcolati il numero di clienti e le medie delle variabili (groupby + agg)
    profile = filtered_df.groupby("cluster_label").agg(
        clienti=("customer_id", "count"),
        recency=("recency", "mean"),
        frequenza=("frequency", "mean"),
        importo_medio=("monetary_avg", "mean"),
        spesa_totale=("total_spent", "mean"),
    ).round(2)
    # percentuale di ciascun segmento sul totale dei clienti filtrati
    profile.insert(1, "quota_%", (profile["clienti"] / profile["clienti"].sum() * 100).round(1))
    profile = profile.rename(columns={
        "clienti": "Numero clienti", "quota_%": "Quota %", "recency": "Recency media (gg)",
        "frequenza": "Frequenza media", "importo_medio": "Importo medio (€)", "spesa_totale": "Spesa totale media (€)",
    }).sort_values("Spesa totale media (€)", ascending=False)
    profile.index.name = "Segmento"
    st.dataframe(profile, width="stretch")

    st.subheader("💡 Interpretazione e azioni di marketing")
    # tabella con interpretazione e azione consigliata, limitata ai segmenti effettivamente presenti
    actions = pd.DataFrame([
        {"Segmento": s, "Interpretazione": SEGMENT_INFO[s][0], "Azione consigliata": SEGMENT_INFO[s][1]}
        for s in profile.index if s in SEGMENT_INFO
    ])
    st.table(actions.set_index("Segmento"))

#  TAB 2: nuovo cliente  #
with tab2:
    st.subheader("Classifica un nuovo cliente")
    st.caption("Il cliente viene assegnato con il modello già addestrato (nessun riaddestramento).")

    # Il messaggio di conferma viene salvato prima del re-run, poi mostrato e infine eliminato
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))

    # viene utilizzato un form in modo che Streamlit non riesegua lo script a ogni modifica di un campo
    with st.form("new_customer_form"):
        f1, f2, f3 = st.columns(3)
        rec = f1.number_input("Recency (giorni dall'ultima richiesta)", 1, 365, 15)
        freq = f2.number_input("Frequenza (n. richieste)", 1, 100, 10)
        avg = f3.number_input("Importo medio per richiesta (€)", 1.0, 1000.0, 90.0)
        g1, g2, g3 = st.columns(3)
        reg = g1.selectbox("Zona", REGIONS)
        cat = g2.selectbox("Categoria preferita", CATEGORIES)
        signup = g3.date_input("Data di iscrizione", value=max_date, max_value=max(date.today(), max_date))
        add_to_dash = st.checkbox("Aggiungi il cliente ai dati della dashboard (solo per questa sessione)", value=True)
        submitted = st.form_submit_button("Assegna segmento")

    if submitted:
        # viene costruita una tabella con una sola riga, nello stesso formato utilizzato nel training
        row = pd.DataFrame([{
            "customer_id": f"NEW-{len(st.session_state.new_customers) + 1:03d}",
            "signup_date": pd.Timestamp(signup), "region": reg, CAT_COL: cat,
            "recency": rec, "frequency": freq, "monetary_avg": avg, "total_spent": round(freq * avg, 2),
        }])
        result = classify(row).iloc[0]  # prima e unica riga del risultato
        label = result["cluster_label"]
        msg = f"🎯 Il cliente appartiene al segmento **{label}**. Azione consigliata: {SEGMENT_INFO.get(label, ('', ''))[1]}"
        if add_to_dash:
            st.session_state.new_customers.append(result.to_dict())
            st.session_state.flash = msg + " Cliente aggiunto alla dashboard."
            st.rerun()
        else:
            st.success(msg)

    # pulsante per annullare gli inserimenti effettuati durante questa sessione
    if st.session_state.new_customers and st.button("Rimuovi i clienti aggiunti in questa sessione"):
        st.session_state.new_customers = []
        st.rerun()

#  TAB 3: batch CSV  #
with tab3:
    st.subheader("Classificazione di più clienti da file CSV")
    st.caption("Colonne richieste: recency, frequency, monetary_avg (importo medio per richiesta).")
    uploaded = st.file_uploader("Carica un CSV", type=["csv"])
    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as exc:  #se il file non può essere letto
            st.error(f"Impossibile leggere il file: {exc}")
            batch = None
        if batch is not None:
            missing_cols = [c for c in FEATURES if c not in batch.columns]
            if missing_cols:
                st.error(f"Colonne mancanti: {', '.join(missing_cols)}")
            else:
                # le colonne vengono convertite in numeri: i valori non convertibili diventano NaN e vengono scartati
                num = batch[FEATURES].apply(pd.to_numeric, errors="coerce")
                bad = num.isna().any(axis=1)
                if bad.any():
                    st.warning(f"{int(bad.sum())} righe con valori mancanti o non numerici sono state escluse.")
                batch = batch[~bad].copy()
                batch[FEATURES] = num[~bad]
                if batch.empty:
                    st.error("Nessuna riga valida da classificare.")
                else:
                    # tutto il file viene classificato e viene data la possibilità di scaricare il risultato
                    out = classify(batch).drop(columns=["PCA1", "PCA2"])
                    st.dataframe(out, width="stretch")
                    st.download_button("Scarica i risultati (CSV)", out.to_csv(index=False).encode("utf-8"),
                                       "clienti_segmentati.csv", "text/csv")

#  TAB 4: validazione  #
with tab4:
    st.subheader("Scelta del numero di cluster")
    st.info(meta["motivazione_k"])
    # grafico gomito/silhouette e tabella di confronto dei valori di K, già prodotti dalla pipeline
    img_col, tab_col = st.columns([3, 2])
    if (REPORTS_DIR / "evaluation_k.png").exists():
        img_col.image(str(REPORTS_DIR / "evaluation_k.png"))
    if (REPORTS_DIR / "k_comparison.csv").exists():
        tab_col.dataframe(pd.read_csv(REPORTS_DIR / "k_comparison.csv"), width="stretch", hide_index=True)

    st.subheader("Confronto tra modelli (K-Means vs Gaussian Mixture)")
    if (REPORTS_DIR / "model_metrics.csv").exists():
        st.dataframe(pd.read_csv(REPORTS_DIR / "model_metrics.csv"), width="stretch", hide_index=True)
        st.caption("Silhouette: più alta è meglio. Davies-Bouldin: più bassa è meglio. "
                   "ARI: accordo con i profili reali usati per generare i dati (1 = perfetto).")
    st.metric("Stabilità del K-Means (ARI medio su 20 seed)", f"{meta['stabilita_ari_media']:.3f}",
              delta=f"± {meta['stabilita_ari_std']:.3f}", delta_color="off")

    if (REPORTS_DIR / "confusion_matrix.csv").exists():
        st.subheader("Profilo reale vs segmento individuato")
        # heatmap: righe = profilo reale, colonne = segmento individuato (sulla diagonale si trovano i cluster "corretti")
        cm = pd.read_csv(REPORTS_DIR / "confusion_matrix.csv", index_col=0)
        fig_cm = px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                           labels={"x": "Segmento individuato", "y": "Profilo reale", "color": "Clienti"})
        st.plotly_chart(fig_cm, width="stretch")

    # PCA: riduce le 3 feature a 2 componenti in modo che i cluster possano essere rappresentati su un piano.
    # Nel titolo viene riportata la varianza spiegata dalle due componenti
    var = pca.explained_variance_ratio_
    fig_pca = px.scatter(
        filtered_df, x="PCA1", y="PCA2", color="cluster_label", hover_name="customer_id",
        color_discrete_map=COLOR_MAP, labels={"cluster_label": "Segmento"},
        title=f"Proiezione PCA 2D (varianza spiegata: {var[0] * 100:.1f}% + {var[1] * 100:.1f}%)",
    )
    st.plotly_chart(fig_pca, width="stretch")
