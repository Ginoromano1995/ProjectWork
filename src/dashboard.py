import os
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# Caricamento Dati e Modelli
@st.cache_data
def load_data():
    if not os.path.exists("data/processed_customers.csv"):
        st.error("Esegui prima la pipeline con 'python run_all.py'")
        st.stop()
    return pd.read_csv("data/processed_customers.csv")

@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler.pkl")
    kmeans = joblib.load("models/kmeans.pkl")
    mapping = joblib.load("models/label_mapping.pkl")
    return scaler, kmeans, mapping

df = load_data()
scaler, kmeans, mapping = load_models()

st.title("📊 Dashboard Analisi Segmentazione Clienti (RFM)")

# --- SIDEBAR FILTRI ---
st.sidebar.header("Filtri Dati")

# Filtro Data
df["signup_date"] = pd.to_datetime(df["signup_date"])
min_date, max_date = df["signup_date"].min().date(), df["signup_date"].max().date()

start_date, end_date = st.sidebar.date_input("Intervallo Data Iscrizione", [min_date, max_date])

# Filtro Regione
regions = ["Tutte"] + list(df["region"].unique())
selected_region = st.sidebar.selectbox("Regione", regions)

# Applicazione Filtri
filtered_df = df[(df["signup_date"].dt.date >= start_date) & (df["signup_date"].dt.date <= end_date)]
if selected_region != "Tutte":
    filtered_df = filtered_df[filtered_df["region"] == selected_region]

# Controllo Filtri Vuoti
if filtered_df.empty:
    st.warning("⚠️ Nessun cliente trovato con i filtri selezionati.")
    st.stop()

# --- KPI PRINCIPALI ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Totale Clienti", len(filtered_df))
col2.metric("Spesa Media (€)", f"{filtered_df['total_spent'].mean():.2f}")
col3.metric("Recency Media (gg)", f"{filtered_df['recency'].mean():.1f}")
col4.metric("Frequenza Media", f"{filtered_df['frequency'].mean():.1f}")

st.markdown("---")

# --- TAB DASHBOARD ---
tab1, tab2, tab3, tab4 = st.tabs(["📌 Analisi Cluster", "🔮 Predizione Nuovo Cliente", "📁 Batch Upload CSV", "📐 Validazione & Modelli"])

# TAB 1: ANALISI CLUSTER
with tab1:
    st.subheader("Distribuzione dei Segmenti")
    
    # Fix Plotly: colori discreti con cluster_label (stringa)
    fig_scatter = px.scatter(
        filtered_df,
        x="recency",
        y="total_spent",
        size="frequency",
        color="cluster_label",
        hover_name="customer_id",
        title="Segmenti RFM: Recency vs Spesa Totale",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie = px.pie(filtered_df, names="cluster_label", title="Proporzione Segmenti", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_b:
        st.subheader("💡 Azioni di Marketing Consigliate")
        actions = pd.DataFrame([
            {"Segmento": "Campioni", "Azione": "Programmi VIP, anteprime esclusive, upsell di prodotto."},
            {"Segmento": "Fedeli", "Azione": "Programmi fedeltà, incentivi a recensioni, cross-selling."},
            {"Segmento": "A Rischio", "Azione": "Campagne Win-back, sconti personalizzati via email urgente."},
            {"Segmento": "Nuovi", "Azione": "Onboarding guidato, coupon di benvenuto per secondo acquisto."}
        ])
        st.table(actions)

# TAB 2: PREDIZIONE NUOVO CLIENTE
with tab2:
    st.subheader("Classifica un Nuovo Cliente")
    with st.form("new_customer_form"):
        c1, c2, c3 = st.columns(3)
        rec = c1.number_input("Recency (giorni dall'ultimo acquisto)", 1, 365, 15)
        freq = c2.number_input("Frequenza (numero ordini)", 1, 100, 5)
        spent = c3.number_input("Spesa Totale (€)", 1.0, 10000.0, 150.0)
        
        reg = st.selectbox("Regione", ["Nord", "Centro", "Sud", "Isole"])
        cat = st.selectbox("Categoria Principale", ["Elettronica", "Abbigliamento", "Casa", "Sport"])
        
        submitted = st.form_submit_button("Classifica Cliente")
        
    if submitted:
        # Preprocessing input e Predict senza riaddestrare
        input_data = pd.DataFrame([[rec, freq, spent]], columns=["recency", "frequency", "total_spent"])
        scaled_input = scaler.transform(input_data)
        cluster_id = kmeans.predict(scaled_input)[0]
        label = mapping.get(cluster_id, f"Cluster {cluster_id}")
        
        st.success(f"🎉 Il nuovo cliente appartiene al segmento: **{label}**")

# TAB 3: BATCH UPLOAD CSV
with tab3:
    st.subheader("Classificazione Batch da CSV")
    uploaded_file = st.file_uploader("Carica un CSV con le colonne: recency, frequency, total_spent", type=["csv"])
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        if set(["recency", "frequency", "total_spent"]).issubset(batch_df.columns):
            scaled_batch = scaler.transform(batch_df[["recency", "frequency", "total_spent"]])
            batch_df["cluster"] = kmeans.predict(scaled_batch)
            batch_df["cluster_label"] = batch_df["cluster"].map(mapping)
            st.dataframe(batch_df)
            st.download_button("Scarica Risultati", batch_df.to_csv(index=False), "clienti_segmentati.csv")
        else:
            st.error("Il file CSV deve contenere le colonne: recency, frequency, total_spent")

# TAB 4: VALIDAZIONE E METRICHE AVANZATE
with tab4:
    st.subheader("Visualizzazione PCA 2D e Performance Modelli")
    
    # PCA 2D
    features = ["recency", "frequency", "total_spent"]
    scaled_data = scaler.transform(filtered_df[features])
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(scaled_data)
    
    filtered_df["PCA1"] = pca_coords[:, 0]
    filtered_df["PCA2"] = pca_coords[:, 1]
    
    fig_pca = px.scatter(
        filtered_df, x="PCA1", y="PCA2", color="cluster_label",
        title="Proiezione PCA 2D dei Cluster",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    
    if os.path.exists("reports/model_metrics.csv"):
        st.subheader("Confronto Modelli (ARI & Silhouette)")
        metrics_df = pd.read_csv("reports/model_metrics.csv")
        st.table(metrics_df)