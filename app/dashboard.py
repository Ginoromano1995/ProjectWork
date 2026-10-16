import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Configurazione della pagina Streamlit
st.set_page_config(
    page_title="Dashboard Segmentazione Clienti - PW 28", page_icon="📊", layout="wide"
)


@st.cache_data
def load_data():
  csv_path = "data/clustered_customers.csv"
  if not os.path.exists(csv_path):
    st.error(
        f"File {csv_path} non trovato. Esegui prima i due script di generazione"
        " e clustering!"
    )
    st.stop()
  df = pd.read_csv(csv_path)
  return df


base_df = load_data()

# Inizializziamo lo stato della sessione per la gestione dinamica dei clienti
if "dynamic_df" not in st.session_state:
  st.session_state.dynamic_df = base_df.copy()

df = st.session_state.dynamic_df

# Titolo e Contesto
st.title("📊 Dashboard di Segmentazione Automatica Clienti")
st.markdown("""
**Prototipo di Business Intelligence & Machine Learning Non Supervisionato (PW 28)**  
*Analisi comportamentale dei clienti tramite algoritmo K-Means basato su metriche RFM (Recency, Frequency, Monetary).*
""")

# Sidebar per Filtri
st.sidebar.header("🔍 Filtri Interattivi")

# Filtro per Zona
all_regions = ["Tutte"] + list(df["region"].unique())
selected_region = st.sidebar.selectbox("Area Geografica:", all_regions)

# Filtro per Categoria Preferita
all_categories = ["Tutte"] + list(df["preferred_category"].unique())
selected_category = st.sidebar.selectbox("Categoria Preferita:", all_categories)

# Filtro per Intervallo di Anzianità Account (giorni)
min_days, max_days = int(df["signup_days_ago"].min()), int(
    df["signup_days_ago"].max()
)
selected_days = st.sidebar.slider(
    "Anzianità Account (Giorni):", min_days, max_days, (min_days, max_days)
)

# Applicazione dei Filtri
filtered_df = df.copy()
if selected_region != "Tutte":
  filtered_df = filtered_df[filtered_df["region"] == selected_region]
if selected_category != "Tutte":
  filtered_df = filtered_df[
      filtered_df["preferred_category"] == selected_category
  ]
filtered_df = filtered_df[
    (filtered_df["signup_days_ago"] >= selected_days[0])
    & (filtered_df["signup_days_ago"] <= selected_days[1])
]

# KPI Principali
st.subheader("📌 Metriche Generali (Dati Filtrati)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Totale Clienti", f"{len(filtered_df)}")
col2.metric("Spesa Media", f"€ {filtered_df['monetary'].mean():.2f}")
col3.metric("Frequenza Media", f"{filtered_df['frequency'].mean():.1f} ordini")
col4.metric("Recency Media", f"{filtered_df['recency'].mean():.1f} giorni")

st.markdown("---")

# Sezione 1: Grafico a Dispersione (Scatter Plot)
st.subheader("📈 Analisi Visiva dei Cluster")
col_chart1, col_chart2 = st.columns([2, 1])

with col_chart1:
  fig_scatter = px.scatter(
      filtered_df,
      x="frequency",
      y="monetary",
      color="cluster",
      size="recency",
      hover_data=["customer_id", "preferred_category", "region"],
      labels={
          "frequency": "Frequenza (N. Ordini)",
          "monetary": "Importo Medio Speso (€)",
          "cluster": "Cluster ID",
          "recency": "Recency (Giorni)",
      },
      title="Dispersione Frequenza vs Spesa (Dimensione bolla = Recency in giorni)",
  )
  st.plotly_chart(fig_scatter, width="stretch")

with col_chart2:
  cluster_counts = filtered_df["cluster"].value_counts().reset_index()
  cluster_counts.columns = ["Cluster", "Conteggio"]
  fig_pie = px.pie(
      cluster_counts,
      values="Conteggio",
      names="Cluster",
      title="Distribuzione % dei Cluster",
      hole=0.4,
  )
  st.plotly_chart(fig_pie, width="stretch")

# Sezione 2: Profili dei Cluster
st.subheader("📋 Profilo Medio per Ciascun Cluster")

rfm_cols = ["recency", "frequency", "monetary"]
summary_df = filtered_df.groupby("cluster")[rfm_cols].mean().round(2)
summary_df["Numero Clienti"] = filtered_df.groupby("cluster").size()
summary_df.columns = [
    "Recency Media (gg)",
    "Frequenza Media (n°)",
    "Spesa Media (€)",
    "Numero Clienti",
]

st.dataframe(summary_df, width="stretch")

st.markdown("---")

# Sezione 3: Classifica e Aggiungi un Nuovo Cliente
st.subheader("🔮 Classifica e Aggiungi un Nuovo Cliente")
st.markdown(
    "Inserisci le caratteristiche di un nuovo cliente per calcolarne il gruppo"
    " di appartenenza e aggiornare la dashboard in tempo reale:"
)

col_input1, col_input2, col_input3 = st.columns(3)
with col_input1:
  new_recency = st.number_input(
      "Recency (giorni dall'ultimo ordine):", min_value=1, max_value=365, value=10
  )
with col_input2:
  new_frequency = st.number_input(
      "Frequenza (numero di ordini fatti):", min_value=1, max_value=100, value=15
  )
with col_input3:
  new_monetary = st.number_input(
      "Importo Medio Speso (€):", min_value=1.0, max_value=1000.0, value=120.0
  )

if st.button("Assegna Cluster e Aggiorna Dataset"):
  # 1. Prepariamo lo scaler con i nomi delle colonne per evitare i warning di sklearn
  scaler = StandardScaler()
  X_all = scaler.fit_transform(st.session_state.dynamic_df[rfm_cols])

  kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
  kmeans.fit(X_all)

  # 2. Prepariamo il DataFrame con il nuovo cliente per mantenere la coerenza dei nomi delle feature
  new_data_df = pd.DataFrame(
      [[new_recency, new_frequency, new_monetary]], columns=rfm_cols
  )
  new_data_scaled = scaler.transform(new_data_df)

  predicted_cluster = kmeans.predict(new_data_scaled)[0]

  # 3. Creiamo la nuova riga da aggiungere allo stato dinamico
  new_customer_row = {
      "customer_id": f"NEW-{len(st.session_state.dynamic_df)+1}",
      "region": selected_region if selected_region != "Tutte" else "Nord",
      "signup_days_ago": 5,
      "frequency": new_frequency,
      "monetary": new_monetary,
      "recency": new_recency,
      "preferred_category": (
          selected_category if selected_category != "Tutte" else "Consulenza IT"
      ),
      "cluster": int(predicted_cluster),
  }

  # Aggiorniamo il dataset in sessione
  st.session_state.dynamic_df = pd.concat(
      [
          st.session_state.dynamic_df,
          pd.DataFrame([new_customer_row]),
      ],
      ignore_index=True,
  )

  st.success(
      f"🎯 Il cliente è stato classificato nel **Cluster"
      f" {predicted_cluster}** e aggiunto con successo alla dashboard!"
  )
  st.rerun()