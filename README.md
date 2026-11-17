# PW 28 - Segmentazione automatica della clientela

Il progetto consiste in un prototipo di segmentazione dei clienti tramite Machine Learning non supervisionato
(K-Means su metriche RFM) e in una dashboard interattiva realizzata con Streamlit.
Tutti i dati utilizzati sono **sintetici**, quindi non sono presenti informazioni personali.

## Esecuzione rapida

'''bash
python run_all.py
'''

Lo script crea l'ambiente virtuale 'venv', installa le dipendenze elencate in 'requirements.txt', genera i dati,
esegue il clustering e avvia la dashboard. Se dati e modelli sono già presenti, la fase di training viene saltata.

In alternativa a 'run_all.py' è possibile eseguire manualmente i singoli passaggi: 'python src/data_generator.py', 'python src/clustering_pipeline.py', 'streamlit run src/dashboard.py'.

## Struttura del progetto

**File**: Funzione

**'src/data_generator.py'**: Genera 'data/raw_customers.csv' (con valori mancanti, outlier e duplicati) e 'data/true_profiles.csv' (utilizzato solo per la validazione)
**'src/clustering_pipeline.py'**: Esegue pulizia, standardizzazione, scelta di K, K-Means, confronto con GMM, validazione e salvataggio dei modelli
**src/dashboard.py'**: Dashboard con filtri, scatter, profili dei segmenti, classificazione di un nuovo cliente, caricamento di un file CSV e validazione
**'run_all.py'**: Crea il venv, installa i pacchetti, esegue la pipeline e avvia la dashboard

File prodotti: 
'data/processed_customers.csv' (dataset con i cluster);
'models/' (scaler, K-Means, PCA, etichette);
'reports/' (grafico gomito/silhouette, confronto tra i valori di K, metriche, matrice di confusione, stabilità).

## Scelte tecniche

- **Feature**: recency, frequency e importo medio per richiesta ('monetary_avg'). Il totale speso non viene inserito nel modello perché corrisponde a frequenza per importo medio (collinearità).
- **Pulizia**: i duplicati vengono rimossi; l'importo medio mancante viene ricostruito dal rapporto tra totale e frequenza, gli altri valori mancanti vengono sostituiti con la mediana; gli outlier dell'importo medio vengono limitati al 99° percentile.
- **Normalizzazione**: StandardScaler (z-score).
- **Scelta di K**: viene selezionata la silhouette massima tra K=3 e K=5 (intervallo richiesto dalla traccia) e il risultato viene verificato tramite il grafico del gomito. Tutti i valori di K da 2 a 7 sono riportati in 'reports/k_comparison.csv'.
- **Etichette dei segmenti**: vengono assegnate automaticamente tramite una classifica dei centroidi (Campioni, A Rischio, Nuovi, Fedeli, Occasionali), in modo che ogni nome sia univoco.
- **Nuovi clienti**: vengono classificati con lo scaler e il modello già salvati, senza alcun riaddestramento.
- **Validazione**: silhouette, Davies-Bouldin, ARI rispetto ai profili utilizzati per generare i dati, stabilità su 20 seed e confronto con il Gaussian Mixture.
- **Riproducibilità**: sono stati utilizzati seed fissi ('random_state=42') e una data di riferimento fissa per le date di iscrizione.

## Limiti

I dati sono generati a partire da profili noti, quindi i cluster vengono individuati "per costruzione": con dati reali i risultati sarebbero meno netti. Il K-Means assume cluster compatti ed è sensibile agli outlier, zona e categoria non vengono inserite nel modello e la segmentazione dovrebbe essere ricalcolata periodicamente.
