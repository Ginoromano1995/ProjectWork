import random
import pandas as pd

# Impostiamo un seed per rendere i dati generati perfettamente riproducibili
random.seed(42)

def generate_synthetic_customers(num_customers=400):
    """
    Genera un dataset sintetico di clienti per la startup digitale di servizi on-demand.
    Metriche simulate:
    - customer_id: Identificativo unico
    - region: Area geografica (Nord, Centro, Sud, Isole)
    - signup_days_ago: Anzianità dell'account in giorni
    - frequency: Numero totale di richieste effettuate nell'ultimo periodo
    - monetary: Importo medio speso per ciascuna richiesta (in Euro)
    - recency: Giorni trascorsi dall'ultima richiesta effettuata
    - preferred_category: Categoria di servizio preferita
    """
    
    regions = ['Nord', 'Centro', 'Sud', 'Isole']
    categories = ['Manutenzione Casa', 'Pulizia', 'Consulenza IT', 'Logistica Rapida']
    
    data = []
    
    for i in range(1, num_customers + 1):
        customer_id = f"CUST-{i:04d}"
        region = random.choice(regions)
        signup_days_ago = random.randint(30, 730) # Iscritti negli ultimi 2 anni
        
        # Creazione di profili comportamentali realistici (es. clienti attivi, occasionali, inattivi)
        profile_type = random.choices(
            ['Active_Frequent', 'Occasional', 'At_Risk', 'New'], 
            weights=[0.3, 0.4, 0.2, 0.1], 
            k=1
        )[0]
        
        if profile_type == 'Active_Frequent':
            frequency = random.randint(12, 30)
            monetary = round(random.uniform(70.0, 250.0), 2)
            recency = random.randint(1, 10)
        elif profile_type == 'Occasional':
            frequency = random.randint(4, 11)
            monetary = round(random.uniform(40.0, 120.0), 2)
            recency = random.randint(11, 45)
        elif profile_type == 'At_Risk':
            frequency = random.randint(5, 20)
            monetary = round(random.uniform(60.0, 200.0), 2)
            recency = random.randint(60, 180) # Tanto tempo dall'ultima richiesta
        else: # New
            frequency = random.randint(1, 3)
            monetary = round(random.uniform(30.0, 90.0), 2)
            recency = random.randint(1, 15)
            
        preferred_category = random.choice(categories)
        
        data.append({
            'customer_id': customer_id,
            'region': region,
            'signup_days_ago': signup_days_ago,
            'frequency': frequency,
            'monetary': monetary,
            'recency': recency,
            'preferred_category': preferred_category
        })
        
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generazione del dataset sintetico in corso...")
    df_customers = generate_synthetic_customers(450) # Generiamo 450 record (nel range 300-600 richiesto)
    
    # Salvataggio del file CSV nella cartella data/
    output_path = "data/raw_customers.csv"
    df_customers.to_csv(output_path, index=False)
    
    print(f"Dataset generato con successo! Salvato in: {output_path}")
    print("\nPrime 5 righe del dataset:")
    print(df_customers.head())