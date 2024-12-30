# ==== IMPORTS ====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==== INDIVIDUAZIONE DEGLI OUTLIERS =====
def highlight_outliers(df):
    outliers_number = {} # Questo dizionario conterrà il numero di outliers per Features
    outliers_val = {} # Questo dizionario conterrà il valore degli outliers per Features

    for col in df.columns[:-1]:
        # Quartili per Target = 0
        Q1_0 = df[df['Bancarotta'] == 0][col].quantile(0.25)
        Q3_0 = df[df['Bancarotta'] == 0][col].quantile(0.75)

        # Intervallo Interquartile per Target = 0
        IQR_0 = Q3_0 - Q1_0

        # Limite inferiore e superiore per Target = 0
        lower_bound_0 = Q1_0 - 1.5 * IQR_0 
        upper_bound_0 = Q3_0 + 1.5 * IQR_0

        # Quartili per Target = 1
        Q1_1 = df[df['Bancarotta'] == 1][col].quantile(0.25)
        Q3_1 = df[df['Bancarotta'] == 1][col].quantile(0.75)

        # Intervallo Interquartile per Target = 1
        IQR_1 = Q3_1 - Q1_1

        # Limite inferiore e superiore per Target = 1
        lower_bound_1 = Q1_1 - 1.5 * IQR_1
        upper_bound_1 = Q3_1 + 1.5 * IQR_1

        # Individuo gli outliers per Target = 0 e li salvo nel dizionario
        outliers_0 = [val for val in df[df['Bancarotta'] == 0][col] if val < lower_bound_0 or val > upper_bound_0]
        outliers_number[f'{col}_0'] = len(outliers_0) # Salvo il numero di outliers per colonna, con Target = 0
        outliers_val[f'{col}_0'] = outliers_0 # Salvo il valore degli outliers per colonna, con Target = 0


        # Individuo gli outliers per Target = 1 e li salvo nel dizionario
        outliers_1 = [val for val in df[df['Bancarotta'] == 1][col] if val < lower_bound_1 or val > upper_bound_1]
        outliers_number[f'{col}_1'] = len(outliers_1) # Salvo il numero di outliers per colonna, con Target = 1
        outliers_val[f'{col}_1'] = outliers_1 # Salvo il valore degli outliers per colonna, con Target = 1

    # Output
    i = 0
    print("Numero di Outliers per Features\n")
    for val in outliers_number:
        if(i % 2 == 0):
            print(f'{val}: {outliers_number.get(val)}')
        else:
            print(f'{val}: {outliers_number.get(val)}')
            print("\n")
        i += 1

    return outliers_number

# ==== CREAZIONE DEI DATAFRAME PER GLI OUTLIERS DISTINTI PER CLASSE DEL TARGET ====
def outliers_to_df (outliers_number):
    # Creo prima i dataframe funzionali alla visualizzazione grafica
    outliers_0_table = pd.DataFrame(columns = ['Features', 'Outliers']) # Inizializzo la tabella che contiene gli outliers per Target = 0
    outliers_1_table = pd.DataFrame(columns = ['Features', 'Outliers']) # Inizializzo la tabella che contiene gli outliers per Target = 1

    index_0 = 0 # Indici outliers per Target = 0
    index_1 = 0 # Indici outliers per Target = 1

    for key in outliers_number.keys(): # Itero sul dizionario costruito precedentemente
        feature = key[:-2] # Salvo nome della colonna escludendo gli ultimi 2 caratteri (_0/_1)
        value = outliers_number.get(key) # Salvo il numero di outliers per la colonna iterata
        if(key.endswith('0')): # Filtro in base alla classe del Target e riempio di valori i rispettivi dataset
            outliers_0_table.loc[index_0] = [feature, value] 
            index_0 += 1
        else:
            outliers_1_table.loc[index_1] = [feature, value]
            index_1 += 1

    return outliers_0_table, outliers_1_table 

# ==== RIMOZIONE DEGLI OUTLIERS ====
# Tecnica della Deviazione Standard
def remove_outliers (df, df_clean, features_with_outliers):
    k = 3 # Coefficiente di distanza 
    outliers_number = 0 # Inizializzo il numero di outliers

    for col in features_with_outliers: # Itero sulle features con outliers estremi

        # Media e deviazione standard
        mean = df[col].mean() 
        std_dev = df[col].std()

        # Limite inferiore e superiore
        lower_bound = mean - k * std_dev 
        upper_bound = mean + k * std_dev

        # Salvo gli outliers
        outliers = [val for val in df[col] if val > upper_bound or val < lower_bound]

        # Tengo il conto del numero degli outliers rilevati
        outliers_number += len(outliers) 

        # Pulisco il dataset eliminando gli outliers rilevati
        for i in range(len(outliers)):
            df_clean = df_clean[~(df_clean[col] == outliers[i])]

    print(f'Outliers individuati: {outliers_number}')
    print(f'Osservazioni rimosse: {df.shape[0] - df_clean.shape[0]}')

    return df_clean



