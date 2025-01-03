# ==== IMPORTS =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

# ==== COEFFICIENTE DI VARIAZIONE =====
# Calcolo del coefficiente di variazione per ogni Features
def calculate_coeff_var(df_clean):
     # Inizializzo una lista vuota che conterrà i coefficienti di variazione per ogni Features
    coeff_var = []

    for col in df_clean.columns:
         # Calcolo la media della distribuzione numerica della Features iterata
        mean = np.mean(df_clean[col])

         # Calcolo la deviazione standard della distribuzione numerica della Features iterata
        std_dev = np.std(df_clean[col])

         # Calcolo il coefficiente di variazione della distribuzione numerica della Features iterata
        coeff_var.append((std_dev / mean) * 100)

    return coeff_var

# Selezione delle Features con coefficiente di variazione maggiore del 50%
def calculate_high_coeff_var(df_clean):
    # Inizializzo una dizionario vuoto che conterrà i nomi delle Features con elevato coefficiente di variazione
    high_cv_features = {
    "Features": [],
    "Coefficiente di Variazione (%)": []
    }

     # Procedo a calcolare il coefficiente di variazione per ogni Features
    cv = calculate_coeff_var(df_clean)

    for i in range(len(cv)):

        # Stampo il nome delle Features che hanno un coefficiente di variazione > 50% e il relativo valore
        if(cv[i] > 50): 
            high_cv_feature = df_clean.columns[i] 
            print(f'{high_cv_feature}: {round(cv[i], 2)}%')

            # Salvo nel dizionario la Feature che ha un coefficiente di variazione > 50% 
            high_cv_features["Features"].append(high_cv_feature)
            high_cv_features["Coefficiente di Variazione (%)"].append(round(cv[i], 2))

    high_cv_features_df = pd.DataFrame(high_cv_features) # Trasformo il dizionario in un dataframe

    return   high_cv_features_df.sort_values(by = 'Coefficiente di Variazione (%)', ascending = False)

# ==== SQRT TRANSFORMATION ====
def sqrt_transformation(X_train_smote, X_test, cols_to_transform):

     # Inizializzo un dataset per le Features di training identico a quello con SMOTE, che conterrà le Features trasformate con la radice quadrata
    X_train_sqrt = X_train_smote.copy()

     # Inizializzo un dataset per le Features di test identico a quello con SMOTE, che conterrà le Features trasformate con la radice quadrata
    X_test_sqrt = X_test.copy()
    
    # Aggiungo un offset per i valori della distribuzione che sono pari a 0
    offset = 1

    # Prcedo alla trasformazione
    X_train_sqrt[cols_to_transform] = np.sqrt(X_train_sqrt[cols_to_transform]) + offset
    X_test_sqrt[cols_to_transform] = np.sqrt(X_test_sqrt[cols_to_transform]) + offset

    return X_train_sqrt, X_test_sqrt

# ==== ROBUST SCALER =====
def robust_scaler(X_train_smote, X_test, cols_to_transform):

    # Inizializzo un dataset per le Features di training identico a quello con SMOTE, che conterrà le Features scalate con RobustScaler
    X_train_norm = X_train_smote.copy()

    # Inizializzo un dataset per le Features di test identico a quello con SMOTE, che conterrà le Features scalate con RobustScaler
    X_test_norm = X_test.copy() 

     # Inizializzo lo Scaler
    scaler = RobustScaler()

    # Procedo a scalare i valori
    X_train_norm[cols_to_transform] = scaler.fit_transform(X_train_norm[cols_to_transform])
    X_test_norm[cols_to_transform] = scaler.fit_transform(X_test_norm[cols_to_transform])

    return X_train_norm, X_test_norm
    
# ==== STANDARD SCALER ====
def standard_scaler(X_train_smote, X_test):
    # Inizializzo un dataset per le Features di Train che conterrà le Features scalate con StandardScaler
    X_train_norm = X_train_smote.copy()

    # Inizializzo un dataset per le Features di test che conterrà le Features scalate con StandardScaler
    X_test_norm = X_test.copy() 

    # Inizializzo lo Scaler
    scaler = StandardScaler()

    # Procedo a scalare i valori: fit sui dati di train e transform sui dati di test
    X_train_norm[X_train_norm.columns] = scaler.fit_transform(X_train_norm[X_train_norm.columns])
    X_test_norm[X_test_norm.columns] = scaler.transform(X_test_norm[X_test_norm.columns])

    return X_train_norm, X_test_norm
