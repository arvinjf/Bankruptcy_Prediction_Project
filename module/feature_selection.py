# ==== IMPORTS ====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
 
 # ==== FEATURES SELECTION CON SKB E RFE=====
def feature_selection(type, model, X, y, k):
    if type == 'skb':
        # Inizializzo SKB con la funzione di punteggio f_classif
        skb = SelectKBest(score_func=f_classif, k=k)
        
        # Eseguo il fitting del modello SKB sui dati X e y
        skb = skb.fit(X, y)
        
        # Ottengo le caratteristiche selezionate da SKB (indice delle Features migliori)
        selected_features = X.columns[skb.get_support(indices=True)]
    else:
        # Creo l'oggetto RFE con il modello come stimatore e k come numero di caratteristiche da selezionare
        rfe = RFE(estimator=model, n_features_to_select=k)

        # Eseguo il fitting del modello RFE sui dati X e y
        rfe = rfe.fit(X, y)  

        # Ottengo le caratteristiche selezionate RFE (indice delle Features migliori)
        selected_features = X.columns[rfe.support_]  
    
    # Ritorno X con solo le caratteristiche selezionate
    return X[selected_features]  

# ==== K-FOLD CROSS VALIDATION ===== 
def cross_validation_accuracy(model, X, y, k_fold=5):
    # Creo il K-Fold cross-validator, che suddivide i dati in 5 parti (k_fold = 5), mescola i dati e imposta un seed per la riproducibilità
    cv = KFold(n_splits=k_fold, shuffle=True, random_state=42)

    # Eseguo la cross-validation misurando l'Accuracy
    scores = cross_val_score(
        model, X, y, cv=cv, scoring='accuracy', n_jobs=-1  # n_jobs=-1 per usare tutti i core della CPU
    )
    
    # Restituisco la media dell'Accuracy
    return scores.mean()

# ==== TRAIN ACCURACY =====
# Calcolo dell'accuracy del modello sui dati di Train con le Features selezionate da SKB e RFE
def calculate_accuracy(type, model, X_train_smote, y_train_smote):
    # Calcolo l'accuracy per ogni k features
    accuracies = []  # Lista per memorizzare l'accuracy per ogni valore di k
    new_col = []  # Lista per memorizzare le nuove caratteristiche selezionate

    # Ciclo su tutti i valori di k (numero di caratteristiche selezionate) da 1 al numero totale di caratteristiche nel dataset
    for k in tqdm(range(1, len(X_train_smote.columns)+1)):
        # Seleziono le k caratteristiche migliori utilizzando la funzione di selezione (SKB o RFE)
        X_selected = feature_selection(type, model, X_train_smote, y_train_smote, k)
        
        # Calcolo l'accuracy utilizzando la funzione di cross-validation (con 10 fold)
        accuracy = cross_validation_accuracy(model, X_selected, y_train_smote, 10)
        
        # Aggiungo l'accuracy calcolata alla lista delle accuracies
        accuracies.append(f'{accuracy}')
        
        # Trovo le nuove colonne selezionate (le caratteristiche selezionate che non sono già presenti in skb_new_col)
        new_elements = [i for i in X_selected.columns if i not in new_col]
        
        # Se ci sono nuove caratteristiche selezionate, le aggiungo alla lista delle colonne
        if new_elements:  # Controlla se ci sono nuovi elementi
            new_col.append(new_elements[0])  # Aggiungo solo la prima nuova caratteristica

    # Stampo l'accuracy per ogni k-esima Features
    for i, val in enumerate(accuracies):
        print(f'{new_col[i]}: {val}')

    return accuracies

# ==== FEATURES COMUNI ====
# Selezione delle Features comuni tra SKB e RFE
def select_common_features(skb_selected_features, rfe_selected_features):

    common_selected_features = list(set(skb_selected_features) & set(rfe_selected_features))

    print('Features comuni:')

    for features in common_selected_features:
        print(f'- {features}')

    return common_selected_features

# ==== MATRICE DI CONFUSIONE ====
# Creazione del dataframe che contiene le metriche della matrice di confusione, divisa per SKB e RFE, funzionale alla visualizzazione grafica
def conf_matrix_coefficients_to_df(precision, recall, f1, specificity):
    # Inizializzo un dizionario che contiene i valori del dataframe
    conf_matrix_coefficients_data = {
        'Metriche': ['Precisione', 'Recall', 'F1-score', 'Specificità'] * 2, 
        'Valore': [precision[0], recall[0], f1[0], specificity[0],
                   precision[1], recall[1], f1[1], specificity[1]],
        'Modello': ['SKB'] * 4 + ['RFE'] * 4
    }

    # Trasformo il dizionario in un dataframe
    conf_matrix_coefficients_df = pd.DataFrame(conf_matrix_coefficients_data)

    return conf_matrix_coefficients_df
