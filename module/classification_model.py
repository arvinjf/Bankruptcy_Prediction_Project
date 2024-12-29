# ==== IMPORTS ====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# ==== REGRESSIONE LOGISTICA =====
# Analisi dei coefficienti della Regressione Logistica
def log_reg_coefficients_summary(log_reg_X_train, y_train_smote): 
    # Aggiungo una colonna di bias per l'intercetta
    X_train_with_intercept = sm.add_constant(log_reg_X_train)

    # Inizializzo il modello
    model = sm.Logit(y_train_smote, X_train_with_intercept)
    result = model.fit()

    print(result.summary())

    return result

# Creazione di un dataframe che contiene i coefficienti della Regressione Logistica, funzionale alla visualizzazione grafica
def log_reg_coefficients_df(result):
# Creo un dataframe con coefficienti, p-value ed errori standard
    model_summary_df = pd.DataFrame({
        "Coefficiente": result.params,
        "P-value": result.pvalues,
        "Errore Standard": result.bse,
        "Intervallo di Confidenza Inferiore": result.conf_int()[0],
        "Intervallo di Confidenza Superiore": result.conf_int()[1]
    })

    model_summary_df = model_summary_df.reset_index().rename(columns = {'index':'Features'})

    return model_summary_df

# K-NEAREST NEIGHBORS
# Ottimizzazione degli iperparametri di KNN con GridSearchCV
def tune_knn_hyperparameters(model, model_X_train, y_train_smote):
    # Configuro la griglia di ricerca con Cross-validation
    param_grid = {
        'n_neighbors': np.arange(1, 23),
        'metric': ['manhattan']
        }

    model_cv = GridSearchCV(model, param_grid, cv=5)

    # Eseguo Gird Search per trovare la configurazione ottimale del modello
    model_cv.fit(model_X_train, y_train_smote)

    # Stampo gli iperparametri e la loro accuracy
    print("Best K:", model_cv.best_params_['n_neighbors'])
    print("Best Metric:", model_cv.best_params_['metric'])
    print("Best Accuracy:", model_cv.best_score_)

    return model_cv

# ==== DECISION TREE ====
# Ottimizzazione della potatura dell'albero decisionale con cross-validation (Post-pruning x K-Fold Cross-validation)
def tune_decision_tree_pruning(dt_model, dt_X_train, y_train_smote):
    path = dt_model.cost_complexity_pruning_path(dt_X_train, y_train_smote)
    ccp_alphas = path.ccp_alphas

    best_model = None
    best_score = 0
    for ccp_alpha in ccp_alphas:
        model_temp = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        scores = cross_val_score(model_temp, dt_X_train, y_train_smote, cv=5)
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_model = model_temp

    print(f'Best cross-validated accuracy: {best_score:.2f}')

    return best_score, best_model

# ==== RANDOM FORESTS =====
# Ottimizzazione degli iperparametri di RF con GridSearchCV
def tune_rf_hyperparameters(model, model_X_train, y_train_smote):
    # Configuro la griglia di ricerca con Cross-validation
    params_grid = {
            'n_estimators' : [100,350,500],
            'max_features' : ['log2','sqrt'],
            'min_samples_leaf' : [2,10,30]         
    }

    model_cv = GridSearchCV(estimator=model,
                       param_grid=params_grid,
                       scoring='accuracy',
                       cv=5,
                       # verbose=1,
                       n_jobs=-1)


    # Eseguo Gird Search per trovare la configurazione ottimale del modello
    model_cv.fit(model_X_train, y_train_smote)

    # Stampo gli iperparametri e la loro accuracy
    print("Best N. Estimators:", model_cv.best_params_['n_estimators'])
    print("Best Max Features:", model_cv.best_params_['max_features'])
    print("Best Min Samples Leaf:", model_cv.best_params_['min_samples_leaf'])
    print("Best Accuracy:", model_cv.best_score_)

    return model_cv

# ==== TREE CLASSIFIER =====
# Creazione del dataframe con le Features e la relativa importanza, funzionale alla visualizzazione grafica
def feature_importance_df(model, model_X_train):

    # Salvo le importanze associate alle Features
    importances = model.feature_importances_

    # Inizializzo e riempio il dataset
    feature_importance_df = pd.DataFrame({'Feature': model_X_train.columns, 'Importance': importances})

    # Ordino il dataset in ordine decrescente in base all'Importanza delle Features
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return feature_importance_df

# ==== MATRICE DI CONFUSIONE =====
# Analisi dei coefficienti della matrice di confusione
def conf_matrix_coefficients_analysis(y_test, y_pred, conf_matrix):
    # Precisione
    precision = precision_score(y_test, y_pred)

    # Recall
    recall = recall_score(y_test, y_pred)

    # F1-score
    f1 = f1_score(y_test, y_pred)

    # Specificità
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)

    print(f'Precisione: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
    print(f'Specificità: {specificity}')

    return precision, recall, f1, tn, fp, fn, tp, specificity

# Creazione del dataframe che contiene le metriche della matrice di confusione, funzionale alla visualizzazione grafica
def conf_matrix_coefficients_to_df(precision, recall, f1, specificity):
    # Inizializzo un dizionario che contiene i valori del dataframe
    conf_matrix_coefficients_data = {
        'Metriche': ['Precisione', 'Recall', 'F1-score', 'Specificità'], 
        'Valore': [precision, recall, f1, specificity]
    }

    # Trasformo il dizionario in un dataframe
    conf_matrix_coefficients_df = pd.DataFrame(conf_matrix_coefficients_data)

    return conf_matrix_coefficients_df

# ==== TEST ACCURACY ====
# Calcolo dell'accuracy del modello sui dati di Test
def model_accuracy(model, model_X_train, X_test, y_train_smote, y_test):
    model.fit(model_X_train, y_train_smote)

    y_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)

    print("Test Accuracy:", test_accuracy)

    return y_pred, test_accuracy

# ==== AUC-ROC E INDICE DI GINI DEL MODELLO ====
# Calcolo dell'AUC-ROC e dell'Indice di Gini del modello
def model_auc_roc_gini(model_name, y_pred, y_test):

    auc = roc_auc_score(y_test, y_pred)
    gini = 2 * auc - 1 # 

    print(f"{model_name} - AUC-ROC: {auc}")
    print(f"{model_name} - Indice di Gini: {gini}") 

    return auc, gini

# ==== CONFRONTRO DEI MODELLI ====
# Creazione di un dataset che contiene i Test accuracy, Recall, F1-score, Specificità, AUC-ROC, Indice di Gini dei modeli, funzionale alla visualizzazione grafica
def model_metrics_df(model, test_accuracy, recall, f1, specificity, auc, gini):
    # Inizializzo un dataset vuoto che contiene il modello e le relative metriche
    model_comparison_df = pd.DataFrame(columns=[
            'Model', 
            'Test Accuracy',
            'Recall',
            'F1-score',
            'Specificity',
            'AUC-ROC',
            'Indice di Gini'
            ])
    
    for i in range (0,len(model)):
        model_comparison_df.loc[i] = [
            model[i],
            test_accuracy[i],
            recall[i],
            f1[i],
            specificity[i],
            auc[i],
            gini[i]
        ]

    model_comparison_df_long = pd.melt(
    model_comparison_df,
    id_vars=['Model'], 
    var_name='Metric', 
    value_name='Value' 
    )

    return model_comparison_df_long