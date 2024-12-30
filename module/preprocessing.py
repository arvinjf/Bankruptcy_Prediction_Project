# ==== IMPORTS ====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor

from imblearn.over_sampling import SMOTE

# ==== SCREMATURA DELLE COLONNE =====
def rename_columns(df):
    # Rinomino le colonne in Italiano
    df.rename(columns={"Bankrupt?" : "Bancarotta",
                     " ROA(C) before interest and depreciation before interest" : "ROA(C) prima degli interessi e della svalutazione",
                     " ROA(A) before interest and % after tax" :"ROA(A) prima degli interessi e dopo le tasse",
                     " ROA(B) before interest and depreciation after tax" : "ROA(B) al lordo di interessi e ammortamenti dopo le imposte",
                     " Operating Gross Margin" : "Margine Lordo Operativo",
                     " Realized Sales Gross Margin" : "Margine Lordo sulle Vendite Effettive",
                     " Operating Profit Rate" : "Margine di Profitto Operativo",
                     " Pre-tax net Interest Rate" : "Aliquota di interesse netta ante imposte",
                     " After-tax net Interest Rate" : "Tasso di interesse netto dopo le tasse",
                     " Non-industry income and expenditure/revenue" : "Entrate e spese/redditività non legate all'industria",
                     " Continuous interest rate (after tax)" : "Tasso di interesse continuo (al netto delle imposte)",
                     " Operating Expense Rate" : "Tasso di Spesa Operativa",
                     " Research and development expense rate" : "Aliquota delle spese di ricerca e sviluppo",
                     " Cash flow rate" : "Tasso di flusso di cassa", 
                     " Interest-bearing debt interest rate" : "Aliquota degli interessi sul debito", 
                     " Tax rate (A)" : "Aliquota fiscale (A)", 
                     " Net Value Per Share (B)" : "Valore Netto per Azione (B)", 
                     " Net Value Per Share (A)" : "Valore Netto per Azione (A)", 
                     " Net Value Per Share (C)" : "Valore Netto per Azione (C)", 
                     " Persistent EPS in the Last Four Seasons" : "EPS persistente nelle ultime quattro stagioni", 
                     " Cash Flow Per Share" : "Flusso di cassa per azione", 
                     " Revenue Per Share (Yuan ¥)" : "Entrate per azione (Yuan ¥)", 
                     " Operating Profit Per Share (Yuan ¥)" : "Profitto Operativo per Azione (Yuan ¥)",
                     " Per Share Net profit before tax (Yuan ¥)" : "Profitto netto per azione prima delle tasse (Yuan ¥)", 
                     " Realized Sales Gross Profit Growth Rate" : "Tasso di crescita del margine lordo delle vendite", 
                     " Operating Profit Growth Rate" : "Tasso di crescita dell'utile operativo", 
                     " After-tax Net Profit Growth Rate" : "Tasso di crescita del profitto netto dopo le tasse", 
                     " Regular Net Profit Growth Rate" : "Aliquota crescita regolare del profitto netto", 
                     " Continuous Net Profit Growth Rate" : "Tasso di crescita continua del profitto netto", 
                     " Total Asset Growth Rate" : "Tasso di crescita totale degli asset", 
                     " Net Value Growth Rate" : "Tasso crescita valore netto",
                     " Total Asset Return Growth Rate Ratio" : "Tasso di crescita del rendimento totale degli asset ",
                     " Cash Reinvestment %" : "Percentuale di reinvestimento del contante", 
                     " Current Ratio" : "Rapporto Corrente", 
                     " Quick Ratio" : "Rapporto Rapido", 
                     " Interest Expense Ratio" : "Rapporto Spese Interesse", 
                     " Total debt/Total net worth" : "Debito totale/ Patrimonio netto totale", 
                     " Debt ratio %" : "Rapporto debito %", 
                     " Net worth/Assets" : "Patrimonio netto/Attività", 
                     " Long-term fund suitability ratio (A)" : "Livello di idoneità dei fondi a lungo termine (A)", 
                     " Borrowing dependency" : "Indebitamento da prestiti", 
                     " Contingent liabilities/Net worth" : "Crediti in sospeso/Patrimonio netto", 
                     " Operating profit/Paid-in capital" : "Profitto operativo su capitale versato", 
                     " Net profit before tax/Paid-in capital" : "Profitto netto prima delle imposte / Capitale sociale versato", 
                     " Inventory and accounts receivable/Net value" : "Inventario e crediti commerciali/Valore netto", 
                     " Total Asset Turnover" : "Rotazione totale degli asset", 
                     " Accounts Receivable Turnover" : "Rotazione dei crediti", 
                     " Average Collection Days" : "Giorni medi di incasso", 
                     " Inventory Turnover Rate (times)" : "Rotazione dell'inventario (volte)", 
                     " Fixed Assets Turnover Frequency" : "Frequenza del turnover degli asset fissi", 
                     " Net Worth Turnover Rate (times)" :  "Net Worth Turnover Rate (times)", 
                     " Revenue per person" : "Entrate per persona", 
                     " Operating profit per person" : "Profitto operativo per persona", 
                     " Allocation rate per person" : "Tariffa di allocazione per persona", 
                     " Working Capital to Total Assets" : "Capitale circolante su attivo totale", 
                     " Quick Assets/Total Assets" : "Attività rapide/Attività totali", 
                     " Current Assets/Total Assets" : "Attività correnti/Attività totali", 
                     " Cash/Total Assets" : "Cassa/Attività Totali", 
                     " Quick Assets/Current Liability" : "Attività rapide su passività corrente", 
                     " Cash/Current Liability" : "Cassa/Passività correnti", 
                     " Current Liability to Assets" : "Passività correnti su attività", 
                     " Operating Funds to Liability" : "Rapporto tra Fondi Operativi e Passività", 
                     " Inventory/Working Capital" : "Inventario/Capitale circolante", 
                     " Inventory/Current Liability" : "Inventario/Passività corrente", 
                     " Current Liabilities/Liability" : "Correnti Passività/Passivo", 
                     " Working Capital/Equity" : "Capitale circolante/patrimonio netto", 
                     " Current Liabilities/Equity" : "Passività correnti su patrimonio netto", 
                     " Long-term Liability to Current Assets" : "Debito a lungo termine su attività correnti", 
                     " Retained Earnings to Total Assets" : "Riserve utili su attivo totale", 
                     " Total income/Total expense" : "Reddito totale/Spese totali", 
                     " Total expense/Assets" : "Spese totali/Attività", 
                     " Current Asset Turnover Rate" : "Indice di Rotazione dell'Attivo Corrente", 
                     " Quick Asset Turnover Rate" : "Velocità di rotazione rapida degli asset", 
                     " Working capitcal Turnover Rate" : "Rapporto di rotazione del capitale circolante", 
                     " Cash Turnover Rate" : "Velocità di Rotazione del Contante", 
                     " Cash Flow to Sales" : "Flusso di cassa su vendite", 
                     " Fixed Assets to Assets" : "Attivi fissi su attivi", 
                     " Current Liability to Liability" : "Rapporto attuale di passività", 
                     " Current Liability to Equity" : "Passività corrente su patrimonio netto", 
                     " Equity to Long-term Liability" : "Rap. Patrimonio a Passività a Lungo Term.", 
                     " Cash Flow to Total Assets" : "Cash Flow su Attività Totali",
                     " Cash Flow to Liability" : "Flusso di cassa sulle passività",
                     " CFO to Assets" : "Rapporto CFO su Attività", 
                     " Cash Flow to Equity" : "Flusso di cassa verso il patrimonio netto", 
                     " Current Liability to Current Assets" : "Rapporto Passività Correnti su Attività Correnti", 
                     " Liability-Assets Flag" : "Indicatore Passività-Attività", 
                     " Net Income to Total Assets" : "Redditività sull'Attivo Totale (NITA)", 
                     " Total assets to GNP price" : "Totale attività rispetto al prezzo PNL", 
                     " No-credit Interval" : "Intervallo senza credito", 
                     " Gross Profit to Sales" : "Utile lordo sulle vendite",
                     " Net Income to Stockholder's Equity" : "Utile netto/patrimonio netto", 
                     " Liability to Equity" : "Responsabilità verso il patrimonio netto", 
                     " Degree of Financial Leverage (DFL)" : "Grado di leva finanziaria (DFL)", 
                     " Interest Coverage Ratio (Interest expense to EBIT)" : "Rapp. di copertura degli interessi (interessi passivi/EBIT)", 
                     " Net Income Flag" : "Indicatore di Reddito Netto", 
                     " Equity to Liability" : "Capitale proprio/Passività"}, inplace = True)
    
    return df
     
def first_column_selection(df):
    # Delle 96 colonne di cui è composto il dataset originale, rimuovo manualmente quelle che descrivono la stessa misura
    manual_selected_features = [
    'ROA(B) al lordo di interessi e ammortamenti dopo le imposte',
    'Margine Lordo Operativo',
    'Tasso di interesse netto dopo le tasse',
    'Tasso di Spesa Operativa',
    'Aliquota delle spese di ricerca e sviluppo',
    'Tasso di flusso di cassa',
    'Aliquota degli interessi sul debito',
    'Aliquota fiscale (A)',
    'Valore Netto per Azione (A)',
    'EPS persistente nelle ultime quattro stagioni',
    'Flusso di cassa per azione',
    "Tasso di crescita dell'utile operativo",
    "Tasso crescita valore netto",
    "Percentuale di reinvestimento del contante",
    "Debito totale/ Patrimonio netto totale",
    "Rapporto debito %",
    "Patrimonio netto/Attività",
    "Livello di idoneità dei fondi a lungo termine (A)",
    "Indebitamento da prestiti",
    "Crediti in sospeso/Patrimonio netto",
    "Profitto operativo su capitale versato",
    "Inventario e crediti commerciali/Valore netto",
    "Rotazione totale degli asset",
    "Profitto operativo per persona",
    "Passività correnti su attività",
    "Correnti Passività/Passivo",
    "Capitale circolante/patrimonio netto",
    "Debito a lungo termine su attività correnti",
    "Riserve utili su attivo totale",
    "Indice di Rotazione dell'Attivo Corrente",
    "Passività corrente su patrimonio netto",
    "Redditività sull'Attivo Totale (NITA)",
    "Grado di leva finanziaria (DFL)",
    "Capitale proprio/Passività",
    'Bancarotta'
    ]

    df = pd.DataFrame(df[manual_selected_features])

    return df

def second_column_selection(df):
    # Seleziono le colonne che presentano minore correlazione tra loro tramite il VIF (Variance Inflation Factor)
    def calculate_vif(df):
        """
        Calculate VIF
        """
        vif = pd.DataFrame() # Inizializzo un nuovo dataframes
        vif["Variable"] = df.columns # Nella colonna Variable inserisco il nome delle colonne di df
        vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])] # Nella colonna VIF calcolo il VIF corrispondente a ogni colonna
        vif = vif.sort_values(by="VIF", ascending=False).reset_index(drop=True) # Ordino il dataframe in ordine decrescente in base a VIF
        return vif

    vif = calculate_vif(df) # Calcolo il VIF sul dataset della bancarotta

    # Conto le features con un VIF alto
    vif_threshold = 5 # Imposto un limite per il VIF
    features_to_remove = vif[vif['VIF'] > vif_threshold]['Variable'] # Seleziono le colonne che hanno un VIF maggiore del limite impostato (vif_threshold) e le salvo in una variabile

    # Rimuovo le features con VIF > 5
    df = df.drop(columns = features_to_remove)

    return df

# ==== UNDERSAMPLING ===== 
def undersampling(df_clean):
    y = df_clean["Bancarotta"] # Isoliamo la variabile Y

    # Estraiamo l'indice delle righe di ciascuna classe
    index_class_0 = y[y == 0].index
    index_class_1 = y[y == 1].index

    np.random.seed(42)  # Impostiamo il seme per la generazione casuale

    # Selezioniamo lo stesso numero di righe per entrambi le classi
    same_size_index_class_0 = np.random.choice(index_class_0, y.value_counts().min(), replace=False) # Vengono selezionati 169 valori casuali (y.value_counts().min()) tra gli indici della classe 0 (index_class_0) 

    all_index = np.concatenate([index_class_1, same_size_index_class_0]) # Unisco le due liste di indici
    df_balanced = df_clean.loc[all_index] # Procedo al bilanciamento delle classi

    print(f'Numero di osservazioni totali: {df_balanced['Bancarotta'].value_counts()}') # Ora la distribuzione delle classi è uniforme (169 per classe)

    return df_balanced, index_class_1, same_size_index_class_0, index_class_0

# ===== OVERSAMPLING =====
def oversampling(df_clean, X_train, X_test, y_train, y_test):
    # Indice delle osservazioni nel test set
    X_test_0_ind = X_test.index

    # Indice delle osservazioni per target = 0, escludendo quelle nel test set
    available_X_0_indexes = [i for i in df_clean[df_clean['Bancarotta'] == 0].index if i not in list(X_test_0_ind)]

    # Definisco il fattore di incremento per SMOTE
    increasing_factor_smote = 1.8

    # Calcolo il numero di campioni necessari per la classe 1 (oversampling)
    dim_class_feature = int(X_train[y_train == 1].shape[0] * increasing_factor_smote)

    # Seleziono gli indici per la classe 1 (già presenti nel training set)
    X_train_1_ind = X_train[y_train == 1].index

    # Seleziono un campione casuale di indici per la classe 0, per ottenere il numero di campioni desiderato
    X_train_0_ind = df_clean.loc[available_X_0_indexes].sample(n=dim_class_feature, random_state=42).index

    # Combino gli indici delle due classi per il training set prima di applicare SMOTE
    pre_smote_indexes = X_train_0_ind.union(X_train_1_ind)

    # Seleziono le colonne di input (escludendo la colonna del target 'Bancarotta')
    x_cols = [i for i in df_clean if i not in y_train.name]

    # Applico SMOTE per eseguire l'oversampling
    oversample = SMOTE(random_state=42)
    X_train_smote, y_train_smote = oversample.fit_resample(df_clean.loc[pre_smote_indexes, x_cols], df_clean['Bancarotta'].loc[pre_smote_indexes])

    # Mostro i risultati (dimensioni del training set e distribuzione delle classi)
    print(X_train_smote.shape)
    print(y_train_smote.value_counts())

    return X_train_smote, y_train_smote



