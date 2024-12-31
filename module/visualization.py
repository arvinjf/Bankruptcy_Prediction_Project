# ==== IMPORTS ====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib_venn import venn2
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve
from sklearn.tree import plot_tree

# ==== DISTRIBUZIONE NUMERICA DELLE CLASSI PER FEATURES =====
def features_numerical_distribution_boxplot(df):
    for column in df.columns[:-1]:

        plt.figure(figsize=(8, 4))

        sns.boxplot(y=df[column], color='green') # Plotto il boxplot sulla colonna iterata

        plt.title(f"Boxplot - {column}")
        plt.ylabel("")
        plt.tight_layout()

        plot_title = f"{column.lower().replace('/', ' su ').replace(' ', '_')}_boxplot"  
        # plt.savefig(f"plot/boxplot/numerical_distributions/pre_norm/{plot_title}.png", bbox_inches='tight')
        
        plt.show()

# ==== MATRICE DI CORRELAZIONE =====
def correlation_matrix_heatmap(correlation_matrix):
    # Heatmap
    plt.figure(figsize=(20, 8)) 

    sns.heatmap(
        correlation_matrix, 
        vmin = -1, # vmin e vmax delimitano i valori che possono assumere le correlazioni
        vmax = 1, 
        cmap = "RdBu", # cmap indica i colori del grafico
        annot = True, # annot = True indica che su ogni cella della heatmap viene riportato il valore della correlazione
        fmt=".2f" # fmt indica la precisione (decimali) delle correlazioni
        ) 
    
    plt.tight_layout()

    # plt.savefig("plot/heatmap/correlation_matrix_heatmap.png", bbox_inches='tight')

    plt.show()

# ==== DISTRIBUZIONE DELLE CLASSI PER FEATURES DIVISA PER CLASSI DEL TARGET ====
def features_numerical_distribution_per_target_boxplot(df):
    # Boxplot
    for i in range(len(df.drop(['Bancarotta'], axis = 1).columns)): # Itero solo su tutte le colonne tranne l'ultima, che rappresenta il Target

        plt.figure(figsize=(8, 4)) 

        c_palette = {1:'skyblue', 0:'orange'} # Inizializzo la palette di colori per distinguere le classi del Target (0,1)
        
        sns.boxplot(data = df, y = df.columns[i], x = 'Bancarotta', hue = 'Bancarotta', palette = c_palette, legend = False) 

        # Stile
        plt.title(f"Boxplot - {df.columns[i]} per Target")
        plt.ylabel("")
        plt.tight_layout()

        # Salvo il grafico con un titolo personalizzato per ogni colonna iterata
        plot_title = f"{df.columns[i].lower().replace('/', ' su ').replace(' ', '_')}_per_target_boxplot"
        # plt.savefig(f"plot/boxplot/conditioned_plots/{plot_title}.png", bbox_inches='tight')

        # Mostro il grafico
        plt.show()

# ==== NUMERO DI OUTLIERS PER FEATURES DIVISI PER CLASSI DEL TARGET =====
def outliers_barplot(outliers_0_table, outliers_1_table):
    # Visualizzazione grafica
    fig = plt.figure(figsize=(12,8))

    gs = mpl.gridspec.GridSpec(1, 2) # Crea una griglia con 1 riga e 2 colonne per la disposizione dei subplot

    ax1 = plt.subplot(gs[0,0]) # Creo il primo subplot che occupa prima riga e prima colonna
    ax2 = plt.subplot(gs[0,1]) # Creo il secondo subplot che occupa prima riga e seconda colonna

    # Barplot
    sns.barplot(data = outliers_0_table, y = 'Features', x = 'Outliers', ax = ax1, color = 'darkred') # Barplot degli outliers per Target = 0
    sns.barplot(data = outliers_1_table, y = 'Features', x = 'Outliers', ax = ax2, color = 'darkblue') # Barplot degli outliers per Target = 1

    # Stile generale
    fig.suptitle('Number of Outliers per Feature Grouped by Target Class', fontsize = 16) # Titolo generale

    # Stile subplot di sinistra
    ax1.set_title('Target = 0')
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    # Stile subplot di destra
    ax2.set_title('Target = 1')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_yticklabels('')
    plt.tight_layout()
    
    # plt.savefig('plot/barplot/numero_di_outliers_per_features_divisi_per_classe_del_target_barplot.png', bbox_inches='tight')
    plt.show()

# ==== DISTRIBUZIONE NUMERICA DELLE CLASSI PER FEATURES SENZA OUTLIERS =====
def df_clean_boxplot(df, df_clean):
    for column in df.columns:
        fig = plt.figure(figsize=(12,5))
        
        # Creo la griglia per la visualizzazione grafica
        gs = mpl.gridspec.GridSpec(1, 2) # Griglia con 1 riga e 2 colonne
        ax1 = plt.subplot(gs[0,0]) # Il boxplot pre-normalizzazione occupa la prima riga e la prima colonna
        ax2 = plt.subplot(gs[0,1]) # Il boxplot post-normalizzazione occupa la prima riga e la seconda colonna

        # Boxplot pre-cleaning
        sns.boxplot(y=df[column], color='darkred', ax = ax1)

        # Boxplot post-cleaning
        sns.boxplot(y=df_clean[column], color='darkblue', ax = ax2)

        # Stile generale
        fig.suptitle(f'Boxplot - {column}', fontsize = 16, y = 1.025) # Titolo generale

        # Stile subplot di sinistra
        ax1.set_title('Pre-Cleaning')
        ax1.set_xlabel('')
        ax1.set_ylabel('')

        # Stile subplot di destra
        ax2.set_title('Post-Cleaning')
        ax2.set_xlabel('')
        ax2.set_ylabel('')

        plt.tight_layout()

        plot_title = f"{column.lower().replace('/', ' su ').replace(' ', '_')}_pre_vs_post_outliers_cleaning_boxplot"
        # plt.savefig(f"plot/boxplot/numerical_distributions/post_outliers/{plot_title}.png", bbox_inches='tight')

        plt.show()

# ==== IMPATTO DELL'UNDERSAMPLING ====
def undersampling_impact_pie(index_class_1, same_size_index_class_0, index_class_0):
    # Inizializzo i dati per il grafico a torta
    labels = ['Target', 'Features', 'Removed']
    sizes = [index_class_1.size, same_size_index_class_0.size, index_class_0.size - same_size_index_class_0.size]

    # Grafico a torta
    fig = plt.figure(figsize=(12,5))

    wedges, texts, autotexts = plt.pie(
        sizes, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors = ['red', 'blue', 'orange']
        )

    # Stile
    plt.title("Balancing Impact")

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    plt.legend(wedges, labels, bbox_to_anchor=(1.25, 1))
    plt.tight_layout()

    # Salvataggio del grafico
    plot_title = "balancing_dataset_pie"
    # plt.savefig(f"plot/piechart/{plot_title}.png", bbox_inches='tight')
    plt.show()

# ==== ACCURACY DI SKB VS RFE====
# Creazione del dataset che contiene le Featurs selezionate da SKB e RFE, con le relative accuracy
def accuracy_to_df(accuracies):

    # Creo un dataset con due colonne funzionale alla visualizzazione grafica
    accuracies_df = pd.DataFrame(columns=['Selected Features', 'Accuracy'])

    # Riempimento del dataset
    for i in range(len(accuracies)):
        accuracies_df.loc[i] = [i+1, round(float(accuracies[i]), 4)] # Selected Features: numero di feature considerate; Accuracy: rispettiva accuracy

    return accuracies_df

# Confronto grafico delle accuracy sui dati di Train tra SKB e RFE
def k_accuracy_lineplot(model_name, skb_accuracies, rfe_accuracies, skb_optimal_k, rfe_optimal_k):

    # Creo due dataset distinti per modello di Feature Selection
    skb_accuracies_df = accuracy_to_df(skb_accuracies)
    rfe_accuracies_df = accuracy_to_df(rfe_accuracies)

    # Visualizzazione grafica
    plt.figure(figsize = (12,5))

    # SKB
    sns.lineplot(data=skb_accuracies_df, x='Selected Features', y='Accuracy', color = 'darkred')

    # Aggiungo un punto in corrispondenza del k ottimale per SKB
    plt.scatter( 
        x = skb_optimal_k, 
        y = skb_accuracies_df[skb_accuracies_df['Selected Features'] == skb_optimal_k]['Accuracy'].values[0],
        s = 50,
        color = 'darkred'
        ) 
    
    # Aggiungo una label con informazioni sul k ottimale
    plt.text(
        skb_optimal_k,
        skb_accuracies_df[skb_accuracies_df['Selected Features'] == skb_optimal_k]['Accuracy'].values[0] + 0.005, 
        f'Optimal K: {skb_optimal_k}\nAccuracy: {skb_accuracies_df[skb_accuracies_df['Selected Features'] == skb_optimal_k]['Accuracy'].values[0]}', 
        color='white', 
        ha='center',
        fontsize = 8,
        fontweight = 'bold',
        bbox=dict(facecolor='darkred', edgecolor='black', boxstyle='round, pad=0.5'))
        
    # RFE
    sns.lineplot(data=rfe_accuracies_df, x='Selected Features', y='Accuracy', color = 'darkblue')

    # Aggiungo un punto in corrispondenza del k ottimale per RFE
    plt.scatter( 
        x = rfe_optimal_k, 
        y = rfe_accuracies_df[rfe_accuracies_df['Selected Features'] == rfe_optimal_k]['Accuracy'].values[0],
        s = 50,
        color = 'darkblue'
        ) 
    
    # Aggiungo una label con informazioni sul k ottimale
    plt.text(
        rfe_optimal_k,
        rfe_accuracies_df[rfe_accuracies_df['Selected Features'] == rfe_optimal_k]['Accuracy'].values[0] + 0.005, 
        f'Optimal K: {rfe_optimal_k}\nAccuracy: {rfe_accuracies_df[rfe_accuracies_df['Selected Features'] == rfe_optimal_k]['Accuracy'].values[0]}', 
        color='white', 
        ha='left',
        fontsize = 8,
        fontweight = 'bold',
        bbox=dict(facecolor='darkblue', edgecolor='black', boxstyle='round, pad=0.5'))

    # Stile
    plt.title(f'{model_name} - SKB vs - RFE - Accuracy per Number of Features', pad=20, fontsize = 16)
    plt.xlabel('Number of Features', labelpad=20, fontsize = 12)
    plt.ylabel('Mean of accuracies', labelpad=20, fontsize = 12)

    # Legenda
    skb_line = mlines.Line2D([], [], color="darkred", label="SKB")
    rfe_line = mlines.Line2D([], [], color="darkblue", label="RFE")

    plt.legend(handles=[skb_line, rfe_line], loc="lower right")

    plt.tight_layout()

    # plt.savefig(f"plot/lineplot/{model_name.replace(" ", "_").lower()}_skb_vs_rfe_accuracy_per_number_of_features_lineplot.png", bbox_inches='tight')
    plt.show()

# ==== ACCURACY DI UN SOLO ALGORITMO DI FEATURE SELECTION ====
def k_accuracy_lineplot2(model_name, accuracies, optimal_k):
    # Creo il dataset per il modello di Feature Selection
    accuracies_df = accuracy_to_df(accuracies)

    # Visualizzazione grafica
    plt.figure(figsize = (12,5))

    sns.lineplot(data=accuracies_df, x='Selected Features', y='Accuracy', color = 'darkred')

    # Aggiungo un punto in corrispondenza del k ottimale
    plt.scatter( 
        x = optimal_k, 
        y = accuracies_df[accuracies_df['Selected Features'] == optimal_k]['Accuracy'].values[0],
        s = 50,
        color = 'darkred'
        ) 
    
    # Aggiungo una label con informazioni sul k ottimale
    plt.text(
        optimal_k,
        accuracies_df[accuracies_df['Selected Features'] == optimal_k]['Accuracy'].values[0] + 0.005, 
        f'Optimal K: {optimal_k}\nAccuracy: {accuracies_df[accuracies_df['Selected Features'] == optimal_k]['Accuracy'].values[0]}', 
        color='white', 
        ha='center',
        fontsize = 8,
        fontweight = 'bold',
        bbox=dict(facecolor='darkred', edgecolor='black', boxstyle='round, pad=0.5'))
    
    # Stile
    plt.title(f'{model_name} - Accuracy per Number of Features', pad=20, fontsize = 16)
    plt.xlabel('Number of Features', labelpad=20, fontsize = 12)
    plt.ylabel('Mean of accuracies', labelpad=20, fontsize = 12)
    plt.tight_layout()

    # plt.savefig(f"plot/lineplot/{model_name.replace(" ", "_").lower()}_accuracy_per_number_of_features_lineplot.png", bbox_inches='tight')
    plt.show()

# ==== FEATURES COMUNI SELEZIONATE DA RFE E SKB ====
def features_selection_venn_diagram(model_name, skb_selected_features, rfe_selected_features):
    # DIAGRAMMA DI VENN
    set2 = set(skb_selected_features)
    set1 = set(rfe_selected_features)

    # Crea il diagramma di Venn
    plt.figure(figsize=(12, 12))
    venn_diagram = venn2([set1, set2], ('RFE', 'SKB'))

    # Assegno i nomi delle parole alle etichette nelle aree corrispondenti del diagramma di Venn
    venn_diagram.get_label_by_id('10').set_text('\n'.join(set1 - set2))
    venn_diagram.get_label_by_id('11').set_text('\n'.join(set1 & set2))
    venn_diagram.get_label_by_id('01').set_text('\n'.join(set2 - set1))

    # Stile
    plt.title(f"{model_name} - SKB vs RFE - Common Features")
    plt.tight_layout()

    # plt.savefig(f"plot/venn/{model_name.replace(" ", "_").lower()}_skb_and_rfe_common_features_venn_diagram.png", bbox_inches='tight')
    plt.show()

# ==== MATRICE DI CONFUSIONE ====
# Matrice di confusione per un solo modello
def confusion_matrix_heatmap(model_name, conf_matrix):
    # Visualizzo la matrice di confusione
    fig = plt.figure(figsize=(12,6))

    # Heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], square = True, cbar = False)

    # Stile
    plt.title(f'{model_name} - Confusion Matrix', fontsize = 16) # Titolo
    plt.tight_layout()
    
    # plt.savefig(f"plot/heatmap/{model_name.replace(" ", "_").lower()}_confusion_matrix_heatmap.png", bbox_inches='tight')
    plt.show()

# Confronto delle matrici di confusione tra SKB e RFE
def confusion_matrix_heatmap_comparison(model_name, skb_conf_matrix, rfe_conf_matrix):

    # Visualizzo la matrice di confusione
    fig = plt.figure(figsize=(15,8))

    gs = mpl.gridspec.GridSpec(1, 2) # Crea una griglia con 1 riga e 2 colonne per la disposizione dei subplot

    ax1 = plt.subplot(gs[0,0]) # Creo il primo subplot che occupa prima riga e prima colonna
    ax2 = plt.subplot(gs[0,1]) # Creo il secondo subplot che occupa prima riga e seconda colonna

    # Heatmap
    sns.heatmap(skb_conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], square = True, cbar = False, ax = ax1)
    sns.heatmap(rfe_conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], square = True, cbar = False, ax = ax2)

    # Stile generale
    fig.suptitle(f'{model_name} - SKB vs RFE - Confusion Matrix', fontsize = 16, y=1.02) # Titolo generale
    plt.tight_layout()

    # Stile subplot di sinistra
    ax1.set_title('SKB')

    # Stile subplot di destra
    ax2.set_title('RFE')

    # plt.savefig(f"plot/heatmap/{model_name.replace(" ", "_").lower()}_skb_vs_rfe_confusion_matrix_heatmap.png", bbox_inches='tight')
    plt.show()

# METRICHE DELLA MATRICE DI CONFUSIONE ====
# Confronto  delle metriche della matrice di confusione tra SKB e RFE
def conf_matrix_coefficients_barplot(model_name, conf_matrix_coefficients_df):
        plt.figure(figsize=(10, 4))

        # Palette per distinguere le barre
        pal = {
            'SKB': 'darkred',
            'RFE': 'darkblue'
        }

        # Barplot
        sns.barplot(
            data = conf_matrix_coefficients_df,
            x = 'Metriche',
            y = 'Valore',
            hue = 'Modello',
            palette = pal
            )

        # Stile
        plt.title(f"{model_name} - SKB vs RFE - Confusion Matrix Coefficients")
        plt.xlabel("")
        plt.ylabel("")
        plt.tight_layout()

        # plt.savefig(f"plot/barplot/{model_name.replace(" ", "_").lower()}_skb_vs_rfe_confusion_matrix_coefficients_barplot.png", bbox_inches='tight')   
        plt.show()

# Metriche della matrice di confusione di un solo modello
def conf_matrix_coefficients_barplot2(model_name, conf_matrix_coefficients_df):
        plt.figure(figsize=(6, 4))

        # Barplot
        sns.barplot(
            data = conf_matrix_coefficients_df,
            x = 'Metriche',
            y = 'Valore',
            color = 'darkred'
            )

        # Stile
        plt.title(f"{model_name} - Confusion Matrix Coefficients")
        plt.xlabel("")
        plt.ylabel("")
        plt.tight_layout()

        # plt.savefig(f"plot/barplot/{model_name.replace(" ", "_").lower()}_confusion_matrix_coefficients_barplot.png", bbox_inches='tight')   
        plt.show()


# ==== COEFFICIENTI DELlA REGRESSIONE LOGISTICA ====
def log_reg_coefficients_barplot(model_result_df):
    
    # Escludo la costante dalla visualizzazione grafica
    model_result_df = model_result_df.copy()
    model_result_df = model_result_df[model_result_df['Features']!= 'const']

    # Creo una colonna che rappresenta il T-Value (Coefficiente / Errore Standard)
    model_result_df['T-value'] = abs(model_result_df['Coefficiente'] / model_result_df['Errore Standard'])

    # Creazione legende personalizzate
    legend_coefficients = [
        mpatches.Patch(color='darkred', label='Positivo'),
        mpatches.Patch(color='darkblue', label='Negativo')
    ]

    legend_pvalue = [
        mpatches.Patch(color='darkred', label='Significativo (<= 0.5)'),
        mpatches.Patch(color='darkblue', label='Non significativo (> 0.5)')
    ]

    legend_tvalue = [
        mpatches.Patch(color='darkred', label='Significativo (> 2)'),
        mpatches.Patch(color='darkblue', label='Non significativo (<= 2)')
    ]

    legend_confidence = [
        mpatches.Patch(color='darkred', label='Include zero'),
        mpatches.Patch(color='darkblue', label='Non include zero')
    ]

    # Visualizzazioen grafica
    fig = plt.figure(figsize=(15,8))

    gs = mpl.gridspec.GridSpec(2, 2) # Crea una griglia con 1 riga e 2 colonne per la disposizione dei subplot

    ax1 = plt.subplot(gs[0,0]) # Creo il primo subplot che occupa prima riga e prima colonna
    ax2 = plt.subplot(gs[0,1]) # Creo il secondo subplot che occupa prima riga e seconda colonna
    ax3 = plt.subplot(gs[1,0]) # Creo il terzo subplot che occupa seconda riga e prima colonna
    ax4 = plt.subplot(gs[1,1]) # Creo il quarto subplot che occupa seconda riga e seconda colonna

    # Barplot Coefficiente
    pal = ['darkblue' if value < 0 else 'darkred' for value in model_result_df['Coefficiente']] # Blu per coefficienti negativi e Rosso per coefficienti positivi

    sns.barplot(
        data = model_result_df, 
        x = 'Coefficiente', 
        y = 'Features', 
        hue = 'Features', 
        palette = pal, 
        legend = False,
        ax = ax1
        )

    # Barplot P-value
    pal = ['darkred' if value <= 0.5 else 'darkblue' for value in model_result_df['P-value']] # Rosso per le Features significative e Blu per quelle non significative

    sns.barplot(
        data = model_result_df, 
        x = 'P-value', 
        y = 'Features', 
        hue = 'Features',
        palette = pal,
        legend = False,
        ax = ax2
        )

    # Barplot T-value
    pal = ['darkred' if value > 2 else 'darkblue' for value in model_result_df['T-value']] # Rosso per le Features con T-value > 2 e Blu per le altre

    sns.barplot(
        data = model_result_df, 
        x = 'T-value', 
        y = 'Features', 
        hue = 'Features',
        palette = pal,
        legend = False,
        ax = ax3
        )

    # Barplot Intervallo di Confidenza
    pal = model_result_df.apply(
        lambda row: 'darkred' if (row['Intervallo di Confidenza Inferiore'] < 0) and (row['Intervallo di Confidenza Superiore'] > 0) else 'darkblue',
        axis=1
        ).tolist()

    sns.barplot(
        data = model_result_df, 
        x = 'Intervallo di Confidenza Inferiore', 
        y = 'Features', 
        hue = 'Features',
        palette = pal,
        legend = False,
        ax = ax4)
    
    sns.barplot(
        data = model_result_df, 
        x = 'Intervallo di Confidenza Superiore', 
        y = 'Features', 
        hue = 'Features',
        palette = pal,
        legend = False,
        ax = ax4)

    # Stile generale
    fig.suptitle(f'Logistic Regression - Coefficients Results', fontsize = 16, y=1.02) # Titolo generale

    # Stile subplot Coefficiente
    ax1.set_title('Coefficiente')
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.legend(handles=legend_coefficients, loc='upper right')

    # Stile subplot P-value
    ax2.set_title('P-value')
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_yticklabels("")
    ax2.legend(handles=legend_pvalue, loc='upper right')

    # Stile subplot T-value
    ax3.set_title('T-value')
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.legend(handles=legend_tvalue, loc='upper right')

    # Stile subplot Intervallo di Confidenza
    ax4.set_title('Intervallo di Confidenza')
    ax4.set_xlabel("")
    ax4.set_ylabel("")
    ax4.set_yticklabels("")
    ax4.legend(handles=legend_confidence, loc='upper right')

    plt.tight_layout()
    # plt.savefig(f"plot/barplot/log_reg_coefficients_results.png", bbox_inches='tight')
    plt.show()

# ==== CURVA ROC ====
# Confronta della Curva ROC tra SKB e RFE
def roc_curve_plot(model_name, y_test, skb_y_pred, rfe_y_pred, skb_auc, rfe_auc):
    # Calcoliamo la curva ROC del modello diviso per SKB ed RFE
    skb_fpr, skb_tpr, skb_thresholds = roc_curve(y_test, skb_y_pred)
    rfe_fpr, rfe_tpr, rfe_thresholds = roc_curve(y_test, rfe_y_pred)

    # Visualizzazione grafica
    fig = plt.figure(figsize=(16,6))

    gs = mpl.gridspec.GridSpec(1, 2) # Crea una griglia con 1 riga e 2 colonne per la disposizione dei subplot

    ax1 = plt.subplot(gs[0,0]) # Creo il primo subplot che occupa prima riga e prima colonna
    ax2 = plt.subplot(gs[0,1]) # Creo il secondo subplot che occupa prima riga e seconda colonna

    # Curva ROC SKB
    ax1.plot(skb_fpr, skb_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % skb_auc)
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Curva ROC RFE
    ax2.plot(rfe_fpr, rfe_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % rfe_auc)
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Stile generale
    fig.suptitle(f'{model_name} - SKB vs RFE - ROC Curve', fontsize = 16, y=1.02) # Titolo generale
    plt.tight_layout()

    # Stile subplot di sinistra
    ax1.set_title('SKB')
    ax1.legend(loc='lower right') 

    # Stile subplot di destra
    ax2.set_title('RFE')
    ax2.legend(loc='lower right') 

    # plt.savefig(f"plot/roc_curve/{model_name.replace(" ", "_").lower()}_skb_vs_rfe_roc_curve_plot.png", bbox_inches='tight')
    plt.show()

# Curva Roc di un solo modello
def roc_curve_plot2(model_name, y_test, y_pred, auc):
    # Calcoliamo la curva ROC del modello
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Visualizzazione grafica
    plt.figure(figsize=(8,4))

    # Curva ROC 
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Stile
    plt.title(f'{model_name} - ROC Curve', fontsize = 16)
    plt.legend(loc='lower right') 
    plt.tight_layout()

    # plt.savefig(f"plot/roc_curve/{model_name.replace(" ", "_").lower()}_roc_curve_plot.png", bbox_inches='tight')
    plt.show()

# ==== FEATURES IMPORTANTI =====
# Featues e relativa importanza dell'Albero Decisionale o della Random Forests
def feature_importance_barplot(model_name, feature_importance_df):

    plt.figure(figsize=(12, 8))

    # Barplot
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, hue = 'Feature', palette='coolwarm_r')

    # Stile
    plt.title(f'{model_name} - Feature Importance', fontsize=16)
    plt.xlabel("")
    plt.ylabel("")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    # plt.savefig(f"plot/barplot/tree_classifier/{model_name.replace(" ", "_").lower()}_feature_importance_barplot.png", bbox_inches='tight')
    plt.show()

# ==== PLOT TREE ====
def model_plot_tree(best_dt_model, best_dt_score, dt_X_train, y_train_smote):
    class_names = [str(class_label) for class_label in y_train_smote.unique()]

    class_names = ['Classe 0', 'Classe 1']  # Adatta i nomi al tuo dataset

    # Best model visualization
    plt.figure(figsize=(20, 10))

    plot_tree(best_dt_model, feature_names=dt_X_train.columns, class_names=class_names, filled=True)
    plt.title(f'Best Decision Tree with ccp_alpha={best_dt_model.ccp_alpha:.4f} and cross-validated accuracy={best_dt_score:.2f}')

    # plt.savefig("plot/plot_tree/best_decision_tree.png", bbox_inches='tight')
    plt.tight_layout()

    plt.show()

# ==== CONFRONTO DEI MODELLI ====
# Confronto dell'accuracy tra i modelli
def model_comparison_barplot(model_comparison_df_long, pca = False):

    plt.figure(figsize=(12, 8))

    # Barplot
    g = sns.catplot(
        data=model_comparison_df_long,
        x='Value',
        y='Model',
        hue='Model',
        kind='bar',
        palette='coolwarm',
        col = 'Metric',
        col_wrap = 3,
        height=4,  
        aspect=1.2 
    )

    # Modifico i subplot
    for i, ax in enumerate(g.axes.flatten()):  # Itero su ogni subplot
        # Rimuovo i ticks sull'asse y per tutti i subplot
        ax.tick_params(axis='y', which='both', length=0)
        
        # Mostro i ticks sull'asse x solo per gli ultimi 3 subplot
        if i >= len(g.axes.flatten()) - 3:
            ax.tick_params(axis='x', which='both', length=5)  # Mostro i ticks sull'asse x
        else:
            ax.tick_params(axis='x', which='both', length=0)  # Rimuovo i ticks sull'asse x

        # Ottengo la metrica dal titolo del subplot
        metric_name = ax.get_title().split('=')[-1].strip()  # Otteniamo solo il nome della metrica

        # Trovo il valore massimo per quella metrica
        metric_data = model_comparison_df_long[model_comparison_df_long['Metric'] == metric_name]
        max_value = metric_data['Value'].max()
            
        # Aggiungo il valore dentro le barre per ogni subplot
        for p in ax.patches:  # Itero su tutte le barre in ogni subplot
            width = p.get_width()  # Ottengo la larghezza della barra (per barplot orizzontale)
            color = 'red' if width == max_value else 'black'  # Colore rosso per il valore massimo, altrimenti nero
            # Aggiungo il valore dentro la barra (vicino alla fine della barra)
            ax.text(
                width - 0.02,  # Posizione x (vicino alla fine della barra, sottraendo un piccolo offset)
                p.get_y() + p.get_height() / 2,  # Posizione y (al centro della barra)
                f'{width:.2f}',  # Il valore che voglio visualizzare, formattato
                ha='right',  # Allineamento orizzontale (a destra, quindi vicino alla fine della barra)
                va='center',  # Allineamento verticale (centrato)
                color=color,  # Colore del testo
                fontweight = 'bold'
            )

    # Titolo del grafico
    title = 'PCA - CONFRONTO DEI MODELLI PER METRICA' if pca else 'CONFRONTO DEI MODELLI PER METRICA'

    # Stile
    plt.suptitle(title, y=1.05, fontsize=16)
    g.set_titles("{col_name}", fontweight='bold', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', linewidth=2))  
    g.set_axis_labels("", "")
    g.despine(left=True, bottom=True) 
    plt.tight_layout()

    # plt.savefig("plot/barplot/{title.replace(" ", "_").lower()}_barplot.png", bbox_inches='tight')
    plt.show()


# ==== PRINCIPAL COMPONENT ANALYSIS ==== 
def plot_pca_variance(pca):

    # Calcolo della varianza spiegata cumulativa
    cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Trovo l'indice della componente che raggiunge o supera il 75% della varianza cumulativa
    index_selected = np.argmax(cum_explained_variance >= 0.75) + 1

    # Creo il grafico
    plt.figure(figsize=(10, 6))

    # Traccio l'istogramma della varianza spiegata da ciascuna componente
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_, alpha=0.5, label='Varianza spiegata individuale')

    # Traccio la curva della varianza spiegata cumulativa
    plt.step(range(1, len(cum_explained_variance) + 1), 
             cum_explained_variance, where='mid', label='Varianza spiegata cumulativa')

    # Evidenzio il punto dove la varianza cumulativa raggiunge il 75% con un punto rosso
    plt.scatter(index_selected, cum_explained_variance[index_selected - 1], color='red', label='75% Varianza spiegata', zorder=5)

    #Stile
    plt.title('Varianza spiegata dalle componenti principali')
    plt.xlabel('Indice della componente principale', labelpad = 15)
    plt.ylabel('Quota di varianza spiegata', labelpad = 20)

    # Aggiungo la legenda
    plt.legend(loc='best')

    # Ottimizzo il layout per evitare sovrapposizioni
    plt.tight_layout()

    # plt.savefig("plot/barplot/varianza_spiegata_dalle_componenti_principali.png", bbox_inches='tight')
    plt.show()