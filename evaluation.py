# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:52:55 2023

@author: Florian Korn
"""

from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier, RandomForestClassifier
import xgboost as xgb

import os
path = 'FILEPATH'
os.chdir(path)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator 
import dask.dataframe as dd
import datetime
import locale
import itertools
import pyarrow as pa

def scoring_manuel(y_pred, y_true):
    '''
    This function helps to evaluate precision, recall, f1-score and roc-auc of given input.

    Parameters
    ----------
    y_pred : pd.DataFrame.Series
        The prediction of the data
    y_true : pd.DataFrame.Series
        The true value of the data

    Returns
    -------
    pd.DataFrame
        With precisin, recall, f1-score, roc-auc and confusion matrix.

    '''
    precision = precision_score(y_true,
                                y_pred)
    recall = recall_score(y_true,
                          y_pred)
    f1 = f1_score(y_true,
                  y_pred)
    roc_auc = roc_auc_score(y_true,
                            y_pred)
    confusion = confusion_matrix(y_true,
                                 y_pred)
    return pd.concat([pd.DataFrame(data = {'precision':[precision], 'recall':[recall], 'f1':[f1], 'roc_auc':[roc_auc], 'confusion (tn, fp) (fn, tp)':[confusion]})], axis = 0)
    
def plotting_precision_recall_curve(y_train, y_prob_pred):
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_prob_pred)
    
    sns.set(font_scale=1.6)
    sns.set_style("whitegrid")
    
    plt.plot(precisions, recalls, '-o', label="logistic regression") # IMPORTANT
    positive_fraction = np.sum(y_train == 1) / len(y_train)
    plt.plot([0,1], [positive_fraction, positive_fraction], '--', lw=3, label="'no skill' baseline")
    plt.xlabel("precision")
    plt.ylabel("recall")
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.tick_params(axis='both', which='both', labelsize=12)
    plt.title("precision-recall curve")
    plt.xlim([0.5, 1.03]) # to only show the relevant part of the plot
    #plt.grid(which='both')
    ax.grid(which='major', color='gray', linestyle='--')
    plt.show()
    
    # from the documentation:
    # The last precision and recall values are 1. and 0. respectively and do not have a corresponding threshold.
    # -> we need to drop the last precision and recall values in this plot
    plt.plot(thresholds, precisions[:-1], '-o', label="precision")
    plt.plot(thresholds, recalls[:-1], '-o', label="recall")
    plt.xlabel("threshold")
    plt.legend()
    #plt.grid(which='both')
    plt.title("Plot for reading off thresholds for given operating point", size=15)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.tick_params(axis='both', which='both', labelsize=12)
    ax.grid(which='major', color='gray', linestyle='--')
    ax.set_ylim([0.5, 1])
    plt.show()
    return precisions, recalls, thresholds

def custom_threshold_logistic_regression(threshold_list, y_prob_pred, y_train, suffix):
    df = pd.DataFrame(data = {'threshold':[], 'precision':[], 'recall':[], 'f1':[], 'roc_auc':[], 'confusion (tn, fp) (fn, tp)': []})
    
    y_prob_pred = pd.DataFrame(y_prob_pred)
    for i in threshold_list:
        Y_test_pred = y_prob_pred.applymap(lambda x: 1 if x>i else 0)
        precision = precision_score(y_train,
                                    Y_test_pred[0])
        recall = recall_score(y_train,
                              Y_test_pred[0])
        f1 = f1_score(y_train,
                      Y_test_pred[0])
        roc_auc = roc_auc_score(y_train,
                                Y_test_pred[0])
        confusion = confusion_matrix(y_train,
                                     Y_test_pred[0])
        df = pd.concat([df, pd.DataFrame(data = {'threshold':[i], 'precision':[precision], 'recall':[recall], 'f1':[f1], 'roc_auc':[roc_auc], 'confusion (tn, fp) (fn, tp)':[confusion]})], axis = 0)
    df.to_excel(f'plots/finetuning/logistic_regression_custom_threshold_{suffix}_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.xlsx')

def addresses_per_address(tx_in2, tx_out2, bool_2021):
    '''
    This function generates a list of addresses per address

    Parameters
    ----------
    tx_in2 : string
        The name of the tx_in file
    tx_out2 : string
        The name of the tx_out file
    2021_bool: boolean
        True: than year 2021, otherwise 2020

    Returns
    -------
    df : pd.DataFrame
        A dataframe with all addresses and the addresses the address trades with

    '''
    schema = pa.schema([('index', pa.string()), ('count_unique_addresses', pa.list_(pa.string()))])
    
    tx_in = dd.read_parquet(tx_in2, columns = ['txid', 'address'])
    tx_out = dd.read_parquet(tx_out2, columns = ['txid', 'address'])
    df = dd.concat([tx_in, tx_out], axis = 0)
    df = df.repartition(1000)
    
    address_per_txid = addresses_per_txid(df)
    
    if bool_2021:
        addresses_used = pd.concat([pd.read_parquet('illegal_addresses_used_2021'), pd.read_parquet('sample_legal_addresses_2021_new')], axis = 0)
    else:
        addresses_used = pd.concat([pd.read_parquet('illegal_addresses_2020'), pd.read_parquet('sample_legal_addresses_2020')], axis = 0)
    
    df = df[df['address'].isin(addresses_used['address'])]
    df = df.merge(address_per_txid, left_on = 'txid', right_on = 'index', how = 'left')
    df = df.groupby('address_x')['address_y'].apply(list, meta = ('count_unique_addresses', 'object'))
    df = df.apply(lambda x: list(itertools.chain(*x)), meta=('count_unique_addresses', 'object'))
    df = df.reset_index()
    df.to_parquet('temp_addresses_per_address', engine = 'pyarrow', schema = schema)
    
    return dd.read_parquet('temp_addresses_per_address').compute()

def smart_local_moving(X_train, df_ml, tx_in, tx_out, bool_2021, schwellenwerte = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5], suffix = 'training'):
    '''
    The implementation of the smart local moving algorithm

    Parameters
    ----------
    X_train : array
        The training data (with independent variables).
    df_ml : pd.DataFrame
        complete data set with all data.
    tx_in : string
        Name of the tx_in file (after complete csv)
    tx_out : string
        Name of the tx_out file
    2021_bool: boolean
        True: than year 2021, otherwise 2020
        
    Returns
    -------
    None.

    '''
    df = df_ml.iloc[X_train.index, :]
    df = df.loc[:, ['illicit']]
    df['pred'] = df['illicit'].copy()
    
    df_addresses_per_address = addresses_per_address(tx_in, tx_out, bool_2021)
    
    df = df.reset_index()
    df = df.merge(df_addresses_per_address, left_on = 'address', right_on = 'index', how = 'left')
    df = df.drop(columns = ['index'])
    df['count_unique_addresses'] = df.apply(lambda x: [i for i in list(x['count_unique_addresses']) if i is not None and i not in x['address']], axis = 1)
    df['count_transactions'] = df['count_unique_addresses'].apply(lambda x: len(x))
    df = df.set_index('address')
    
    sum_equal = 0
    
    for schwellenwert in schwellenwerte:
        changed = True
        df2 = df.copy()
        sum_equal_count = 0
        while changed:
            changed = False
            temp_dict = df2['pred'].to_dict()
            df2['count_transaction_illegal'] = df2['count_unique_addresses'].copy()
            df2['count_transaction_illegal'] = df2['count_transaction_illegal'].apply(lambda x: [temp_dict[item] if (item != None and item in temp_dict.keys()) else None for item in x].count(1) if type(x) == list else None)
            df2['pred2'] = df2['count_transaction_illegal'] / df2['count_transactions']
            df2 = df2.drop(columns = 'count_transaction_illegal')
            df2['pred2'] = df2['pred2'].apply(lambda x: 0 if x < schwellenwert else 1)
            if (((df2['pred'].equals(df2['pred2'])) == False) and (sum_equal_count < 4)):
                changed = True
                sum_equal_prev = sum_equal
                sum_equal = sum(df2['pred'] == df2['pred2'])
                if sum_equal_prev == sum_equal:
                    sum_equal_count += 1
                print(sum_equal)
                df2['pred'] = df2['pred2']
    
            print(changed, datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"))
                        
        df2 = scoring_manuel(df2['illicit'], df2['pred'])
        df2.to_excel(f'plots/evaluation/smart_local_moving_algorithm_{suffix}_{schwellenwert}_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.xlsx')
        print(schwellenwert)
        
def plot_barplot(feature_importance, features_names_dict, name):
    '''
    This function plots numerical and continious data.

    Parameters:
    data_encoded : this is a pandas DataFrame which has encoded (ordinal) data in it
    x : is the variable we want to plot

    Returns:
    None, but prints the plots.
    '''
    #binwidth = np.round((np.max(data_upsampled[x]) - np.min(data_upsampled[x])) * ((len(data_upsampled[x]) ** (1/3))) / (3.49 * np.std(data_upsampled[x])))
    #binwidth = int(binwidth)
    locale.setlocale(locale.LC_ALL, 'de_DE')
    cm = 1/2.54
    fig, ax = plt.subplots(1,1, figsize = (16 * cm, 8 * cm))
    dict_font_subtitles = {'fontsize': 12, 'fontweight': 'bold', 'family': 'Arial'}
    sns.set(style="whitegrid")
    ax = sns.barplot(feature_importance,
                orient = 'h',
                y = feature_importance['feature_names'].map(features_names_dict),
                x = 'feature_importance',
                palette = 'Greys',
                edgecolor='black')
    plt.title(name, fontdict = dict_font_subtitles)

    plt.xlabel('Einfluss des Features', fontdict = dict_font_subtitles)
    plt.ylabel('Name des Features', fontdict = dict_font_subtitles)
    plt.ticklabel_format(axis = 'x', style = 'plain', useLocale = True)
    plt.tick_params(axis='both', colors='black')
    plt.savefig(f'plots/feature_importance/feature_importance_{name}.pdf', format='pdf', bbox_inches = 'tight')

def plot_knn(feature_importance2, feature_names_dict):
    locale.setlocale(locale.LC_ALL, 'de_DE')
    cm = 1/2.54
    fig, ax = plt.subplots(1,1, figsize = (16 * cm, 4 * cm))
    dict_font_subtitles = {'fontsize': 12, 'fontweight': 'bold', 'family': 'Arial'}
    sns.set(style="whitegrid")
    ax = sns.barplot(data=feature_importance2[:-1].sort_values('recall', ascending=False),
                     orient='h',
                     y=feature_importance2[:-1].sort_values('recall', ascending=False)['features'].map(feature_names_dict),
                     x='recall',
                     palette='Greys',
                     edgecolor='black')

    # Annotate each bar with its value
    for p in ax.patches:
        value = p.get_width()
        formatted_value = locale.format('%1.3f', value, grouping=True)
        ax.annotate(f'{formatted_value}',  
                    (0.3 / 2.54, p.get_y() + p.get_height() / 2),
                    ha='right', va='center',  
                    color='black',
                    fontsize=11,
                    fontfamily = 'Arial', 
                    fontweight='bold')
    plt.title('Ausprägung der Feature Importance beim k-Nearest Neighbor', fontdict = dict_font_subtitles)

    plt.xlabel('Einfluss des Features', fontdict = dict_font_subtitles)
    plt.ylabel('Name des Features', fontdict = dict_font_subtitles)
    plt.ticklabel_format(axis = 'x', style = 'plain', useLocale = True)
    plt.tick_params(axis='both', colors='black')
    plt.savefig('plots/feature_importance/feature_importance_knn.pdf', format='pdf', bbox_inches = 'tight')

if __name__ == '__main__':
    from modeling import build_data
    from create_dataset import addresses_per_txid
    # Read Data
    path = 'FILEPATH'
    os.chdir(path)
    df_2020 = pd.read_parquet('final_data_set_610682-663904')
    df_2021 = pd.read_parquet('final_data_set_663891-716590')
    X_train_2020, X_test_2020, y_train_2020, y_test_2020, df_ml_2020 = build_data(df_2020)
    X_train_2021, X_test_2021, y_train_2021, y_test_2021, df_ml_2021 = build_data(df_2021)
    
    # normalisation
    num_attribs = df_ml_2020.columns
    num_attribs = num_attribs.to_list()
    num_attribs = [i for i in num_attribs if i != 'illicit' ]
    
    num_pipeline = make_pipeline(StandardScaler())
    
    preprocessing = make_column_transformer((num_pipeline, num_attribs), 
                                            remainder = 'passthrough',
                                            verbose_feature_names_out = False)
    
    # Evaluation Model
    ## Logistic Regression
    # Trainingscore
    preprocessing = make_column_transformer((num_pipeline, ['count_transactions_receiver', 
                                                            'count_transactions_s_equal_r', 
                                                            'min_transaction_value', 
                                                            'max_transaction_value', 
                                                            'std_transaction_value', 
                                                            'min_transaction_value_sender', 
                                                            'max_transaction_value_sender', 
                                                            'std_transaction_value_sender', 
                                                            'min_transaction_value_receiver', 
                                                            'max_transaction_value_receiver', 
                                                            'std_transaction_value_receiver', 
                                                            'mean_balance', 
                                                            'std_balance', 
                                                            'min_addresses_per_transaction_receiver', 
                                                            'min_addresses_per_transaction', 
                                                            'transaction_volume_btc', 
                                                            'transaction_volume_receiver_btc', 
                                                            'transaction_volume_receiver_euro', 
                                                            'transaction_fee', 
                                                            'transaction_fee_receiver', 
                                                            'mean_transactions_receiver', 
                                                            'mean_transactions_s_equal_r', 
                                                            'mean_transactions_fee', 
                                                            'mean_transactions_fee_sender', 
                                                            'mean_transactions_fee_receiver', 
                                                            'mean_transactions_volume', 
                                                            'mean_transactions_volume_sender', 
                                                            'mean_transactions_volume_receiver', 
                                                            'concentration_addresses_receiver']), 
                                            remainder = 'drop',
                                            verbose_feature_names_out = False)
    pipe_LR = Pipeline([('preprocess', preprocessing),
                        ('model2', LogisticRegression(C=2, max_iter = 1000, penalty = 'l2', random_state = 190, solver= 'lbfgs'))])
    pipe_LR = pipe_LR.fit(X_train_2020, y_train_2020)
    
    y_prob_pred = pipe_LR.predict_proba(X_train_2020)[:, 1] # IMPORTANT: note the "[:, 1]"
    
    # Testscore
    y_prob_pred_test = pipe_LR.predict_proba(df_ml_2021.loc[:, ~df_ml_2021.columns.isin(['illicit'])])[:, 1] # IMPORTANT: note the "[:, 1]"
    
    precisions, recalls, thresholds = plotting_precision_recall_curve(y_train_2020, y_prob_pred)
    good_precisions = precisions * (recalls >= 0.97) # set all precisions to zero where the recall is too low
    best_index = np.argmax(good_precisions)
    threshold = thresholds[best_index]
    
    threshold_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,.7,.75,.8,.85,.9,.95,.99]
    
    custom_threshold_logistic_regression(threshold_list, y_prob_pred, y_train_2020, 'training')
    custom_threshold_logistic_regression(threshold_list, y_prob_pred_test, df_ml_2021['illicit'], 'test')
    
    ## Smart Local Moving Algorithm
    smart_local_moving(X_train_2020, 
                       df_ml_2020, 
                       'tx_in-610682-663904', 
                       'tx_out-610682-663904', 
                       False, 
                       suffix = 'training')
    smart_local_moving(df_ml_2021.loc[:,~df_ml_2021.columns.isin(['illicit'])], 
                       df_ml_2021, 
                       'tx_in-663891-716590', 
                       'tx_out-663891-716590', 
                       True, 
                       suffix = 'test')
    
    # finales Stacking Modell  
    dt = ('decisiontree', make_pipeline(DecisionTreeClassifier(max_depth = 18,
                                                                     max_features = 30,
                                                                     random_state = 190)))
    knn = ('knn', make_pipeline(make_column_transformer((num_pipeline,
                                                           ['count_transactions_receiver', 
                                                            'count_transactions_s_equal_r', 
                                                            'max_addresses_perr_transaction_receiver', 
                                                            'transaction_fee_sender']),
                                                          remainder = 'drop',
                                                          verbose_feature_names_out = False),
                                KNeighborsClassifier(n_neighbors = 3,
                                                     weights = 'distance')))
    xgb = ('XGB', make_pipeline(xgb.XGBClassifier(reg_alpha = 0.3,
                                                  eta = 0.7,
                                                  gamma = 0.3,
                                                  reg_lambda = 0.6,
                                                  max_depth = 9,
                                                  seed = 190)))
    ab = ('adaboost', make_pipeline(AdaBoostClassifier(learning_rate = 0.9,
                                                       n_estimators = 250,
                                                       random_state = 190)))
    rf = ('rf', make_pipeline(RandomForestClassifier(max_depth = 18,
                                                max_features = 10,
                                                n_estimators = 100,
                                                random_state = 190)))
    
    models00 = [dt, knn, xgb]
    models01 = [knn, ab, xgb]
    models02 = [dt, ab, xgb]
    models03 = [knn, rf, xgb]
    models04 = [dt, rf, xgb]
    
    model1 = LogisticRegression(C = 0.1,
                                l1_ratio = 0.6,
                                max_iter = 1000,
                                penalty = 'elasticnet',
                                random_state = 190,
                                solver = 'saga')
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=190)
    
    stacked_model = StackingClassifier(estimators = models00,
                       final_estimator = model1,
                       cv = kfold,
                       stack_method = 'predict_proba',
                       n_jobs = -1,
                       verbose = 2)
    stacked_model1 = StackingClassifier(estimators = models01,
                       final_estimator = model1,
                       cv = kfold,
                       stack_method = 'predict_proba',
                       n_jobs = -1,
                       verbose = 2)
    stacked_model2 = StackingClassifier(estimators = models02,
                       final_estimator = model1,
                       cv = kfold,
                       stack_method = 'predict_proba',
                       n_jobs = -1,
                       verbose = 2)
    stacked_model3 = StackingClassifier(estimators = models03,
                       final_estimator = model1,
                       cv = kfold,
                       stack_method = 'predict_proba',
                       n_jobs = -1,
                       verbose = 2)
    stacked_model4 = StackingClassifier(estimators = models04,
                       final_estimator = model1,
                       cv = kfold,
                       stack_method = 'predict_proba',
                       n_jobs = -1,
                       verbose = 2)
    
    stacked_model.fit(X_train_2020, y_train_2020)
    stacked_model1.fit(X_train_2020, y_train_2020)
    stacked_model2.fit(X_train_2020, y_train_2020)
    stacked_model3.fit(X_train_2020, y_train_2020)
    stacked_model4.fit(X_train_2020, y_train_2020)
    
    y_pred_stacked = stacked_model.predict(df_2021.loc[:, ~df_2021.columns.isin(['illicit'])])
    y_pred_stacked1 = stacked_model1.predict(df_2021.loc[:, ~df_2021.columns.isin(['illicit'])])
    y_pred_stacked2 = stacked_model2.predict(df_2021.loc[:, ~df_2021.columns.isin(['illicit'])])
    y_pred_stacked3 = stacked_model3.predict(df_2021.loc[:, ~df_2021.columns.isin(['illicit'])])
    y_pred_stacked4 = stacked_model4.predict(df_2021.loc[:, ~df_2021.columns.isin(['illicit'])])
    scored_stacked_model = scoring_manuel(y_pred_stacked, df_2021['illicit'])
    scored_stacked_model.to_excel(f'plots/stacking/final_stacking_model_performance_test_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}_DT_KNN_XGB.xlsx')
    scored_stacked_model = scoring_manuel(y_pred_stacked1, df_2021['illicit'])
    scored_stacked_model.to_excel(f'plots/stacking/final_stacking_model_performance_test_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}_KNN_AB_XGB.xlsx')
    scored_stacked_model = scoring_manuel(y_pred_stacked2, df_2021['illicit'])
    scored_stacked_model.to_excel(f'plots/stacking/final_stacking_model_performance_test_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}_DT_AB_XGB.xlsx')
    scored_stacked_model = scoring_manuel(y_pred_stacked3, df_2021['illicit'])
    scored_stacked_model.to_excel(f'plots/stacking/final_stacking_model_performance_test_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}_KNN_RF_XGB.xlsx')
    scored_stacked_model = scoring_manuel(y_pred_stacked4, df_2021['illicit'])
    scored_stacked_model.to_excel(f'plots/stacking/final_stacking_model_performance_test_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}_DT_RF_XGB.xlsx')

    # Feature Importance Stacking Model
    feature_names_dict = {'count_addresses': 'Anzahl der Adressen',
                      'count_addresses_sender': 'Anzahl der Adressen als Sender',
                      'count_addresses_receiver': 'Anzahl der Adressen als Empfänger',
                      'count_transactions': 'Anzahl der Transaktionen',
                      'count_transactions_sender': 'Anzahl der Transaktionen als Sender',
                      'count_transactions_receiver': 'Anzahl der Transaktionen als Empfänger', 
                      'count_transactions_s_equal_r': 'Anzahl der Transaktionen Sender gleich Empfänger',
                      'darknet_markets': 'Durchschnittlich aktive DarkNet-Marktplätze', 
                      'lifetime': 'Lebenszeit der Adresse', 
                      'min_transaction_value': 'Minimaler Transaktionswert',
                      'max_transaction_value': 'Maximaler Transaktionswert', 
                      'std_transaction_value': 'Standardabweichung Transaktionsvolumen',
                      'min_transaction_value_sender': 'Minimaler Transaktionswert als Sender', 
                      'max_transaction_value_sender': 'Maximaler Transaktionswert als Sender',
                      'std_transaction_value_sender': 'Standardabweichung Transaktionsvolumen als Sender', 
                      'min_transaction_value_receiver': 'Minimaler Transaktionswert als Empfänger',
                      'max_transaction_value_receiver': 'Maximaler Transaktionswert als Empfänger', 
                      'std_transaction_value_receiver': 'Standardabweichung Transaktionsvolumen als Empfänger',
                      'mean_balance': 'Durchschnittlicher Kontostand in Bitcoins', 
                      'std_balance': 'Standardabweichung des Kontostands in Bitcoins', 
                      'mean_addresses_per_transaction_sender': 'Durchschnittliche Anzahl an Adressen pro Transaktion als Sender',
                      'min_addresses_per_transaction_sender': 'Minimale Anzahl an Adressen pro Transaktion als Sender',
                      'max_addresses_per_transaction_sender': 'Maximale Anzahl an Adressen pro Transaktion als Sender',
                      'std_addresses_per_transaction_sender': 'Standardabweichung der Anzahl an Adressen pro Transaktion als Sender',
                      'mean_addresses_per_transaction_receiver': 'Durchschnittliche Anzahl an Adressen pro Transaktion als Empfänger',
                      'min_addresses_per_transaction_receiver': 'Minimale Anzahl an Adressen pro Transaktion als Empfänger',
                      'max_addresses_perr_transaction_receiver': 'Maximale Anzahl an Adressen pro Transaktion als Empfänger',
                      'std_addresses_per_transaction_receiver': 'Standardabweichung der Anzahl an Adressen pro Transaktion als Empfänger',
                      'mean_addresses_per_transaction': 'Durchschnittliche Anzahl an Adressen pro Transaktion', 
                      'min_addresses_per_transaction': 'Minimale Anzahl an Adressen pro Transaktion',
                      'max_addresses_per_transaction': 'Maximale Anzahl an Adressen pro Transaktion', 
                      'std_addresses_per_transaction': 'Standardabweichung der Anzahl an Adressen pro Transaktion',
                      'transaction_volume_btc': 'Transaktionsvolumen in Bitcoins', 
                      'transaction_volume_sender_btc': 'Transaktionsvolumen in Bitcoins als Sender',
                      'transaction_volume_receiver_btc': 'Transaktionsvolumen in Bitcoins als Empfänger', 
                      'transaction_volume_euro': 'Transaktionsvolumen in Euro',
                      'transaction_volume_sender_euro': 'Transaktionsvolumen in Euro als Sender', 
                      'transaction_volume_receiver_euro': 'Transaktionsvolumen in Euro als Empfänger',
                      'transaction_fee': 'Gesamtwert der Transaktionsgebühren', 
                      'transaction_fee_sender': 'Gesamtwert der Transaktionsgebühren als Sender', 
                      'transaction_fee_receiver': 'Gesamtwert der Transaktionsgebühren als Empfänger',
                      'mean_time_diff_transaction': 'Durchschnittliche Zeit zwischen Transaktionen', 
                      'std_time_diff_transaction': 'Standardabweichung der Zeit zwischen Transaktionen',
                      'mean_time_diff_transaction_sender': 'Durchschnittliche Zeit zwischen Transaktionen als Sender', 
                      'std_time_diff_transaction_sender': 'Standardabweichung der Zeit zwischen Transaktionen als Sender',
                      'mean_time_diff_transaction_receiver': 'Durchschnittliche Zeit zwischen Transaktionen als Empfänger',
                      'std_time_diff_transaction_receiver': 'Standardabweichung der Zeit zwischen Transaktionen als Empfänger', 
                      'mean_transactions': 'Durchschnittliche Anzahl der Transaktionen',
                      'mean_transactions_sender': 'Durchschnittliche Anzahl der Transaktionen als Sender', 
                      'mean_transactions_receiver': 'Durchschnittliche Anzahl der Transaktionen als Empfänger',
                      'mean_transactions_s_equal_r': 'Durchschnittliche Anzahl der Transaktionen Sender gleich Empfänger', 
                      'mean_transactions_fee': 'Durchschnittliche Transaktionsgebühren',
                      'mean_transactions_fee_sender': 'Durchschnittliche Transaktionsgebühren als Sender', 
                      'mean_transactions_fee_receiver': 'Durchschnittliche Transaktionsgebühren als Empfänger',
                      'mean_transactions_volume': 'Durchschnittliches Transaktionsvolumen', 
                      'mean_transactions_volume_sender': 'Durchschnittliches Transaktionsvolumen als Sender',
                      'mean_transactions_volume_receiver': 'Durchschnittliches Transaktionsvolumen als Empfänger', 
                      'concentration_addresses': 'Adresskonzentration',
                      'concentration_addresses_sender': 'Adresskonzentration als Sender', 
                      'concentration_addresses_receiver': 'Adresskonzentration als Empfänger'}
    
    ## Coefficient and Intercept of Meta Learner
    stacked_model.final_estimator_.intercept_
    names = ['decisiontree', 'knn', 'XGB']
    stacked_model.final_estimator_.coef_
    
    # Base Learner
    ## Decision Tree
    stacked_model.estimators_[0][0].feature_names_in_
    stacked_model.estimators_[0][0].feature_importances_
    decision_tree_feature_importance = pd.DataFrame()
    decision_tree_feature_importance['feature_names'] = stacked_model.estimators_[0][0].feature_names_in_
    decision_tree_feature_importance['feature_importance'] = stacked_model.estimators_[0][0].feature_importances_
    plot_barplot(decision_tree_feature_importance.sort_values('feature_importance', ascending = False).head(10), feature_names_dict, 'Die 10 einflussreichsten Features im Entscheidungsbaumverfahren')
    
    ## knn
    feature_importance = pd.DataFrame()
    for i in range(4):
        temp_attribs = ['count_transactions_receiver', 
         'count_transactions_s_equal_r', 
         'max_addresses_perr_transaction_receiver',
         'transaction_fee_sender']
        temp_attribs.pop(i)
        preprocessing = make_column_transformer((num_pipeline, temp_attribs), 
                                                remainder = 'drop',
                                                verbose_feature_names_out = False)
        temp_model = make_pipeline(preprocessing,
                      KNeighborsClassifier(n_neighbors = 3,
                      weights = 'distance'))
        temp_model.fit(X_train_2020, y_train_2020)
        feature_importance = pd.concat([feature_importance, scoring_manuel(temp_model.predict(X_train_2020), y_train_2020)], axis = 0)
    feature_importance['feature'] = num_attribs
    
    preprocessing = make_column_transformer((num_pipeline, ['count_transactions_receiver', 
     'count_transactions_s_equal_r', 
     'max_addresses_perr_transaction_receiver',
     'transaction_fee_sender']), 
                                            remainder = 'drop',
                                            verbose_feature_names_out = False)
    temp_model = make_pipeline(preprocessing,
                  KNeighborsClassifier(n_neighbors = 3,
                  weights = 'distance'))
    temp_model.fit(X_train_2020, y_train_2020)
    feature_importance['features'] = ['count_transactions_receiver', 
     'count_transactions_s_equal_r', 
     'max_addresses_perr_transaction_receiver',
     'transaction_fee_sender']
    feature_importance = pd.concat([feature_importance, scoring_manuel(temp_model.predict(X_train_2020), y_train_2020)], axis = 0)
    feature_importance['precision'] = feature_importance.iloc[-1, 0] - feature_importance['precision']
    feature_importance['recall'] = feature_importance.iloc[-1, 1] - feature_importance['recall']
    feature_importance['f1'] = feature_importance.iloc[-1, 2] - feature_importance['f1']
    feature_importance['roc_auc'] = feature_importance.iloc[-1, 3] - feature_importance['roc_auc']
    plot_knn(feature_importance, feature_names_dict)    
    
    ## XGB
    stacked_model.estimators_[-1][0].feature_names_in_
    stacked_model.estimators_[--1][0].feature_importances_
    xgb_feature_importance = pd.DataFrame()
    xgb_feature_importance['feature_names'] = stacked_model.estimators_[-1][0].feature_names_in_
    xgb_feature_importance['feature_importance'] = stacked_model.estimators_[-1][0].feature_importances_
    plot_barplot(xgb_feature_importance.sort_values('feature_importance', ascending = False).head(10), feature_names_dict, 'Feature Importance des XGBoosts')
