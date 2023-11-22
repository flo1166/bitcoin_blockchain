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
from sklearn.ensemble import StackingClassifier
import xgboost as xgb

import os
path = 'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/githubrepo/'
os.chdir(path)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator 
import dask.dataframe as dd
import datetime

def build_data(df):
    '''
    This function builds the training and test set with up and downsampling

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process.

    Returns
    -------
    X_train : array
        This are the independent variables of the training data.
    X_test : array
        This are the independent variables of the test data.
    y_train : array
        This is the dependent variables of the training data.
    y_test : array
        This is the dependent variables of the test data.
    df_ml : pd.DataFrame
        This is the complete machine learning data frame.

    '''
    # Preprocessing
    df = df.set_index('address')
    df = df.fillna(0)
    df = df.replace(float('inf'), 0)
    df['lifetime'] = df['lifetime'].replace(0, 1)
    df['mean_transactions'] = df['count_transactions'] / df['lifetime']
    df['mean_transactions_sender'] = df['count_transactions_sender'] / df['lifetime']
    df['mean_transactions_receiver'] = df['count_transactions_receiver'] / df['lifetime']
    df_illicit = df[df['illicit'] == 1]
    df_licit = df[df['illicit'] == 0]
    
    # Upsample
    df_illicit = df_illicit.sample(frac = 4.6108,
                                   replace = True,
                                   random_state = 190)
    
    index_df = df_illicit.index.unique()
    df_1 = df[df['illicit'] == 1]
    df_1 = df_1[~df_1.index.isin(index_df)]
    
    df_ml = pd.concat([df_licit, df_illicit, df_1], axis = 0)
    
    # Split in train and test data
    X_train, X_test, y_train, y_test = train_test_split(df_ml.loc[:, df_ml.columns != 'illicit'], 
                                                        df_ml['illicit'],
                                                        train_size = 0.7, 
                                                        random_state = 190, 
                                                        stratify = df_ml['illicit'], 
                                                        shuffle=True)
    return X_train, X_test, y_train, y_test, df_ml

# Read Data
path = 'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/'
os.chdir(path)
df = pd.read_parquet('final_data_set')
X_train, X_test, y_train, y_test, df_ml = build_data(df)

# normalisation
num_attribs = df_ml.columns
num_attribs = num_attribs.to_list()
num_attribs = [i for i in num_attribs if i != 'illicit' ]

num_pipeline = make_pipeline(StandardScaler())

preprocessing = make_column_transformer((num_pipeline, num_attribs), 
                                        remainder = 'passthrough',
                                        verbose_feature_names_out = False)

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
    
# Evaluation Model
## Logistic Regression
# Trainingscore
pipe = Pipeline([('preprocess', preprocessing),
                 ('model2', LogisticRegression(C=2, max_iter = 1000, penalty = 'l2', random_state = 190, solver= 'lbfgs'))])
pipe = pipe.fit(X_train, y_train)

y_prob_pred = pipe.predict_proba(X_train)[:, 1] # IMPORTANT: note the "[:, 1]"

# Testscore
y_prob_pred_test = pipe.predict_proba(X_test)[:, 1] # IMPORTANT: note the "[:, 1]"

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

precisions, recalls, thresholds = plotting_precision_recall_curve(y_train, y_prob_pred)
good_precisions = precisions * (recalls >= 0.97) # set all precisions to zero where the recall is too low
best_index = np.argmax(good_precisions)
threshold = thresholds[best_index]

threshold_list = [0.05,0.1,0.13026619778545878, 0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,.7,.75,.8,.85,.9,.95,.99]

def custom_threshold_logistic_regression(threshold_list, y_prob_pred, y_train):
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
    df.to_excel(f'plots/finetuning/logistic_regression_custom_threshold_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.xlsx')

custom_threshold_logistic_regression(threshold_list, y_prob_pred, y_train)

## Smart Local Moving Algorithm

def count_transaction_illegal(df, df_illegal):
    '''
    This function helps to count the transactions per address, which trade with illicit thoughts

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with all data of X_train and y_train.
    df_illegal : List
        With all addresses which have traded with illicit thoughts.

    Returns
    -------
    df : pd.DataFrame
        With the count of illegal transactions per address.

    '''
    df_all = dd.concat([dd.read_parquet('tx_in-610682-663904'), 
                        dd.read_parquet('tx_out-610682-663904')], axis = 0)
    df_all = df_all[['address', 'txid']]
    txid_illegal = df_all[df_all['address'].isin(df_illegal)]['txid']
    df_all = df_all.compute()
    df_all = df_all[df_all['txid'].isin(txid_illegal)]
    df_all = df_all.groupby('address')['txid'].nunique()
    df_all = df_all.reset_index()
    df_all = df_all.rename(columns = {'txid': 'count_transaction_illegal'})

    df = df.merge(df_all, on = 'address', how = 'left')
    df['count_transaction_illegal'] = df['count_transaction_illegal'].fillna(0)
 
    return df
    
def smart_local_moving(X_train, y_train, df_ml):
    '''
    The implementation of the smart local moving algorithm

    Parameters
    ----------
    X_train : array
        The training data (with independent variables).
    y_train : array
        The training data (with dependent variable)..
    df_ml : pd.DataFrame
        complete data set with all data.

    Returns
    -------
    None.

    '''
    df = pd.concat([X_train, y_train], axis = 1)
    df['address'] = df_ml.reset_index()[['address']].iloc[list(df.index),:]
    df = df.loc[:, ['address', 'count_transactions', 'illicit']]
    df = df.set_index('address')
    df['pred'] = df['illicit']
    df_illegal = list(df[df['pred'] == 1].index)
    df = df.loc[:, ['count_transactions', 'illicit' ,'pred']]
    
    changed = True
    while changed:
        changed = False
        df = count_transaction_illegal(df, df_illegal)
        df['pred2'] = df['count_transaction_illegal'] / df['count_transactions']
        df = df.drop(columns = 'count_transaction_illegal')
        df['pred2'] = np.where(df['pred2'] % 1 == 0.5, df['pred2'] + 0.1, df['pred2'])
        df['pred2'] = round(df['pred2'], 0)
        if (df['pred'].equals(df['pred2'])) == False:
            changed = True
            df['pred'] = df['pred2']
            df_illegal = set(list(df[df['pred'] == 1]['address']))

        print(changed, datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"))
                    
    df = scoring_manuel(df['illicit'], df['pred'])
    df.to_excel(f'plots/evaluation/smart_local_moving_algorithm_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.xlsx')

smart_local_moving(X_train, y_train, df_ml)
smart_local_moving(X_test, y_test, df_ml)

# finales Stacking Modell
models0 = [('decisiontree', make_pipeline(DecisionTreeClassifier(max_depth = 18,
                                                                 max_features = 30,
                                                                 random_state = 190))),
           ('knn', make_pipeline(preprocessing,
                                 KNeighborsClassifier(n_neighbors = 3,
                                                      weights = 'distance'))),
           ('XGB', make_pipeline(xgb.XGBClassifier(reg_alpha = 0.3,
                                                   eta = 0.7,
                                                   gamma = 0.3,
                                                   reg_lambda = 0.6,
                                                   max_depth = 9,
                                                   seed = 190)))]

model1 = LogisticRegression(C = 0.1,
                            l1_ratio = 0.6,
                            max_iter = 1000,
                            penalty = 'elasticnet',
                            random_state = 190,
                            solver = 'saga')

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=190)

stacked_model = StackingClassifier(estimators = models0,
                   final_estimator = model1,
                   cv = kfold,
                   stack_method = 'predict_proba',
                   n_jobs = -1,
                   verbose = 2)

stacked_model.fit(X_train, y_train)

y_pred_stacked = stacked_model.predict(X_test)
scored_stacked_model = scoring_manuel(y_pred_stacked, y_test)
scored_stacked_model.to_excel(f'plots/stacking/final_stacking_model_performance_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.xlsx')

