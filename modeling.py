# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:33:56 2023

@author: Florian Korn
"""
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV

import dask.dataframe as dd
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import locale
import os

# Read Data
path = 'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/'
os.chdir(path)
df = dd.read_parquet('final_data_set')

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

index_df = df_illicit.index.unique().compute()
df_1 = df[df['illicit'] == 1].compute()
df_1 = df_1[~df_1.index.isin(index_df)]

df = df.compute()
df_ml = dd.concat([df_licit, df_illicit, df_1], axis = 0).compute()

# Split in train and test data
X_train, X_test, y_train, y_test = train_test_split(df_ml.loc[:, df_ml.columns != 'illicit'], 
                                                    df_ml['illicit'],
                                                    train_size = 0.7, 
                                                    random_state = 190, 
                                                    stratify = df_ml['illicit'], 
                                                    shuffle=True)

# normalisation
num_attribs = df_ml.columns
num_attribs = num_attribs.to_list()
num_attribs = [i for i in num_attribs if i != 'illicit' ]

num_pipeline = make_pipeline(StandardScaler())

preprocessing = make_column_transformer((num_pipeline, num_attribs), 
                                        remainder = 'passthrough',
                                        verbose_feature_names_out = False).set_output(transform="pandas")

df_encoded_upsample = preprocessing.fit_transform(df_ml)

# model pipelines
pipe_DT = make_pipeline(preprocessing,
                        DecisionTreeClassifier())
pipe_gNB = make_pipeline(preprocessing,
                        GaussianNB())
pipe_LR = make_pipeline(preprocessing,
                        LogisticRegression())
pipe_SVC = make_pipeline(preprocessing,
                        SVC())
pipe_kNN = make_pipeline(preprocessing,
                        KNeighborsClassifier())
pipe_kM = make_pipeline(preprocessing,
                        KMeans())
pipe_RF = make_pipeline(preprocessing,
                        RandomForestClassifier())
pipe_AB = make_pipeline(preprocessing,
                        AdaBoostClassifier())
pipe_GB = make_pipeline(preprocessing,
                        GradientBoostingClassifier())
pipe_XGB = make_pipeline(preprocessing,
                        xgb.XGBClassifier())

def scores(X_train, y_train, kfold, pipeline_list):
    '''
    This function calculates the mean of specific error measures of ML models.

    Parameters:
    X_train : the training data with all features
    y_train : the training data with the target variable
    kfold : the cross validation strategy
    pipeline_list : all pipelines
    preprocess_pipe_target : is the preprocessing pipe for the target variable

    Returns:
    Computed mean of error measures as DataFrame
    '''
    scores_df = pd.concat([pd.DataFrame(pd.DataFrame(cross_validate(pipeline_list[i], 
                                                            X_train,
                                                            y_train,
                                                            scoring=['roc_auc', 'f1', 'precision', 'recall'],
                                                            cv=kfold,
                                                            n_jobs=-1,
                                                            return_train_score=True,
                                                            verbose = 3)).aggregate(['mean', 'std']), columns = [i]) for i in range(len(pipeline_list))], axis = 1)
    return scores_df
    
def score_quick_models(X_train: pd.DataFrame, y_train, kfold, pipeline_list):
    '''
    This function computes the error measures of quick and dirty models.

    Parameters:
    X_train : the training data with all features
    y_train : the training data with the target variable
    kfold : the cross validation strategy
    pipeline_list : all pipelines
    preprocess_pipe_target : is the preprocessing pipe for the target variable
    sub_string : string to put after model names

    Returns:
    Two heatmaps with the mean and standard deviation of the error measures
    '''
    scores_df = scores(X_train, y_train, kfold, pipeline_list)

    columns = [pipeline_list[i].steps[-1][0] for i in range(len(pipeline_list))]
    scores_df.columns = columns
    scores_stds.columns = columns

    # Visualize scores
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,4))
    sns.heatmap(scores_means, vmin = 0, vmax = 1, center = 0.5, linewidth = 1, linecolor = 'white', cmap = 'YlGn', annot = np.round(scores_means, 2), ax = ax1)
    ax1.title.set_text('Mean of Error Measures of each Model')
    sns.heatmap(scores_stds, linewidth = 1, linecolor = 'white', cmap = 'YlGn', annot = np.round(scores_stds, 2), ax = ax2)
    ax2.title.set_text('Standard Deviation of Error Measures of each Model')
    plt.savefig(f'plots/shortlisting/shortlisting_models_{datetime.now()}.pdf', format='pdf')
    scores_means.to_excel(f'plots/shortlisting/scores_means_{datetime.now()}.xlsx')
    scores_std.to_excel(f'plots/shortlisting/scores_std_{datetime.now()}.xlsx')

# K Fold iterator
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=190)

# List of pipelines
pipeline_list = [pipe_DT,
                 pipe_gNB,
                 pipe_LR,
                 pipe_SVC,
                 pipe_kNN,
                 pipe_kM,
                 pipe_RF,
                 pipe_AB,
                 pipe_GB,
                 pipe_XGB]

# Shortlisting
score_quick_models(X_train, y_train, kfold, pipeline_list)

# Finetuning
# Hyperparameters

## decision tree
dt_params = {
    'max_depth': np.arange(3,10,1),
    'min_samples_split': np.arange(10,50,10), # The minimum number of samples required to split an internal node
    'min_samples_leaf': np.arange(10,50,10), # The minimum number of samples required to be at a leaf node. 
    'max_features': np.arange(9,28,3),
    'max_leaf_nodes': np.arange(10,50,10),
    'min_impurity_decrease': np.arange(0,5,1),
    'random_state': 190
    }

## gausian naiva bayes
gnb_params = {
    'var_smoothing': np.logspace(0,-9, num=100)
    }

## logistic regression
lr_params = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': np.arange(0,2.1,0.3),
    'solver': 'saga',
    'max_iter': np.arange(100, 1000, 100),
    'random_state': 190
    }

## Support Vector Machine
svc_params = {
    'C': np.arange(0,2.1,0.3),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'random_state': 190
    }

## k-Nearest-Neighbors
knn_params = {
    'n_neighbors': np.arange(1,10,1),
    'weights': ['uniform', 'distance']
}

## k-Means
km_params = {
    'n_clusters': np.arange(2,10,1),
    'max_iter': np.arange(300,900,300),
    'random_state': 190
}

## random forest
rf_params = {
    'n_estimators': np.arange(100,600,100),
    'max_depth': np.arange(3,10,1),
    'max_features': np.arange(9,28,3),
    'bootstrap': [True, False]
}

## ada boost
ab_params = {
    'learning_rate': np.arange(0.1,1.1,0.1),
    'n_estimators': np.arange(100,600,100),
    'max_depth': np.arange(3,10,1),
    'max_features': np.arange(9,28,3)
}

## gradient boost
gb_params = {
    'learning_rate': np.arange(0.1,1.1,0.1),
    'n_estimators': np.arange(100,600,100),
    'max_depth': np.arange(3,10,1),
    'max_features': np.arange(9,28,3)
}

## xgboost
xgb_params = {
    'n_estimators': np.arange(100,600,100),
    'max_depth': np.arange(2,10,1),
    'learning_rate': np.arange(0.1,1.1,0.1)
}

#GridSearchCV()

# Build the stacked model
#StackingClassifier(estimators, final_estimator)