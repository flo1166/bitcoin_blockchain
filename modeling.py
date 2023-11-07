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
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV

import os
path = 'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/githubrepo/'
os.chdir(path)
from notifier import notify_telegram_bot
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read Data
path = 'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/'
os.chdir(path)
df = pd.read_parquet('final_data_set')

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

# normalisation
num_attribs = df_ml.columns
num_attribs = num_attribs.to_list()
num_attribs = [i for i in num_attribs if i != 'illicit' ]

num_pipeline = make_pipeline(StandardScaler())

preprocessing = make_column_transformer((num_pipeline, num_attribs), 
                                        remainder = 'passthrough',
                                        verbose_feature_names_out = False)

df_encoded_upsample = preprocessing.fit_transform(df_ml)

# K Fold iterator
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=190)

# model pipelines
pipe_DT = make_pipeline(preprocessing,
                        DecisionTreeClassifier())
pipe_gNB = make_pipeline(preprocessing,
                        GaussianNB())
pipe_LR = make_pipeline(preprocessing,
                        LogisticRegression(verbose = 3))
# pipe_SVC = make_pipeline(preprocessing,
#                          sfs(estimator = SVC(), 
#                              k_features = 'best', 
#                              forward = True, 
#                              verbose = 3, 
#                              scoring = 'recall', 
#                              cv = kfold),
#                          SVC(verbose = 3))
pipe_kNN = make_pipeline(preprocessing,
                        KNeighborsClassifier())
# pipe_kM = make_pipeline(preprocessing,
#                         KMeans(verbose = 3))
pipe_RF = make_pipeline(preprocessing,
                        RandomForestClassifier(verbose = 3))
pipe_AB = make_pipeline(preprocessing,
                        AdaBoostClassifier())
pipe_GB = make_pipeline(preprocessing,
                        GradientBoostingClassifier(verbose = 3))
pipe_XGB = make_pipeline(preprocessing,
                        xgb.XGBClassifier(verbose = 3))

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
                                                            scoring = ['roc_auc', 'f1', 'precision', 'recall'],
                                                            cv = kfold,
                                                            n_jobs = -1,
                                                            verbose = 1,
                                                            return_train_score = True)).aggregate(['mean', 'std'])).add_prefix(f'{pipeline_list[i].steps[-1][0]}_', axis = 0) for i in range(len(pipeline_list))], axis = 0)
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
    cm = 1/2.54
    scores_df = scores(X_train, y_train, kfold, pipeline_list)
    dict_font_titles = {'fontsize': 12, 'fontweight': 'bold', 'family': 'Arial', 'color': 'black'}
    dict_font_subtitles = {'fontsize': 11, 'fontweight': 'bold', 'family': 'Arial', 'color': 'black'}
    dict_font_annot = {'fontsize': 12, 'family': 'Arial', 'color': 'black'}
    # Visualize scores
    fig, ax1 = plt.subplots(1,1, figsize = (15 * cm, 6 * cm))
    sns.heatmap(scores_df[scores_df.index.str.contains('mean')].T.iloc[2:,:], 
                vmin = 0, 
                vmax = 1, 
                center = 0.5, 
                linewidth = 1, 
                linecolor = 'white', 
                cmap = 'YlGn', 
                annot = np.round(scores_df[scores_df.index.str.contains('mean')], 2).T.iloc[2:,:], 
                annot_kws = dict_font_annot,
                ax = ax1)
    ax1.set_title('Durchschnitt der Fehlermetriken\n(über 5-k-Fold-Cross-Validation) der Modelle', fontdict = dict_font_titles)
    ax1.set_xticklabels(scores_df[scores_df.index.str.contains('mean')].T.iloc[2:,:].columns, fontdict = dict_font_subtitles)
    ax1.set_yticklabels(scores_df[scores_df.index.str.contains('mean')].T.iloc[2:,:].index, fontdict = dict_font_subtitles)
    ax1.set_xlabel('Modelle', fontdict = dict_font_titles)
    ax1.set_ylabel('Fehlermetriken', fontdict = dict_font_titles)
    plt.savefig(f'plots/shortlisting/shortlisting_models_mean_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.pdf', format='pdf', bbox_inches = 'tight')

    fig, ax2 = plt.subplots(1,1, figsize = (15 * cm, 6 * cm))
    sns.heatmap(scores_df[scores_df.index.str.contains('std')].T.iloc[2:,:],
                linewidth = 1, 
                linecolor = 'white', 
                cmap = 'YlGn', 
                annot = np.round(scores_df[scores_df.index.str.contains('std')], 2).T.iloc[2:,:], 
                annot_kws = dict_font_annot,
                ax = ax2)
    fig.supxlabel('Modelle', 
              weight = 'bold', 
              size = 12, 
              family = 'Arial', 
              color = 'black',
              y=-0.5)
    fig.supylabel('Fehlermetriken', 
              weight = 'bold', 
              size = 12, 
              family = 'Arial', 
              color = 'black',
              x=0.02)
    ax2.set_xlabel('Modelle', fontdict = dict_font_titles)
    ax2.set_ylabel('Fehlermetriken', fontdict = dict_font_titles)
    ax2.set_title('Standardabweichung der Fehlermetriken\n(über 5-k-Fold-Cross-Validation) der Modelle', fontdict = dict_font_titles)
    ax2.set_xticklabels(scores_df[scores_df.index.str.contains('std')].T.iloc[2:,:].columns, fontdict = dict_font_subtitles)
    ax2.set_yticklabels(scores_df[scores_df.index.str.contains('std')].T.iloc[2:,:].index, fontdict = dict_font_subtitles)
    plt.savefig(f'plots/shortlisting/shortlisting_models_std_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.pdf', format='pdf', bbox_inches = 'tight')
    scores_df.to_excel(f'plots/shortlisting/scores_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.xlsx')

# List of pipelines
pipeline_list = [pipe_DT,
                 pipe_gNB,
                 pipe_LR,
                 # pipe_SVC,
                 pipe_kNN,
                 # pipe_kM,
                 pipe_RF,
                 pipe_AB,
                 pipe_GB,
                 pipe_XGB]

# Shortlisting
notify_telegram_bot(f'Starting shortlisting at {datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.')
score_quick_models(X_train, y_train, kfold, pipeline_list)
notify_telegram_bot(f'Finished shortlisting at {datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.')

# Finetuning

def show_gridsearch_finetuning(estimator, params, kfold, preprocessing, X_train, y_train, sequentail_feature_selector = False):
    estimator1 = estimator
    estimator2 = estimator
    
    if sequentail_feature_selector:
        sfs_grid = sfs(
            estimator = estimator1,
            k_features = 'best',
            forward = 'True',
            verbose = 1,
            scoring = 'f1',
            cv = kfold,
            n_jobs = -1
            )
        
        sfs_grid = sfs_grid.fit(X_train, y_train)
        
        
        preprocessing = make_column_transformer((num_pipeline, sfs_grid.k_feature_names_),
                                                remainder = 'drop',
                                                verbose_feature_names_out = False)
        
    pipe_gridsearch = make_pipeline(preprocessing,
                                    estimator2)
    
    dt_grid_search = GridSearchCV(estimator = pipe_gridsearch,
                                param_grid = params,
                                refit = False,
                                scoring = 'f1',
                                cv = kfold,
                                verbose = 2,
                                return_train_score = True,
                                n_jobs = -1)
    
    
    dt_grid_search = dt_grid_search.fit(X_train, y_train)
    
    dt_cv_grid_search_results = pd.DataFrame(dt_grid_search.cv_results_)
    dt_cv_grid_search_results = dt_cv_grid_search_results.sort_values('rank_test_f1')
    dt_cv_grid_search_results = dt_cv_grid_search_results.iloc[:, dt_cv_grid_search_results.columns.str.contains('mean_test') 
                                   + dt_cv_grid_search_results.columns.str.contains('rank_test_f1') 
                                   + dt_cv_grid_search_results.columns.str.contains('param')]
    
    modelname = str(dt_grid_search.estimator).replace('()','')
    if sequentail_feature_selector:
        pd.DataFrame(sfs_grid.subsets_).to_excel(f'plots/finetuning/finetuning_{modelname}_sequential_feature_selector_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.xlsx')
    dt_cv_grid_search_results.to_excel(f'plots/finetuning/finetuning_{modelname}_parameters_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.xlsx')
    
    # figsize_subplots = len(dt_cv_grid_search_results.iloc[:, dt_cv_grid_search_results.columns.str.contains('param')].columns) - 1
    # fig, ax = plt.subplots(int(np.round(figsize_subplots / 2, 0)), figsize_subplots, sharey = True)
    # for i in range(figsize_subplots):
    #     ax[i].plot(dt_cv_grid_search_results.iloc[:, i], dt_cv_grid_search_results.loc[:, 'mean_test_f1'])
    #     ax[i].set_xlabel(dt_cv_grid_search_results.iloc[:, 0].name)
    #     ax[i].set_ylabel('mean_test_f1')
    # fig.suptitle(modelname)
    # plt.savefig(f'plots/finetuning/finetuning_{modelname}_parameters_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.pdf', format='pdf', bbox_inches = 'tight')
    return dt_grid_search
    
# Hyperparameters
## decision tree
dt_params = {
    'max_depth': [3,5,7,9,12,15,18,21],
    #'min_samples_split': [10,30,50], # The minimum number of samples required to split an internal node
    #'min_samples_leaf': [10,30,50], # The minimum number of samples required to be at a leaf node. 
    'max_features': [15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60],
    #'max_leaf_nodes': [10,30,50],
    #'min_impurity_decrease': [0,-0.1,-0.3],
    'random_state': [190]
    }

show_gridsearch_finetuning(DecisionTreeClassifier(), dt_params, kfold, preprocessing, X_train, y_train)

## gausian naiva bayes
gnb_params = {
    'var_smoothing': np.logspace(0,-9, num=100)
    }

show_gridsearch_finetuning(GaussianNB(), gnb_params, kfold, preprocessing.fit_transform(X_train), y_train)

## logistic regression
lr_params = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': np.arange(0,2.1,0.3),
    'solver': ['saga'],
    'max_iter': np.arange(100, 1000, 200),
    'random_state': [190]
    }

gridsearch_LR = show_gridsearch_finetuning(LogisticRegression(max_iter = 500), lr_params, kfold, preprocessing ,X_train, y_train, True)

## Support Vector Machine
# svc_params = {
#     'C': np.arange(0,2.1,0.3),
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'random_state': 190
#     }

## k-Nearest-Neighbors
knn_params = {
    'n_neighbors': np.arange(1,10,1),
    'weights': ['uniform', 'distance']
}

show_gridsearch_finetuning(KNeighborsClassifier(), knn_params, kfold, preprocessing.fit_transform(X_train), y_train)

## k-Means
# km_params = {
#     'n_clusters': np.arange(2,10,1),
#     'max_iter': np.arange(300,900,300),
#     'random_state': 190
# }

## random forest
rf_params = {
    'n_estimators': np.arange(100,600,100),
    'max_depth': np.arange(3,10,1),
    'min_samples_split': np.arange(10,50,10), # The minimum number of samples required to split an internal node
    'min_samples_leaf': np.arange(10,50,10), # The minimum number of samples required to be at a leaf node. 
    'max_features': np.arange(9,28,3),
    'max_leaf_nodes': np.arange(10,50,10),
    'min_impurity_decrease': np.arange(0,5,1),
    'random_state': [190]
}

show_gridsearch_finetuning(RandomForestClassifier(), rf_params, kfold, preprocessing.fit_transform(X_train), y_train)

## ada boost
ab_params = {
    'n_estimators': np.arange(0,150,50),
    'learning_rate': np.arange(0,1,0.2),
    'random_state': [190]
}

pipe_LR = make_pipeline(preprocessing,
                        LogisticRegression(verbose = 3))

show_gridsearch_finetuning(AdaBoostClassifier(), ab_params, kfold, preprocessing.fit_transform(X_train), y_train)

## gradient boost
gb_params = {
    'loss': ['log_loss', 'exponential'],
    'learning_rate': np.arange(0,1,0.2),
    'n_estimators': np.arange(100,300,100),
    'min_samples_split': np.arange(10,50,10), # The minimum number of samples required to split an internal node
    'min_samples_leaf': np.arange(10,50,10), # The minimum number of samples required to be at a leaf node. 
    'max_features': np.arange(9,28,3),
    'max_leaf_nodes': np.arange(10,50,10),
    'min_impurity_decrease': np.arange(0,5,1),
    'random_state': [190]
}

show_gridsearch_finetuning(GradientBoostingClassifier(), gb_params, kfold, preprocessing.fit_transform(X_train), y_train)

## xgboost
xgb_params = {
    'eta': np.arange(0,1,0.2),
    'gamma': np.arange(0,10,2),
    'max_depth': np.arange(3,10,1),
    'lambda': np.arange(0,10,2),
    'alpha': np.arange(0,10,2),
    'max_leaves': np.arange(10,50,10)
}

show_gridsearch_finetuning(xgb.XGBClassifier(), xgb_params, kfold, preprocessing.fit_transform(X_train), y_train)

# Build the stacked model
#StackingClassifier(estimators, final_estimator)