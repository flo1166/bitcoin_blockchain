# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:33:56 2023

@author: Florian Korn
"""
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

import os
path = 'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/githubrepo/'
os.chdir(path)
from notifier import notify_telegram_bot
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator 
import dask.dataframe as dd

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
pipe_DT = make_pipeline(DecisionTreeClassifier()) # no standardization because tree building
pipe_gNB = make_pipeline(GaussianNB()) # no standardization because a probability is learned that when a variable is in a certain distribution, than it is assigned to a class - standardization changes the standard deviation and mean, but not the probability for the class
pipe_LR = make_pipeline(preprocessing,
                        LogisticRegression()) # standardization needed because of lasso ridge regression (big coefficients -> strong punishment)
# pipe_SVC = make_pipeline(preprocessing,
#                          sfs(estimator = SVC(), 
#                              k_features = 'best', 
#                              forward = True, 
#                              verbose = 3, 
#                              scoring = 'recall', 
#                              cv = kfold),
#                          SVC(verbose = 3))
pipe_kNN = make_pipeline(preprocessing,
                        KNeighborsClassifier()) # standardization needed because of distance metrics (need to compare features and distances)
# pipe_kM = make_pipeline(preprocessing,
#                         KMeans(verbose = 3))
pipe_RF = make_pipeline(RandomForestClassifier()) # no standardization because tree building
pipe_AB = make_pipeline(AdaBoostClassifier()) # no standardization because tree building
pipe_GB = make_pipeline(GradientBoostingClassifier())  # no standardization because tree building
pipe_XGB = make_pipeline(xgb.XGBClassifier())  # no standardization because tree building

def scores(X_train, y_train, kfold, pipeline_list):
    '''
    This function calculates the mean of specific error measures of ML models.

    Parameters:
    X_train : the training data with all features
    y_train : the training data with the target variable
    kfold : the cross validation strategy
    pipeline_list : all pipelines

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

def gridsearch_finetuning(estimator, params, kfold, pipe, preprocessing, X_train, y_train, sequentail_feature_selector = False):
    
    if sequentail_feature_selector:
        sfs_grid = sfs(
            estimator = estimator,
            k_features = 'best',
            forward = 'True',
            verbose = 1,
            scoring = 'f1',
            cv = kfold,
            n_jobs = -1
            )
        
        sfs_grid = sfs_grid.fit(X_train, y_train)
            
        preprocessing = make_column_transformer((num_pipeline, list(sfs_grid.k_feature_names_)),
                                                remainder = 'drop',
                                                verbose_feature_names_out = False)
        
        pipe = make_pipeline(preprocessing, estimator)
        
    dt_grid_search = GridSearchCV(estimator = pipe,
                                param_grid = params,
                                refit = 'recall',
                                scoring = ['roc_auc', 'f1', 'precision', 'recall'],
                                cv = kfold,
                                verbose = 3,
                                return_train_score = True,
                                n_jobs = -1)
    
    dt_grid_search = dt_grid_search.fit(X_train, y_train)
    
    dt_cv_grid_search_results = pd.DataFrame(dt_grid_search.cv_results_)

    modelname = str(dt_grid_search.estimator[-1]).replace('()','')
    if sequentail_feature_selector:
        pd.DataFrame(sfs_grid.subsets_).T.to_excel(f'plots/finetuning/{modelname}_sequential_feature_selector_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.xlsx')
    dt_cv_grid_search_results.to_excel(f'plots/finetuning/{modelname}_parameters_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.xlsx')
    
# Hyperparameters
## decision tree
dt_params = {
    'decisiontreeclassifier__max_depth': [3,5,7,9,12,15,18,21],
    #'min_samples_split': [10,30,50], # The minimum number of samples required to split an internal node
    #'min_samples_leaf': [10,30,50], # The minimum number of samples required to be at a leaf node. 
    'decisiontreeclassifier__max_features': [15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60],
    #'max_leaf_nodes': [10,30,50],
    #'min_impurity_decrease': [0,-0.1,-0.3],
    'decisiontreeclassifier__random_state': [190]
    }

gridsearch_finetuning(DecisionTreeClassifier(), 
                      dt_params, 
                      kfold, 
                      pipe_DT, 
                      preprocessing, 
                      X_train, 
                      y_train)

## gausian naive bayes
gnb_params = {
    'gaussiannb__var_smoothing': np.logspace(0,-9, num=100)
    }

gridsearch_finetuning(GaussianNB(), 
                      gnb_params, 
                      kfold, 
                      pipe_gNB, 
                      preprocessing, 
                      X_train, 
                      y_train, 
                      True)

## logistic regression
lr_params = {
    'model2__penalty': ['l1', 'l2', 'elasticnet'],
    'model2__C': [0,0.3,0.6,0.9,1,2,5,10],
    'model2__solver': ['saga'],
    'model2__max_iter': [1000],
    'model2__random_state': [190]
    }

gridsearch_finetuning(LogisticRegression(max_iter = 1000), 
                      lr_params, 
                      kfold, 
                      pipe_LR, 
                      preprocessing, 
                      X_train, 
                      y_train, 
                      True)

## Support Vector Machine
# svc_params = {
#     'C': np.arange(0,2.1,0.3),
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'random_state': 190
#     }

## k-Nearest-Neighbors
knn_params = {
    'model2__n_neighbors': np.arange(1,10,1),
    'model2__weights': ['uniform', 'distance']
}

gridsearch_finetuning(KNeighborsClassifier(), 
                      knn_params, 
                      kfold, 
                      pipe_kNN, 
                      preprocessing, 
                      X_train, 
                      y_train, 
                      True)

## k-Means
# km_params = {
#     'n_clusters': np.arange(2,10,1),
#     'max_iter': np.arange(300,900,300),
#     'random_state': 190
# }

## random forest
rf_params = {
    'randomforestclassifier__n_estimators': np.arange(100,600,200),
    'randomforestclassifier__max_depth': np.arange(3,10,3),
    #'model2__min_samples_split': np.arange(10,50,10), # The minimum number of samples required to split an internal node
    #'model2__min_samples_leaf': np.arange(10,50,10), # The minimum number of samples required to be at a leaf node. 
    'randomforestclassifier__max_features': np.arange(9,50,3),
    #'model2__max_leaf_nodes': np.arange(10,50,10),
    #'model2__min_impurity_decrease': np.arange(0,5,1),
    'randomforestclassifier__random_state': [190]
}

gridsearch_finetuning(RandomForestClassifier(), 
                      rf_params, 
                      kfold, 
                      pipe_RF, 
                      preprocessing, 
                      X_train, 
                      y_train)

## ada boost
ab_params = {
    'model2__n_estimators': np.arange(1,250,50),
    'model2__learning_rate': [0.0, 0.3, 0.6, 0.9, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
    'model2__random_state': [190]
}

gridsearch_finetuning(AdaBoostClassifier(), 
                      ab_params, 
                      kfold, 
                      pipe_AB, 
                      preprocessing, 
                      X_train, 
                      y_train)

## gradient boost
gb_params = {
    'model2__loss': ['log_loss', 'exponential'],
    'model2__learning_rate': [0, 0.3, 0.6, 1, 2, 5],
    'model2__n_estimators': np.arange(100,301,100),
    #'model2__min_samples_split': np.arange(10,50,10), # The minimum number of samples required to split an internal node
    #'model2__min_samples_leaf': np.arange(10,50,10), # The minimum number of samples required to be at a leaf node. 
    'model2__max_features': np.arange(9,60,10),
    #'model2__max_leaf_nodes': np.arange(10,50,10),
    #'model2__min_impurity_decrease': np.arange(0,5,1),
    'model2__random_state': [190]
}

gridsearch_finetuning(GradientBoostingClassifier(), 
                      gb_params, 
                      kfold, 
                      pipe_GB, 
                      preprocessing, 
                      X_train, 
                      y_train)

## xgboost
xgb_params = {
    'model2__eta': np.arange(0.1,1.1,0.3),
    'model2__gamma': [0.3, 0.6, 1],
    'model2__max_depth': np.arange(3,12,4),
    'model2__lambda': [0.3, 0.6, 1],
    'model2__alpha': [0.3, 0.6, 1],
    #'model2__max_leaves': np.arange(10,50,10)
}

gridsearch_finetuning(xgb.XGBClassifier(), 
                      xgb_params, 
                      kfold, 
                      pipe_XGB, 
                      preprocessing, 
                      X_train, 
                      y_train)

# Build the stacked model
#StackingClassifier(estimators, final_estimator)

def gridsearch_stacking_model():
    
    estimators = [[('lr'), ]]
    
    
    estimator = StackingClassifier(estimators, LogisticRegression())
    estimator2 = StackingClassifier(estimators, RandomForest())
    
    pipe_gridsearch = Pipeline([('preprocess', preprocessing),
                                ('model', estimator)])
    
    dt_grid_search = GridSearchCV(estimator = pipe_gridsearch,
                                param_grid = params,
                                refit = 'f1',
                                scoring = ['roc_auc', 'f1', 'precision', 'recall'],
                                cv = kfold,
                                verbose = 2,
                                return_train_score = True,
                                n_jobs = -1)
    
    
    dt_grid_search = dt_grid_search.fit(X_train, y_train)





















# Evaluation Model
## Logistic Regression
pipe = Pipeline([('preprocess', preprocessing),
                 ('model2', LogisticRegression(C=2, max_iter = 1000, penalty = 'l2', random_state = 190, solver= 'lbfgs'))])
pipe = pipe.fit(X_train, y_train)

y_prob_pred = pipe.predict_proba(X_train)[:, 1] # IMPORTANT: note the "[:, 1]"
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
def smart_local_moving(X_train, y_train):
    df_addresses = dd.read_parquet('addresses_per_address')
    df = pd.concat([X_train, y_train], axis = 1)
    df = df.merge(df_addresses, left_on = df.index, right_on = 'address', how = 'left').compute()
    df['pred'] = df['illicit']
    df_illegal = list(df[df['pred'] == 1].index())
    
    changed = True
    while changed:
        changed = False
        for i in df.index:
            if (sum(df.loc['i','adresses']).isin(df_illegal) / len(df.loc['i','adresses'])) > 0.5:
                if df.loc['i','pred'] == 0:
                    df.loc['i','pred'] = 1
                    changed = True
                    df_illegal = list(df[df['pred'] == 1].index())
            else:
                if df.loc['i','pred'] == 1:
                    df.loc['i','pred'] = 0
                    changed = True
                    df_illegal = list(df[df['pred'] == 1].index())
        print(changed, datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"))
                    
    precision = precision_score(df['illicit'],
                                df['pred'])
    recall = recall_score(df['illicit'],
                          df['pred'])
    f1 = f1_score(df['illicit'],
                  df['pred'])
    roc_auc = roc_auc_score(df['illicit'],
                            df['pred'])
    confusion = confusion_matrix(df['illicit'],
                                 df['pred'])
    df = pd.concat([df, pd.DataFrame(data = {'threshold':[i], 'precision':[precision], 'recall':[recall], 'f1':[f1], 'roc_auc':[roc_auc], 'confusion (tn, fp) (fn, tp)':[confusion]})], axis = 0)
    df.to_excel(f'plots/evaluation/smart_local_moving_algorithm_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.xlsx')