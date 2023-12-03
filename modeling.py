# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:33:56 2023

@author: Florian Korn
"""
from sklearnex import patch_sklearn
patch_sklearn()

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

import os
path = 'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/githubrepo/'
os.chdir(path)
from evaluation import scoring_manuel
from notifier import notify_telegram_bot
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

def upsample_factor(y_train, threshold, steps = 0.001):
    '''
    This function upsamples the data in respect to use all samples of illegal addresses

    Parameters
    ----------
    y_train : pd.DataFrame.Series
        A series with the true y label.
    threshold : float
        This indicates the threshold of the classes (20% illegal for 0.2).
    steps : float, optional
        This indicates the searching steps. The default is 0.001.

    Returns
    -------
    x : TYPE
        DESCRIPTION.

    '''
    df = y_train.reset_index()
    df_illicit = df[df['illicit'] == 1]
    df_licit = df[df['illicit'] == 0]
    
    x = 1
    y = 0
    while y < threshold:
        x += steps
        
        df_1 = df_illicit
        df_illicit_temp = df_illicit.sample(frac = x,
                                       replace = True,
                                       random_state = 190)
        index_df = df_illicit_temp.index.unique()

        df_1 = df_1[~df_1.index.isin(index_df)]
        y = (len(df_illicit_temp) + len(df_1)) / (len(df_licit) + len(df_illicit_temp) + len(df_1))
    return x

def upsample_data(X_train, y_train, upsampling_factor):
    df = pd.concat([X_train, y_train], axis = 1)
    df_illicit = df[df['illicit'] == 1]
    df_licit = df[df['illicit'] == 0]
    
    df_1 = df_illicit
    df_illicit = df_illicit.sample(frac = upsampling_factor,
                                   replace = True,
                                   random_state = 190)
    
    index_df = df_illicit.index.unique()
    df_1 = df_1[~df_1.index.isin(index_df)]
    
    df_ml = pd.concat([df_licit, df_illicit, df_1], axis = 0)
    X_train = df_ml.iloc[:, :-1]
    y_train = df_ml.iloc[:, -1]
    return X_train, y_train

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
    
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'illicit'], 
                                                        df['illicit'],
                                                        train_size = 0.7, 
                                                        random_state = 190, 
                                                        stratify = df['illicit'], 
                                                        shuffle=True)
    
    df_illicit = df[df['illicit'] == 1]
    df_licit = df[df['illicit'] == 0]
    
    # Upsample
    upsample_factors = upsample_factor(y_train, 0.2)
    
    X_train, y_train = upsample_data(X_train, y_train, upsample_factors)
    
    df_illicit = df_illicit.sample(frac = upsample_factors,
                                   replace = True,
                                   random_state = 190)
    
    index_df = df_illicit.index.unique()
    df_1 = df[df['illicit'] == 1]
    df_1 = df_1[~df_1.index.isin(index_df)]
    
    df_ml = pd.concat([df_licit, df_illicit, df_1], axis = 0)
    return X_train, X_test, y_train, y_test, df_ml

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

def gridsearch_finetuning(estimator, params, kfold, pipe, preprocessing, X_train, y_train, sequentail_feature_selector = False):
    '''
    This function is able to do sequential feature selection and grid search CV to finetune models.

    Parameters
    ----------
    estimator : Function
        Is the estimator to tune - needed for sequential feature selector
        
    params : Dict
        Is a dictionary with all parameters to try.
        
    kfold : Function
        Is a iterator to do CV
        
    pipe : Pipe
        Is the piped estimator, because some need feature scaling
        
    preprocessing : Pipe
        Is the preprocessing pipe, as some need feature scaling and if we use sequential feature selector, the pipe is newly build (not all features used)
        
    X_train : Array
        Is the training data for all features
        
    y_train : Array
        Is the target variable
        
    sequentail_feature_selector : Boolean
        If True, we use sequential feature selector, else not. The default is False.

    Returns
    -------
    None.

    '''
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

def stacked_model_gridsearch(models0: list, models1: list, params_models: list, kfold):
    '''
    This function helps to generate stacked models and do hyperparametertuning with CV.

    Parameters
    ----------
    models0 : list
        A list of base models to try.
        
    models1 : list
        A list of meta learners to try.
        
    params_models : dict
        Parameters for base models (from hyperparamtertuning) and parameters for hyperparametertuning of meta learner
        
    kfold : Function
        k-Fold iterator

    Returns
    -------
    Saves a Excel File with all scores

    '''
    
    for k, j in enumerate(models1):
        for i in itertools.combinations(models0, 3):
            pipe_gridsearch = Pipeline([('stacking', StackingClassifier(estimators = list(i),
                                                                        final_estimator = j,
                                                                        cv = kfold,
                                                                        stack_method = 'predict_proba',
                                                                        n_jobs = -1,
                                                                        verbose = 2))
                                        ])
            
            dt_grid_search = GridSearchCV(estimator = pipe_gridsearch,
                                          param_grid = params_models[k],
                                          refit = 'recall',
                                          scoring = ['roc_auc', 'f1', 'precision', 'recall'],
                                          cv = kfold,
                                          verbose = 2,
                                          return_train_score = True,
                                          n_jobs = 1)
            
            dt_grid_search = dt_grid_search.fit(X_train, y_train)
            
            dt_cv_grid_search_results = pd.DataFrame(dt_grid_search.cv_results_)
            final_est = str(j).replace('()','')
            base_est = str([l[0] for l in i])
            dt_cv_grid_search_results['final_estimator'] = final_est
            dt_cv_grid_search_results['base_estimators'] = base_est

            dt_cv_grid_search_results.to_excel(f'plots/stacking/stacking_parameters_{final_est}_{base_est.replace("[","").replace("]","").replace(" ","_")}_{datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")}.xlsx')

def test_score_finetuned_models0(models0, X_train, y_train, X_test, y_test):
    for i in models0:
        temp_pipe = i[-1]
        temp_pipe.fit(X_train, y_train)
        temp_scores = scoring_manuel(temp_pipe.predict(X_test), y_test)
        temp_scores.to_excel('plots/finetuning/test_score')
        
if __name__ == '__main__':
    # Read Data
    path = 'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/'
    os.chdir(path)
    df = pd.read_parquet('final_data_set')
    
    # Split in train and test data
    X_train, X_test, y_train, y_test, df_ml = build_data(df)
    
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
    ## Hyperparameters
    ### decision tree
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
    
    ### gausian naive bayes
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
    
    ### logistic regression
    lr_params = {
        'logisticregression__penalty': ['l1', 'l2', 'elasticnet'],
        'logisticregression__C': [0,0.3,0.6,0.9,1,2,5,10],
        'logisticregression__solver': ['lbfgs', 'liblinear', 'saga'],
        'logisticregression__max_iter': [1000],
        'logisticregression__l1_ratio': [0,0.3,0.6,0.9, 1],
        'logisticregression__random_state': [190]
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
    
    ### k-Nearest-Neighbors
    knn_params = {
        'kneighborsclassifier__n_neighbors': np.arange(1,10,1),
        'kneighborsclassifier__weights': ['uniform', 'distance']
    }
    
    gridsearch_finetuning(KNeighborsClassifier(), 
                          knn_params, 
                          kfold, 
                          pipe_kNN, 
                          preprocessing, 
                          X_train, 
                          y_train, 
                          True)
    
    ### k-Means
    # km_params = {
    #     'n_clusters': np.arange(2,10,1),
    #     'max_iter': np.arange(300,900,300),
    #     'random_state': 190
    # }
    
    ### random forest
    rf_params = {
        'randomforestclassifier__n_estimators': [100, 200, 300],
        'randomforestclassifier__max_depth': np.arange(3,22,3),
        #'model2__min_samples_split': np.arange(10,50,10), # The minimum number of samples required to split an internal node
        #'model2__min_samples_leaf': np.arange(10,50,10), # The minimum number of samples required to be at a leaf node. 
        'randomforestclassifier__max_features': [10,20,30,40,50,60],
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
    
    ### ada boost
    ab_params = {
        'adaboostclassifier__n_estimators': np.arange(1,301,50),
        'adaboostclassifier__learning_rate': [0.0, 0.3, 0.6, 0.9, 1.0, 2.0, 5.0, 10.0, 50.0],
        'adaboostclassifier__random_state': [190]
    }
    
    gridsearch_finetuning(AdaBoostClassifier(), 
                          ab_params, 
                          kfold, 
                          pipe_AB, 
                          preprocessing, 
                          X_train, 
                          y_train)
    
    ### gradient boost
    gb_params = {
        'gradientboostingclassifier__loss': ['log_loss'],
        'gradientboostingclassifier__learning_rate': [0.3, 0.6, 1, 5],
        'gradientboostingclassifier__n_estimators': np.arange(100,301,100),
        #'model2__min_samples_split': np.arange(10,50,10), # The minimum number of samples required to split an internal node
        #'model2__min_samples_leaf': np.arange(10,50,10), # The minimum number of samples required to be at a leaf node. 
        'gradientboostingclassifier__max_features': [10,30,60],
        'gradientboostingclassifier__max_depth': [3,6,9],
        #'model2__max_leaf_nodes': np.arange(10,50,10),
        #'model2__min_impurity_decrease': np.arange(0,5,1),
        'gradientboostingclassifier__random_state': [190]
    }
    
    gridsearch_finetuning(GradientBoostingClassifier(), 
                          gb_params, 
                          kfold, 
                          pipe_GB, 
                          preprocessing, 
                          X_train, 
                          y_train)
    
    ### xgboost
    xgb_params = {
        'xgbclassifier__eta': np.arange(0.1,1.1,0.3),
        'xgbclassifier__gamma': [0, 0.3, 0.6, 1],
        'xgbclassifier__max_depth': [3,6,9],
        'xgbclassifier__lambda': [0.3, 0.6, 1, 5],
        'xgbclassifier__alpha': [0.3, 0.6, 1, 5],
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
    models0 = [('decisiontree', make_pipeline(DecisionTreeClassifier(max_depth = 18,
                                                                     max_features = 30,
                                                                     random_state = 190))),
               ('GaussianNB', make_pipeline(make_column_transformer((num_pipeline,
                                                                      ['darknet_markets', 
                                                                       'lifetime', 
                                                                       'min_addresses_per_transaction_sender', 
                                                                       'max_addresses_perr_transaction_receiver', 
                                                                       'std_addresses_per_transaction_receiver', 
                                                                       'mean_addresses_per_transaction', 
                                                                       'min_addresses_per_transaction', 
                                                                       'std_addresses_per_transaction', 
                                                                       'mean_time_diff_transaction', 
                                                                       'concentration_addresses']),
                                                                     remainder = 'drop',
                                                                     verbose_feature_names_out = False),
                                            GaussianNB(var_smoothing = 0.000285))),
               ('LogisticReg', make_pipeline(make_column_transformer((num_pipeline,
                                                                      ['count_transactions_receiver',
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
                                                                     verbose_feature_names_out = False),
                                           LogisticRegression(solver = 'saga',
                                                              max_iter = 1000))),
               ('knn', make_pipeline(make_column_transformer((num_pipeline,
                                                                      ['count_transactions_receiver', 
                                                                       'count_transactions_s_equal_r', 
                                                                       'max_addresses_perr_transaction_receiver', 
                                                                       'transaction_fee_sender']),
                                                                     remainder = 'drop',
                                                                     verbose_feature_names_out = False),
                                     KNeighborsClassifier(n_neighbors = 3,
                                                          weights = 'distance'))),
               ('rf', make_pipeline(RandomForestClassifier(max_depth = 18,
                                                           max_features = 10,
                                                           n_estimators = 100,
                                                           random_state = 190))),
               ('adaboost', make_pipeline(AdaBoostClassifier(learning_rate = 0.9,
                                                             n_estimators = 250,
                                                             random_state = 190))),
               ('GradientBoosting', make_pipeline(GradientBoostingClassifier(learning_rate = 0.3,
                                                                             loss = 'log_loss',
                                                                             max_depth = 9,
                                                                             max_features = 10,
                                                                             n_estimators = 200,
                                                                             random_state = 190))),
               ('XGB', make_pipeline(xgb.XGBClassifier(reg_alpha = 0.3,
                                                       eta = 0.7,
                                                       gamma = 0.3,
                                                       reg_lambda = 0.6,
                                                       max_depth = 9,
                                                       seed = 190)))]
    
    test_score_finetuned_models0(models0, X_train, y_train, X_test, y_test)
    
    model1 = [LogisticRegression(), RandomForestClassifier()]
    
    params_model = [{
        'stacking__final_estimator__penalty': ['elasticnet'],
        'stacking__final_estimator__C': [0.1,0.5,1,5],
        'stacking__final_estimator__solver': ['saga'],
        'stacking__final_estimator__max_iter': [1000],
        'stacking__final_estimator__l1_ratio': [0,0.3,0.6,1],
        'stacking__final_estimator__random_state': [190]
        },
        {
        'stacking__final_estimator__n_estimators': [100, 200, 300],
        'stacking__final_estimator__max_depth': np.arange(3,22,6),
        #'model2__min_samples_split': np.arange(10,50,10), # The minimum number of samples required to split an internal node
        #'model2__min_samples_leaf': np.arange(10,50,10), # The minimum number of samples required to be at a leaf node. 
        'stacking__final_estimator__max_features': [10,30,60],
        #'model2__max_leaf_nodes': np.arange(10,50,10),
        #'model2__min_impurity_decrease': np.arange(0,5,1),
        'stacking__final_estimator__random_state': [190]
        }]
    
    stacked_model_gridsearch(models0, model1, params_model, kfold)
