# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:33:56 2023

@author: Florian Korn
"""
seed()
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV

import dask.dataframe as dd

def build_ml_dataset(filename):
    '''
    Gernerates a ML dataset, depending on classes

    Parameters
    ----------
    filename : string
        the dataframe with all data entries

    Returns
    -------
    df : DataFrame
        a sampled df to learn a ML model on it

    '''
    df = dd.read_parquet(filename)
    
    df_illicit = df[df[''] == 'illicit']
    df_illicit = df_illicit.sample(frac = 0.05, 
                                   replace=False, 
                                   random_state=190)
    
    df_elicit = df[df[''] != 'illicit']
    df_elicit = df_elicit.sample(frac = 0.05, 
                                   replace=False, 
                                   random_state=190)

    df = dd.concat([df_illicit, df_elicit], axis = 0).compute()
    return df

df = build_ml_dataset(filename).compute()

X_train, X_test, y_train, y_test = train_test_split(df, 
                                                    train_size = 0.7, 
                                                    random_state = 190, 
                                                    stratify = df[''], 
                                                    shuffle=True)

num_attribs =
cat_attribs =
num_pipeline = make_pipeline('standardscaler', StandardScaler())
cat_pipeline = make_pipeline('ordinalencoder', OrdinalEncoder())

preprocessing = make_column_transformer( 
        (num_pipeline, num_attribs), 
        (cat_pipeline, cat_attribs))

GridSearchCV()

# Build the stacked model
StackingClassifier(estimators, final_estimator)