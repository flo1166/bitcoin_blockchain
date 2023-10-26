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
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV

import dask.dataframe as dd
import os

# Read Data
path = 'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/'
os.chdir(path)
df = dd.read_parquet('final_data_set')
df_illicit = df[df['illicit'] == 1]
df_licit = df[df['illicit'] == 0]

# Upsample
df_illicit = df_illicit.resample(n_samples = 52828, replace = True, random_state = 190)
df_ml = dd.concat([df_licit, df_illicit], axis = 0)

# Split in train and test data
X_train, X_test, y_train, y_test = train_test_split(df_ml.iloc[:, :-1], 
                                                    df_ml['illicit'],
                                                    train_size = 0.7, 
                                                    random_state = 190, 
                                                    stratify = df['illicit'], 
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