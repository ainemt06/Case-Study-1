import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

kfold = KFold(n_splits=5, shuffle=True, random_state=1)


def preprocess_data(filepath=None):
    if filepath is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, 'Data', 'train.csv')

    data = pd.read_csv(filepath)

    X = data.drop('Quality', axis=1)
    y = data['Quality']

    scaler = StandardScaler()

    X = scaler.fit_transform(X)
    return X, y


def svm_tuning(X, y):
    param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'poly', 'sigmoid']
    }

    grid_search = GridSearchCV(SVC(), param_grid, cv=kfold,scoring='accuracy')

    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-val accuracy:", grid_search.best_score_)

# fix param grid - not all solvers work with l1 regularization

def logregression_tuning(X, y):
    param_grid = {
        'C': [0.1,1,10,100],
        'l1_ratio': np.linspace(0,1, num=5, endpoint=True),
        'solver': ['lbfgs', 'liblinear', 'saga']
    }

    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=kfold, scoring='accuracy')

    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-val accuracy:", grid_search.best_score_)

def tree_tuning(X, y):
    param_grid = {
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy'],
    'class_weight': [None, 'balanced'],
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=kfold, scoring='accuracy')

    grid_search.fit(X,y)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-val accuracy:", grid_search.best_score_)




def main():
    X, y = preprocess_data()
    svm_tuning(X,y)
    logregression_tuning(X,y)
    tree_tuning(X,y)
    

if __name__ == "__main__": main()
