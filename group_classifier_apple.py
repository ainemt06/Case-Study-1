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

    print("Best parameters for SVM:", grid_search.best_params_)
    print("Cross-val accuracy for best SVM:", grid_search.best_score_)

    return grid_search.best_estimator_, grid_search.best_score_

# fix param grid - not all solvers work with l1 regularization

def logregression_tuning(X, y):
    param_grid = {
        'C': [0.1,1,10,100],
        'l1_ratio': [0,1],
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    }

    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=kfold, scoring='accuracy')

    grid_search.fit(X, y)

    print("Best parameters for Logistic Regression:", grid_search.best_params_)
    print("Cross-val accuracy for best Logistic Regression:", grid_search.best_score_)

    return grid_search.best_estimator_, grid_search.best_score_

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

    print("Best parameters for Decision Tree:", grid_search.best_params_)
    print("Cross-val accuracy for best Decision Tree:", grid_search.best_score_)

    return grid_search.best_estimator_, grid_search.best_score_


def run_test_data(model, filepath=None):
    if filepath is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, 'Data', 'test.csv') # change to test when we get it

    data = pd.read_csv(filepath)

    X = data.drop('Quality', axis=1)
    y = data['Quality']

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    print(f"Best Model: {model.__class__} with parameters  {model.get_params()}")
    print(f"Test Accuracy: {model.score(X, y)}")


def main():
    X, y = preprocess_data()
    best_svm, svm_score = svm_tuning(X,y)
    best_logregression, logregression_score =  logregression_tuning(X,y)
    best_tree, tree_score = tree_tuning(X,y)

    models = [best_svm, best_logregression, best_tree]
    scores = np.array([svm_score, logregression_score, tree_score])

    best_model = models[np.argmax(scores)]

    run_test_data(best_model)


    

if __name__ == "__main__": main()
