import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, GridSearchCV, train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

def preprocess_data(filepath='.\\Data\\train.csv'):

    data = pd.read_csv(filepath)

    X = data.drop('Quality', axis=1)
    y = data['Quality']

    scaler = StandardScaler()

    X = scaler.fit_transform(X)
    y = scaler.transform(y) # check if it should be fit or just transformed

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


def logregression_tuning(X, y):
    param_grid = {
        'C': [0.1,1,10,100],
        'l1_ratio': np.linspace(0,1, num=5, endpoint=True),
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
    }

    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=kfold, scoring='accuracy')

    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-val accuracy:", grid_search.best_score_)




def kfold(model, X, y):
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    return scores


def main():
    X, y = preprocess_data()    

if __name__ == "__main__": main()
