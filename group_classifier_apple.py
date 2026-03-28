import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

def preprocess_data(filepath='.\\Data\\train.csv'):

    data = pd.read_csv(filepath)

    X = data.drop('Quality', axis=1)
    y = data['Quality']

    scaler = StandardScaler()

    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)

    return X, y

def kfold(model, X, y):
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    return scores


def main():
    X, y = preprocess_data()    

if __name__ == "__main__": main()
