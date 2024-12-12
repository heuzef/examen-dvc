# Import lib
print("Import des librairies")
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Import features
print("Import des caractéristiques")
X_train = pd.read_csv("data/processed/X_train.csv", sep = ',', index_col="date")
X_test = pd.read_csv("data/processed/X_test.csv", sep = ',', index_col="date")
print(X_train.shape)
print(X_test.shape)

# Normalize features
print("Normalisation des données")
scaler = preprocessing.MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

# Export data
print("Export des données")
X_train_scaled.to_csv("data/processed/X_train_scaled.csv")
X_test_scaled.to_csv("data/processed/X_test_scaled.csv")