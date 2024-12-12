# Import lib
print("Import des librairies")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

# Load parameters
print("Chargement des paramètres")
params = yaml.safe_load(open("src/params.yaml"))["data_split"]
test_size = params["test_size"]
random_state = params["random_state"]
print(params)

# Import dataset
print("Import des données")
df = pd.read_csv("data/raw_data/raw.csv", sep = ',', index_col="date")
X, y = df.drop(['silica_concentrate'], axis = 1), df['silica_concentrate']

# Split data
print("Division des données")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state)
print("X_train : ", X_train.shape)
print("X_test : ", X_test.shape)
print("y_train : ", y_train.shape)
print("y_test : ", y_test.shape)

# Export data
print("Export des données au format CSV")
X_train.to_csv("data/processed/X_train.csv")
X_test.to_csv("data/processed/X_test.csv")
y_train.to_csv("data/processed/y_train.csv")
y_test.to_csv("data/processed/y_test.csv")