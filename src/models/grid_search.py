# Import lib
print("Import des librairies")
import numpy as np
import pandas as pd
import yaml
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import PoissonRegressor

from sklearn.metrics import make_scorer
from sklearn.metrics import root_mean_squared_error

# Load parameters
print("Chargement des paramètres")
params = yaml.safe_load(open("src/params.yaml"))["grid_search"]
alpha = params["alpha"]
fit_intercept = params["fit_intercept"]
solver = params["solver"]
max_iter = params["max_iter"]
print(params)

# Import datasets
print("Import des données")
X_train_scaled = pd.read_csv("data/processed/X_train_scaled.csv", sep = ',', index_col="date")
y_train = pd.read_csv("data/processed/y_train.csv", sep = ',', index_col="date")
y_train = y_train.values.ravel()
print(X_train_scaled.shape)
print(y_train.shape)

# Init parameter grid
print("Configuration de la grille de recherche")
param_grid = {
    'alpha': params["alpha"],
    'fit_intercept': params["fit_intercept"],
    'solver': params["solver"],
    'max_iter': params["max_iter"]
}
param_grid

# Fit model
print("Entrainement du modèle")
model = PoissonRegressor()
grid_search = GridSearchCV(model, param_grid=param_grid, cv=params["cv"], scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Print best parameters
print("Best parameters found :", grid_search.best_params_)
print("Meilleur score :", grid_search.best_score_)

# Export params
print("Export des meilleurs paramètres au format PKL")
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(grid_search.best_params_, f)