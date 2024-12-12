# Import lib
print("Import des librairies")
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import root_mean_squared_error

# Load data
print("Entrainement du modèle PoissonRegressor ...")
X_train_scaled = pd.read_csv("data/processed/X_train_scaled.csv", index_col = "date")
y_train = pd.read_csv("data/processed/y_train.csv", index_col = "date")
y_train = y_train.values.ravel()

# Load best parameters
print("Chargement des meilleurs paramètres trouvés ...")

with open("models/best_params.pkl", "rb") as pkl:
    best_params = pickle.load(pkl)

# Trainning
print("Entrainement du modèle PoissonRegressor ...")
model = PoissonRegressor(**best_params)
model.fit(X_train_scaled, y_train)

# Export trained model
print("Enregistrement du modèle entrainé au format PKL")
with open("models/trained_pr_model.pkl", "wb") as f:
    pickle.dump(model, f)