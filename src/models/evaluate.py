# Import lib
print("Import des librairies")
import numpy as np
import pandas as pd
import pickle
import sklearn.metrics as metrics
import json

# Load model
print("Chargement du model entrainé ...")
with open('models/trained_pr_model.pkl', 'rb') as pkl:
    model = pickle.load(pkl)

# Load data
print("Chargement des données ...")
X_test_scaled = pd.read_csv("data/processed/X_test_scaled.csv", index_col = "date")
y_test = pd.read_csv("data/processed/y_test.csv", index_col = "date")
y_test = y_test.values.ravel()

# Predict
print("Prédictions sur le jeu de test ...")
y_pred = model.predict(X_test_scaled)

# Eval
print("Évaluation ...")
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = metrics.root_mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)

pred_df = pd.DataFrame({"targets": y_test, "preds": y_pred})
pred_df.to_csv("data/preds.csv", index=False)

# Export scores
print("Export des résultats au format JSON ...")
with open("metrics/scores.json", "w") as f:
    json.dump({"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}, f)