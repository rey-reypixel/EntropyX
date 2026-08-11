import joblib
import pandas as pd
from xgboost import plot_importance
import matplotlib.pyplot as plt

model = joblib.load("models/xgboost_model.pkl")

df = pd.read_csv("datasets/ransomware_dataset.csv")

df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

feature_names = df.drop("family", axis=1).columns

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.head(15))