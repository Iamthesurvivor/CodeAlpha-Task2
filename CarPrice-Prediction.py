
#Imported libraries
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

#Loaded data
DATA_PATH = Path("car data.csv")
df = pd.read_csv(DATA_PATH)

#EDA
print(df.head())
print(df.info())

#Featured engineering
df["Brand"]   = df["Car_Name"].str.split().str[0]
df["Car_Age"] = 2025 - df["Year"]

#Defined X / y
X = df.drop(columns=["Selling_Price", "Car_Name", "Year"])
y = df["Selling_Price"]

#Pre-processed pipeline
categorical_cols = ["Fuel_Type", "Selling_type", "Transmission", "Brand"]
numeric_cols     = [c for c in X.columns if c not in categorical_cols]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

#Modeled + training
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

pipe.fit(X_train, y_train)

#Evaluation
y_pred = pipe.predict(X_test)
print(f"R²  : {r2_score(y_test, y_pred):.3f}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.3f} ₹ lakh")

#Cross-validation for robustness
cv_r2 = cross_val_score(pipe, X, y, cv=5, scoring="r2")
print(f"CV R² scores: {cv_r2.round(3)}  |  mean={cv_r2.mean():.3f}")

#Visualion diagnostics
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Selling Price (₹ lakh)")
plt.ylabel("Predicted Selling Price (₹ lakh)")
plt.title("Predicted vs. Actual Selling Prices")
plt.grid(True)
plt.show()
