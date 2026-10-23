
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. LOAD CSV DATASET
# ============================================================

df = pd.read_csv("Smart_Household_Electricity_Consumption_Dataset.csv")

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nORIGINAL DATA:")
print(df)

print("\nDATASET INFORMATION:")
df.info()

print("\nDATASET SHAPE:")
print(df.shape)


# ============================================================
# 2. RESHAPING THE DATA
# ============================================================

print("\nRESHAPED DATA:")

reshaped_data = df.melt(
    id_vars=["Customer_ID"],
    value_vars=["AC_Count", "Fan_Count", "Light_Count"],
    var_name="Appliance",
    value_name="Count"
)

print(reshaped_data.head(10))


# ============================================================
# 3. FILTERING THE DATA
# ============================================================

print("\nCUSTOMERS FROM SOLAPUR:")

solapur_customers = df[df["City"] == "Solapur"]
print(solapur_customers)


print("\nCUSTOMERS WITH HOUSE SIZE GREATER THAN 1000 SQFT:")

large_houses = df[df["House_Size_sqft"] > 1000]
print(large_houses)


# ============================================================
# 4. MERGING THE DATA
# ============================================================

billing = pd.read_csv("customer_billing.csv")

merged_data = pd.merge(
    df,
    billing,
    on="Customer_ID"
)

print("\nMERGED DATASET:")
print(merged_data.head())


# ============================================================
# 5. HANDLING MISSING VALUES
# ============================================================

print("\nMISSING VALUES BEFORE HANDLING:")
print(df.isnull().sum())

# Fill numerical missing values with column means
numerical_columns = df.select_dtypes(include=np.number).columns

for column in numerical_columns:
    df[column] = df[column].fillna(df[column].mean())

# Fill categorical missing values with column modes
categorical_columns = df.select_dtypes(
    include=["object", "str", "category"]
).columns

for column in categorical_columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(df[column].mode()[0])

print("\nMISSING VALUES AFTER HANDLING:")
print(df.isnull().sum())

print("\nFINAL DATASET:")
print(df)


# ============================================================
# 6. PREPARE X AND Y
# ============================================================

X = df.drop(
    columns=[
        "Monthly_Electricity_Consumption_kWh",
        "Customer_ID"
    ]
)

# Convert categorical variables into numerical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

y = df["Monthly_Electricity_Consumption_kWh"]

print("\nINDEPENDENT VARIABLES (X):")
print(X.head())

print("\nDEPENDENT VARIABLE (Y):")
print(y.head())


# ============================================================
# 7. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING DATA SIZE:")
print(X_train.shape)

print("\nTESTING DATA SIZE:")
print(X_test.shape)


# ============================================================
# 8. LINEAR REGRESSION MODEL
# ============================================================

print("\n===== LINEAR REGRESSION =====")

model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("Linear Regression training completed.")

# Predict electricity consumption
y_pred = model.predict(X_test)

# Evaluate Linear Regression
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nSLOPE / COEFFICIENTS:")
print(model.coef_)

print("\nINTERCEPT:")
print(model.intercept_)

print("\nLINEAR REGRESSION RESULTS:")
print("MAE:", round(mae, 2), "kWh")
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2), "kWh")
print("R2 Score:", round(r2, 4))

print("\nACTUAL VALUES (first 10):")
print(y_test.head(10).values)


# ============================================================
# 9. RANDOM FOREST REGRESSION MODEL
# ============================================================

print("\n===== RANDOM FOREST REGRESSION =====")

# Create the model BEFORE training it
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train, y_train)

print("Random Forest training completed.")

# Predict electricity consumption
rf_pred = rf_model.predict(X_test)

# Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)

print("\nRANDOM FOREST RESULTS:")
print("MAE:", round(rf_mae, 2), "kWh")
print("MSE:", round(rf_mse, 2))
print("RMSE:", round(rf_rmse, 2), "kWh")
print("R2 Score:", round(rf_r2, 4))


# ============================================================
# 10. COMPARE BOTH MODELS
# ============================================================

print("\n===== MODEL COMPARISON =====")

comparison_metrics = pd.DataFrame({
    "Metric": ["MAE (kWh)", "MSE", "RMSE (kWh)", "R2 Score"],
    "Linear Regression": [mae, mse, rmse, r2],
    "Random Forest": [rf_mae, rf_mse, rf_rmse, rf_r2]
})

print(comparison_metrics.round(4).to_string(index=False))

# Select the model with the lower RMSE
if rf_rmse < rmse:
    print("\nBetter model by RMSE: Random Forest")
elif rmse < rf_rmse:
    print("\nBetter model by RMSE: Linear Regression")
else:
    print("\nBoth models have the same RMSE.")


# ============================================================
# 11. ACTUAL VS PREDICTED TABLE
# ============================================================

prediction_table = pd.DataFrame({
    "Actual Consumption (kWh)": y_test.values,
    "Linear Regression Prediction (kWh)": y_pred,
    "Random Forest Prediction (kWh)": rf_pred
})

print("\n===== ACTUAL VS PREDICTED VALUES =====")
print(prediction_table.head(20).round(2).to_string(index=False))


# ============================================================
# 12. GRAPH 1: LINEAR REGRESSION ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Electricity Consumption (kWh)")
plt.ylabel("Predicted Electricity Consumption (kWh)")
plt.title("Linear Regression: Actual vs Predicted")
plt.tight_layout()
plt.show()


# ============================================================
# 13. GRAPH 2: LINEAR REGRESSION RESIDUALS
# ============================================================

residuals = y_test.values - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(y_pred, residuals)
plt.axhline(y=0)

plt.xlabel("Predicted Electricity Consumption (kWh)")
plt.ylabel("Prediction Error (kWh)")
plt.title("Linear Regression Residual Plot")
plt.tight_layout()
plt.show()


# ============================================================
# 14. GRAPH 3: ACTUAL VS BOTH MODELS
# ============================================================

plt.figure(figsize=(10, 5))

# Display the first 50 test samples for readability
sample_count = min(50, len(y_test))
sample_indices = np.arange(sample_count)

plt.plot(
    sample_indices,
    y_test.iloc[:sample_count].values,
    label="Actual Consumption",
    marker="o"
)

plt.plot(
    sample_indices,
    y_pred[:sample_count],
    label="Linear Regression",
    marker="x"
)

plt.plot(
    sample_indices,
    rf_pred[:sample_count],
    label="Random Forest",
    marker="."
)

plt.xlabel("Test Sample")
plt.ylabel("Electricity Consumption (kWh)")
plt.title("Actual vs Predicted: Model Comparison")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 15. FEATURE IMPORTANCE: RANDOM FOREST
# ============================================================

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n===== TOP 10 IMPORTANT FEATURES =====")
print(feature_importance.head(10).round(4))

plt.figure(figsize=(9, 5))

feature_importance.head(10).sort_values().plot(kind="barh")

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 10 Features Affecting Predictions")
plt.tight_layout()
plt.show()

print("\n===== PROJECT EXECUTION COMPLETED =====")
