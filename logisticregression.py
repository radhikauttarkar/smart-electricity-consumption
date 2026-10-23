import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("Smart_Household_Electricity_Consumption_Dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# --------------------------------------------------
# 2. Check Missing Values
# --------------------------------------------------

features = [
    "Temperature_C",
    "Occupancy_Hours",
    "AC_Usage_Hours",
    "TV_Usage_Hours",
    "Water_Heater_Usage_Hours",
    "Computer_Usage_Hours"
]

target = "Monthly_Electricity_Consumption_kWh"

print("\nMissing Values:")
print(df[features + [target]].isnull().sum())


# Remove rows containing missing values
df = df.dropna(subset=features + [target])


# --------------------------------------------------
# 3. Create Consumption Level
# --------------------------------------------------
# 0 = Low Consumption
# 1 = High Consumption

threshold = df[target].median()

print("\nMedian Electricity Consumption:", threshold)

df["Consumption_Level"] = (
    df[target] > threshold
).astype(int)

print("\nConsumption Level:")
print(
    df[
        [target, "Consumption_Level"]
    ].head()
)


# --------------------------------------------------
# 4. Select Input Features
# --------------------------------------------------

X = df[features]


# --------------------------------------------------
# 5. Select Target
# --------------------------------------------------

y = df["Consumption_Level"]


# --------------------------------------------------
# 6. Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# --------------------------------------------------
# 7. Create Logistic Regression Model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000
)


# --------------------------------------------------
# 8. Train Model
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# --------------------------------------------------
# 9. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 10. Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")


# --------------------------------------------------
# 11. Actual vs Predicted
# --------------------------------------------------

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)


# --------------------------------------------------
# 12. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# 13. Classification Report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Low", "High"]
    )
)


# --------------------------------------------------
# 14. Model Coefficients
# --------------------------------------------------

print("\nModel Coefficients:")

for feature, coefficient in zip(
    features,
    model.coef_[0]
):
    print(
        f"{feature}: {coefficient:.4f}"
    )

print("\nIntercept:")
print(model.intercept_[0])


# --------------------------------------------------
# 15. Prediction Probabilities
# --------------------------------------------------

probability = model.predict_proba(X_test)

print("\nPrediction Probabilities:")
print(probability[:10])


# --------------------------------------------------
# 16. Plot Confusion Matrix
# --------------------------------------------------

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix - Logistic Regression")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["Low", "High"]
)

plt.yticks(
    [0, 1],
    ["Low", "High"]
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.show()