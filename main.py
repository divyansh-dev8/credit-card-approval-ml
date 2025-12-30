import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("data/credit_card_approval.csv")

print("Shape of data:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# =========================
# 2. DROP ID COLUMN
# =========================
df = df.drop(columns=["Applicant_ID"])

# =========================
# 3. CHECK MISSING VALUES
# =========================
print("\nMissing values per column:")
print(df.isnull().sum())

# =========================
# 4. SEPARATE X AND y
# =========================
X = df.drop(columns=["Status"])
y = df["Status"]

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

# =========================
# ==========================
# 5. ONE-HOT ENCODING
# ==========================

X_encoded = pd.get_dummies(X, drop_first=True)

import joblib
joblib.dump(X_encoded.columns.tolist(), "columns.pkl")

print("\nShape after encoding:", X_encoded.shape)
# =========================
# 6. COMBINE X & y FOR BALANCING
# =========================
data = X_encoded.copy()
data["Status"] = y.values

approved = data[data["Status"] == 1]
rejected = data[data["Status"] == 0]

print("\nBefore balancing:")
print("Approved:", approved.shape)
print("Rejected:", rejected.shape)

# =========================
# 7. UNDERSAMPLING (2:1 RATIO)
# =========================
from sklearn.utils import resample

approved_downsampled = resample(
    approved,
    replace=False,
    n_samples=len(rejected) * 2,
    random_state=42
)

balanced_data = pd.concat([approved_downsampled, rejected])

print("\nAfter balancing:")
print(balanced_data["Status"].value_counts())

# =========================
# 8. SPLIT AGAIN X & y
# =========================
X_balanced = balanced_data.drop(columns=["Status"])
y_balanced = balanced_data["Status"]

# =========================
# 9. TRAIN–TEST SPLIT
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced,
    y_balanced,
    test_size=0.2,
    random_state=42,
    stratify=y_balanced
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# =========================
# 10. FEATURE SCALING
# =========================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 11. LOGISTIC REGRESSION
# =========================
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

print("\nModel training completed")

# =========================
# 12. EVALUATION
# =========================
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import joblib

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully")