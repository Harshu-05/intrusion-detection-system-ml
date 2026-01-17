import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
train_data = pd.read_csv("kdd_train.csv")
test_data = pd.read_csv("kdd_test.csv")

print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)

# -----------------------------
# STEP 2: Convert Labels
# normal -> 0
# attack -> 1
# -----------------------------
train_data["labels"] = train_data["labels"].apply(
    lambda x: 0 if x == "normal" else 1
)
test_data["labels"] = test_data["labels"].apply(
    lambda x: 0 if x == "normal" else 1
)

print("\nTraining label distribution:")
print(train_data["labels"].value_counts())

print("\nTesting label distribution:")
print(test_data["labels"].value_counts())

# -----------------------------
# STEP 3: Encode Categorical Columns
# -----------------------------
categorical_cols = ["protocol_type", "service", "flag"]

encoder = LabelEncoder()

for col in categorical_cols:
    train_data[col] = encoder.fit_transform(train_data[col])
    test_data[col] = encoder.transform(test_data[col])

# -----------------------------
# STEP 4: Split Features & Labels
# -----------------------------
X_train = train_data.drop("labels", axis=1)
y_train = train_data["labels"]

X_test = test_data.drop("labels", axis=1)
y_test = test_data["labels"]

# -----------------------------
# STEP 5: Train ML Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("\nTraining IDS model...")
model.fit(X_train, y_train)

# -----------------------------
# STEP 6: Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# STEP 7: Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy * 100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
