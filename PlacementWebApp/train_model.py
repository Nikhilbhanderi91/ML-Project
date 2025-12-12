# ---------------------------------------------------------------
#  TRAIN MODEL SCRIPT (train_model.py)
#  Converted from your Google Colab notebook into .py format
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------------
csv_path = "Dataset/student_placement_records.csv"     # UPDATE PATH IF NEEDED

df = pd.read_csv(csv_path)
print("Dataset Loaded Successfully!")
print(df.head(), "\n")
print(df.info())

# ---------------------------------------------------------------
# 2. ENCODE CATEGORICAL COLUMNS
# ---------------------------------------------------------------
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':          # Convert object/string columns
        df[col] = le.fit_transform(df[col])

df = df.fillna(0)

# ---------------------------------------------------------------
# 3. SPLIT FEATURES & TARGET
# ---------------------------------------------------------------
target_column = "placement_status"         # dataset uses lowercase column
X = df.drop(target_column, axis=1)
y = df[target_column]

# ---------------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------------------------------------
# 5. SCALING
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------------
# 6. TRAIN MULTIPLE MODELS
# ---------------------------------------------------------------
print("\nTraining Models...\n")

lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

svm = SVC()
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
pred_gb = gb.predict(X_test)

# ---------------------------------------------------------------
# 7. ACCURACY CALCULATIONS
# ---------------------------------------------------------------
print("\n--- MODEL ACCURACIES ---")
accuracies = {
    "Logistic Regression": accuracy_score(y_test, pred_lr),
    "Decision Tree": accuracy_score(y_test, pred_dt),
    "Random Forest": accuracy_score(y_test, pred_rf),
    "SVM": accuracy_score(y_test, pred_svm),
    "KNN": accuracy_score(y_test, pred_knn),
    "Gradient Boosting": accuracy_score(y_test, pred_gb)
}

for model, score in accuracies.items():
    print(f"{model}: {score * 100:.2f}%")

# ---------------------------------------------------------------
# 8. SAVE BEST MODEL (Gradient Boosting)
# ---------------------------------------------------------------
os.makedirs("model", exist_ok=True)

joblib.dump(gb, "model/placement_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nSaved Model → model/placement_model.pkl")
print("Saved Scaler → model/scaler.pkl")

# ---------------------------------------------------------------
# 9. PLOT MODEL COMPARISON GRAPH
# ---------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.bar(list(accuracies.keys()), list(accuracies.values()))
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("model_accuracy.png")     # Save graph to folder
print("\nSaved Graph → model_accuracy.png")

# ---------------------------------------------------------------
# 10. FEATURE IMPORTANCE (RANDOM FOREST)
# ---------------------------------------------------------------
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.savefig("feature_importance_rf.png")
print("Saved Graph → feature_importance_rf.png")

# ---------------------------------------------------------------
# 11. CONFUSION MATRIX (GB MODEL)
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, pred_gb)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Gradient Boosting")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_gb.png")
print("Saved Graph → confusion_matrix_gb.png")

print("\nTRAINING COMPLETED SUCCESSFULLY ✔\n")