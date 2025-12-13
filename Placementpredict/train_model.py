# ---------------------------------------------------------------
#  TRAIN MODEL SCRIPT (train_model.py)
#  FIXED VERSION – FEATURE MISMATCH SOLVED
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
csv_path = "Dataset/student_placement_records.csv"

df = pd.read_csv(csv_path)
print("Dataset Loaded Successfully!\n")
print(df.head())
print(df.info())

# ---------------------------------------------------------------
# 2. DEFINE FINAL FEATURE LIST (30 FEATURES)
# ---------------------------------------------------------------
FEATURE_COLUMNS = [
    "tenth_percentage",
    "twelfth_percentage",
    "cgpa",
    "basic_aptitude",
    "icp_grade",
    "oops_grade",
    "dsa_grade",
    "daa_grade",
    "dbms_grade",
    "os_grade",
    "cn_grade",
    "iwt_grade",
    "cpmad_grade",
    "aj_grade",
    "awt_grade",
    "communication_skill",
    "aptitude_training",
    "coding_training",
    "tcs_google_score",
    "amcat_score",
    "hackathon_level",
    "leetcode_solved",
    "github_activity",
    "no_projects",
    "no_internships",
    "research_papers",
    "attendance",
    "mock_interview",
    "certifications_score",
    "personality_score"
]

TARGET_COLUMN = "placement_status"

# ---------------------------------------------------------------
# 3. ENCODE CATEGORICAL COLUMNS
# ---------------------------------------------------------------
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col].astype(str))

df = df.fillna(0)

# ---------------------------------------------------------------
# 4. SELECT FEATURES & TARGET (IMPORTANT FIX)
# ---------------------------------------------------------------
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# ---------------------------------------------------------------
# 5. TRAIN-TEST SPLIT
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------------------------------------
# 6. SCALING
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 7. TRAIN MODELS
# ---------------------------------------------------------------
print("\nTraining models...\n")

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_test_scaled)

dt = DecisionTreeClassifier()
dt.fit(X_train_scaled, y_train)
pred_dt = dt.predict(X_test_scaled)

rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)
pred_rf = rf.predict(X_test_scaled)

svm = SVC()
svm.fit(X_train_scaled, y_train)
pred_svm = svm.predict(X_test_scaled)

knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
pred_knn = knn.predict(X_test_scaled)

gb = GradientBoostingClassifier()
gb.fit(X_train_scaled, y_train)
pred_gb = gb.predict(X_test_scaled)

# ---------------------------------------------------------------
# 8. ACCURACY COMPARISON
# ---------------------------------------------------------------
accuracies = {
    "Logistic Regression": accuracy_score(y_test, pred_lr),
    "Decision Tree": accuracy_score(y_test, pred_dt),
    "Random Forest": accuracy_score(y_test, pred_rf),
    "SVM": accuracy_score(y_test, pred_svm),
    "KNN": accuracy_score(y_test, pred_knn),
    "Gradient Boosting": accuracy_score(y_test, pred_gb)
}

print("\n--- MODEL ACCURACIES ---")
for model, acc in accuracies.items():
    print(f"{model}: {acc*100:.2f}%")

# ---------------------------------------------------------------
# 9. SAVE MODEL, SCALER & FEATURE LIST
# ---------------------------------------------------------------
os.makedirs("model", exist_ok=True)

joblib.dump(gb, "model/placement_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(FEATURE_COLUMNS, "model/feature_columns.pkl")

print("\nSaved files:")
print("✔ model/placement_model.pkl")
print("✔ model/scaler.pkl")
print("✔ model/feature_columns.pkl")

# ---------------------------------------------------------------
# 10. ACCURACY BAR GRAPH
# ---------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.bar(accuracies.keys(), accuracies.values())
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("model_accuracy.png")

# ---------------------------------------------------------------
# 11. FEATURE IMPORTANCE (RANDOM FOREST)
# ---------------------------------------------------------------
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(len(FEATURE_COLUMNS)), importances[indices])
plt.xticks(range(len(FEATURE_COLUMNS)),
           [FEATURE_COLUMNS[i] for i in indices],
           rotation=90)
plt.tight_layout()
plt.savefig("feature_importance_rf.png")

# ---------------------------------------------------------------
# 12. CONFUSION MATRIX (GRADIENT BOOSTING)
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, pred_gb)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Gradient Boosting")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_gb.png")

print("\nTRAINING COMPLETED SUCCESSFULLY ✔\n")