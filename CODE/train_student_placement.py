# train_student_placement.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

RND = 42
CSV_PATH = "/Users/nikhilbhanderi/Documents/ML PROJECT/Dataset/student_placement_records.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Put the CSV at: {CSV_PATH}")

# 1) Load data
df = pd.read_csv(CSV_PATH)
print("Loaded:", df.shape)

# 2) Basic cleanup: drop duplicate student_id if any
if 'student_id' in df.columns:
    df = df.drop_duplicates(subset=['student_id']).reset_index(drop=True)

# 3) Map letter grades to numeric if needed
grade_map = {
    'A+':4.0,'A':4.0,'A-':3.7,'B+':3.3,'B':3.0,'B-':2.7,'C':2.0,'D':1.0,'F':0.0
}
grade_cols = [c for c in df.columns if c.startswith('grade_')]
for c in grade_cols:
    if df[c].dtype == object or df[c].dtype.name == 'category':
        df[c] = df[c].map(grade_map).astype(float)

# 4) Fill missing columns with defaults (robustness)
expected_cols = [
    "gender","percentage_10","percentage_12","cgpa_sem","basic_aptitude"
] + grade_cols + [
    "communication_skill","aptitude_training_perf","coding_training_perf",
    "national_test_score","amcat_score","num_hackathons","leetcode_solved",
    "github_contrib_score","num_projects","project_depth","num_internships",
    "internship_quality","research_papers","certification_count",
    "attendance_percent","mock_interview_score","personality_score"
]
for c in expected_cols:
    if c not in df.columns:
        print(f"Warning: adding missing column {c} with zeros")
        df[c] = 0

# Ensure target exists
if 'placement_status' not in df.columns:
    raise ValueError("CSV must contain 'placement_status' (0/1)")

# 5) Feature engineering - composite scores
core_grade_cols = [c for c in grade_cols if c in df.columns]
df['technical_score'] = df[core_grade_cols].mean(axis=1)  # mean of technical grades
df['softskill_score'] = df[['communication_skill','personality_score','mock_interview_score']].mean(axis=1)
df['experience_score'] = df['num_internships'] + np.log1p(df['num_projects'])

# Add composite names to numeric list
numeric_cols = [
    "percentage_10","percentage_12","cgpa_sem","basic_aptitude",
    "communication_skill","aptitude_training_perf","coding_training_perf",
    "national_test_score","amcat_score","num_hackathons","leetcode_solved",
    "github_contrib_score","num_projects","project_depth","num_internships",
    "internship_quality","research_papers","certification_count","attendance_percent",
    "mock_interview_score","personality_score","technical_score","softskill_score","experience_score"
]
# remove any not in df
numeric_cols = [c for c in numeric_cols if c in df.columns]

# categorical
categorical_cols = [c for c in ['gender'] if c in df.columns]

# X, y
X = df[numeric_cols + categorical_cols].copy()
y = df['placement_status'].astype(int)

# 6) Train/test split (70/30 to match paper)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RND, stratify=y)
print("Train/test:", X_train.shape, X_test.shape)

# 7) Preprocessing pipeline
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer([
    ('num', num_transformer, numeric_cols),
    ('cat', cat_transformer, categorical_cols)
], remainder='drop')

# 8) Models (as in paper: existing + proposed)
models = {
    'LogisticRegression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RND),
    'DecisionTree': DecisionTreeClassifier(random_state=RND),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=RND, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(random_state=RND),
    'SVM': SVC(probability=True, random_state=RND),
    'KNN': KNeighborsClassifier(),
    'NaiveBayes': GaussianNB()
}

results = []
fitted_pipelines = {}

for name, clf in models.items():
    print(f"\nTraining {name} ...")
    pipe = Pipeline([('preproc', preprocessor), ('clf', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # probability/AUC if available
    try:
        y_proba = pipe.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        y_proba = None
        auc = None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"{name} -> Acc:{acc:.4f} | Prec:{prec:.4f} | Rec:{rec:.4f} | F1:{f1:.4f} | AUC:{auc}")
    results.append({
        'model': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc
    })
    fitted_pipelines[name] = pipe

# 9) Results dataframe
res_df = pd.DataFrame(results).sort_values(by='f1', ascending=False).reset_index(drop=True)
print("\nModel comparison (sorted by F1):")
print(res_df)

# Save comparison table
res_df.to_csv("model_comparison_results.csv", index=False)
print("Saved model_comparison_results.csv")

# 10) Choose best by F1 and save model
best_model_name = res_df.loc[0, 'model']
best_pipeline = fitted_pipelines[best_model_name]
joblib.dump(best_pipeline, "best_model.pkl")
print(f"Best model: {best_model_name} saved to best_model.pkl")

# 11) Detailed report for best model
print(f"\n=== Classification report for {best_model_name} ===")
y_pred_best = best_pipeline.predict(X_test)
print(classification_report(y_test, y_pred_best))

# 12) Confusion matrix plot
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title(f"{best_model_name} Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0,1], ['Not Placed','Placed'])
plt.yticks([0,1], ['Not Placed','Placed'])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.savefig("confusion_matrix_best.png", dpi=150)
print("Saved confusion_matrix_best.png")
plt.show()

# 13) If best is tree-based, show top features
clf = best_pipeline.named_steps['clf']
if hasattr(clf, 'feature_importances_'):
    importances = clf.feature_importances_
    # build feature names after preprocessing
    # numeric names are numeric_cols
    cat_ohe_names = []
    if len(categorical_cols) > 0:
        # extract onehot feature names
        ohe = best_pipeline.named_steps['preproc'].named_transformers_['cat'].named_steps['ohe']
        cat_ohe_names = list(ohe.get_feature_names_out(categorical_cols))
    feat_names = numeric_cols + cat_ohe_names
    feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(25)
    plt.figure(figsize=(8,6))
    feat_imp.plot.barh()
    plt.gca().invert_yaxis()
    plt.title("Top feature importances")
    plt.tight_layout()
    plt.savefig("feature_importances_best.png", dpi=150)
    print("Saved feature_importances_best.png")
    plt.show()

print("All done. Outputs: best_model.pkl, model_comparison_results.csv, confusion_matrix_best.png (and feature_importances_best.png if available).")