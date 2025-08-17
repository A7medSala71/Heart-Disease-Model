"""
Comprehensive ML Pipeline for the UCI Heart Disease Dataset (id=45 via ucimlrepo)
---------------------------------------------------------------------------------
This script:
- Loads data from UCI via ucimlrepo
- Cleans & preprocesses (impute, encode, scale) via ColumnTransformer pipelines
- EDA (histograms, correlation, boxplots)
- PCA (variance plot + 2D scatter)
- Feature selection (RF importance, RFE, Chi^2)
- Supervised models: Logistic Regression, Decision Tree, Random Forest, SVM
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curves
- Unsupervised: KMeans (elbow), Hierarchical (dendrogram), ARI vs labels
- Hyperparameter tuning (RandomizedSearchCV/GridSearchCV)
- Exports best full pipeline (preprocessor + model) to models/final_model.pkl
- Saves the training column names to models/train_columns.json
- Optionally launches Streamlit UI (ui_app.py)

Run:
    python main.py                # trains + launches UI
    python main.py --no-ui        # trains only
    python main.py --ui-only      # skips training and launches UI

Notes:
- Target is binarized: presence = 1 if 'num' > 0 else 0 (if needed).
"""

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

# --------------------------------
# 1) Imports
# --------------------------------
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import sys

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram


# --------------------------------
# CLI
# --------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--no-ui", action="store_true", help="Train only; do not launch Streamlit UI.")
parser.add_argument("--ui-only", action="store_true", help="Skip training and launch Streamlit UI.")
args = parser.parse_args()

# --------------------------------
# 2) Paths
# --------------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

MODEL_PATH = "models/final_model.pkl"
TRAIN_COLS_PATH = "models/train_columns.json"

def launch_ui():
    # Launch ui_app.py via streamlit
    print("\nLaunching Streamlit UI...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ui_app.py"], check=True)
    except FileNotFoundError:
        print("Error: Streamlit not found. Install with `pip install streamlit`.")
    except subprocess.CalledProcessError as e:
        print(f"Streamlit exited with error: {e}")

if args.ui_only:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(TRAIN_COLS_PATH)):
        print("Model/UI artifacts not found. Train first by running `python main.py`.")
        sys.exit(1)
    launch_ui()
    sys.exit(0)

# --------------------------------
# 3) Load dataset (UCI id=45)
# --------------------------------
heart = fetch_ucirepo(id=45)  # UCI Heart Disease
X = heart.data.features.copy()
y = heart.data.targets.copy()

# Try to identify the target column robustly
if isinstance(y, pd.DataFrame):
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    else:
        y = y.iloc[:, 0]
elif isinstance(y, pd.Series):
    pass
else:
    y = pd.Series(y, name="target")

# Standardize target to binary 0/1:
if "num" in X.columns:
    y = (X["num"].astype(float) > 0).astype(int)
    X = X.drop(columns=["num"])
elif y.nunique() > 2:
    y = (y.astype(float) > 0).astype(int)
else:
    y = y.astype(int)

# Save a copy for reference
df = pd.concat([X, y.rename("target")], axis=1)
df.to_csv("data/heart_disease_raw.csv", index=False)

print(f"Data shape: X={X.shape}, y={y.shape}, positive rate={y.mean():.3f}")

# --------------------------------
# 4) EDA (basic)
# --------------------------------
def plot_histograms(dataframe, outpath="results/eda_histograms.png"):
    plt.figure(figsize=(14, 10))
    dataframe.hist(bins=20, figsize=(14, 10))
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_correlation(dataframe, outpath="results/eda_correlation.png"):
    num_df = dataframe.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        sns.heatmap(num_df.corr(), annot=False)
        plt.title("Correlation Heatmap (Numeric Features)")
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()

def plot_boxplots(dataframe, outdir="results"):
    num_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        plt.figure()
        sns.boxplot(x=dataframe[col].dropna())
        plt.title(f"Boxplot: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"box_{col}.png"), dpi=140)
        plt.close()

print("Running EDA ...")
plot_histograms(df.drop(columns=["target"], errors="ignore"))
plot_correlation(df)
plot_boxplots(df.drop(columns=["target"], errors="ignore"))
print("EDA figures saved under results/")

# --------------------------------
# 5) Train/Test split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# --------------------------------
# 6) Column typing (auto-detect)
# --------------------------------
def infer_column_types(X_df):
    categorical_cols, numeric_cols = [], []
    for col in X_df.columns:
        if X_df[col].dtype == "object":
            categorical_cols.append(col)
        else:
            nunique = X_df[col].nunique(dropna=True)
            if nunique <= 10 and not np.issubdtype(X_df[col].dtype, np.floating):
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
    return categorical_cols, numeric_cols

categorical_cols, numeric_cols = infer_column_types(X_train)
print("Detected categorical:", categorical_cols)
print("Detected numeric:", numeric_cols)

# --------------------------------
# 7) Preprocessor (impute + encode + scale)
# --------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ],
    remainder="drop"
)

# Utility to get feature names after preprocessing
def get_preprocessed_feature_names(preprocessor, X_fit_df):
    preprocessor.fit(X_fit_df)
    names = []
    if "num" in [t[0] for t in preprocessor.transformers]:
        names += list(numeric_cols)
    if "cat" in [t[0] for t in preprocessor.transformers]:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        if hasattr(ohe, "get_feature_names_out"):
            cat_names = ohe.get_feature_names_out(categorical_cols)
            names += list(cat_names)
    return names

# --------------------------------
# 8) PCA (on preprocessed data)
# --------------------------------
pca_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("pca", PCA(n_components=0.95, random_state=42))
])

X_train_pca = pca_pipeline.fit_transform(X_train)
X_test_pca = pca_pipeline.transform(X_test)
explained = pca_pipeline.named_steps["pca"].explained_variance_ratio_

plt.figure()
plt.plot(np.cumsum(explained), marker="o")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA – Cumulative Explained Variance (>=95%)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("results/pca_cumulative_variance.png", dpi=150)
plt.close()

if X_train_pca.shape[1] >= 2:
    plt.figure()
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, alpha=0.8)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA – First Two Components (Train)")
    plt.tight_layout()
    plt.savefig("results/pca_scatter_train.png", dpi=150)
    plt.close()

print("PCA results saved under results/")

# --------------------------------
# 9) Feature Selection
# --------------------------------
rf_for_importance = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42))
])
rf_for_importance.fit(X_train, y_train)

feature_names = get_preprocessed_feature_names(preprocessor, X_train)
rf_importances = rf_for_importance.named_steps["rf"].feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": rf_importances}) \
         .sort_values("importance", ascending=False)
imp_df.to_csv("results/feature_importance_random_forest.csv", index=False)

plt.figure(figsize=(8, min(10, 0.35*len(imp_df))))
topn = imp_df.head(20)
sns.barplot(data=topn, x="importance", y="feature")
plt.title("Top 20 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("results/feature_importance_rf_top20.png", dpi=150)
plt.close()

logreg_est = LogisticRegression(max_iter=2000, solver="liblinear")
rfe_selector = RFE(estimator=logreg_est, n_features_to_select=min(15, len(feature_names)))
rfe_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("rfe", rfe_selector),
    ("clf", logreg_est)
])
rfe_pipeline.fit(X_train, y_train)
rfe_mask = rfe_pipeline.named_steps["rfe"].support_
rfe_features = np.array(feature_names)[rfe_mask]
pd.Series(rfe_features).to_csv("results/rfe_selected_features.csv", index=False)

chi_preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")),
                          ("minmax", MinMaxScaler())]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols)
    ],
    remainder="drop"
)
chi_pipeline = Pipeline(steps=[
    ("preprocessor", chi_preprocess),
    ("chi", SelectKBest(score_func=chi2, k=min(15, len(feature_names))))
])
chi_pipeline.fit(X_train, y_train)
chi_support = chi_pipeline.named_steps["chi"].get_support()
chi_features = np.array(get_preprocessed_feature_names(chi_preprocess, X_train))[chi_support]
pd.Series(chi_features).to_csv("results/chi2_selected_features.csv", index=False)

print("Feature selection artifacts saved under results/")

# --------------------------------
# 10) Supervised Models (Baseline)
# --------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, solver="liblinear"),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42)
}

def build_clf_pipeline(clf):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])

metrics_rows = []
for name, clf in models.items():
    pipe = build_clf_pipeline(clf)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["clf"], "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if y_proba is not None:
        auc_score = roc_auc_score(y_test, y_proba)
    else:
        if hasattr(pipe.named_steps["clf"], "decision_function"):
            y_score = pipe.decision_function(X_test)
            auc_score = roc_auc_score(y_test, y_score)
        else:
            auc_score = np.nan

    metrics_rows.append([name, acc, pre, rec, f1, auc_score])

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    plt.savefig(f"results/cm_{name}.png", dpi=150)
    plt.close()

    if not np.isnan(auc_score):
        fpr, tpr, _ = roc_curve(y_test, y_proba if y_proba is not None else y_score)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score, estimator_name=name).plot()
        plt.title(f"ROC Curve – {name} (AUC={auc_score:.3f})")
        plt.tight_layout()
        plt.savefig(f"results/roc_{name}.png", dpi=150)
        plt.close()

metrics_df = pd.DataFrame(metrics_rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"])
metrics_df.to_csv("results/supervised_metrics_baseline.csv", index=False)
print("Baseline supervised metrics:\n", metrics_df)

# --------------------------------
# 11) Unsupervised Clustering
# --------------------------------
preprocessor_fitted = preprocessor.fit(X_train, y_train)
X_train_pre = preprocessor_fitted.transform(X_train)
X_test_pre = preprocessor_fitted.transform(X_test)
X_all_pre = preprocessor_fitted.transform(X)

inertias = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_all_pre)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(list(K_range), inertias, marker="o")
plt.xlabel("k"); plt.ylabel("Inertia"); plt.title("KMeans – Elbow Method")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("results/kmeans_elbow.png", dpi=150)
plt.close()

km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = km2.fit_predict(X_all_pre)
ari_kmeans = adjusted_rand_score(y, kmeans_labels)

sample_idx = np.random.RandomState(42).choice(np.arange(X_all_pre.shape[0]),
                                              size=min(150, X_all_pre.shape[0]),
                                              replace=False)
X_sample = X_all_pre[sample_idx]
Z = linkage(X_sample, method="ward")
plt.figure(figsize=(10, 6))
dendrogram(Z, truncate_mode="level", p=5)
plt.title("Hierarchical Clustering – Dendrogram (sampled)")
plt.tight_layout()
plt.savefig("results/hierarchical_dendrogram.png", dpi=150)
plt.close()

agg = AgglomerativeClustering(n_clusters=2, linkage="ward")
agg_labels = agg.fit_predict(X_all_pre.toarray() if hasattr(X_all_pre, "toarray") else X_all_pre)
ari_agg = adjusted_rand_score(y, agg_labels)

with open("results/unsupervised_summary.txt", "w") as f:
    f.write(f"Adjusted Rand Index (KMeans, k=2): {ari_kmeans:.4f}\n")
    f.write(f"Adjusted Rand Index (Agglomerative, k=2): {ari_agg:.4f}\n")

print(f"Unsupervised ARI – KMeans: {ari_kmeans:.3f}, Agglomerative: {ari_agg:.3f}")

# --------------------------------
# 12) Hyperparameter Tuning
# --------------------------------
scoring = "roc_auc"

rf_pipe = build_clf_pipeline(RandomForestClassifier(random_state=42))
rf_param_dist = {
    "clf__n_estimators": [200, 300, 400, 600],
    "clf__max_depth": [None, 4, 6, 8, 12],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2", None]
}
rf_search = RandomizedSearchCV(
    rf_pipe, rf_param_dist, n_iter=25, cv=5, scoring=scoring,
    random_state=42, n_jobs=-1, verbose=1
)
rf_search.fit(X_train, y_train)

svm_pipe = build_clf_pipeline(SVC(probability=True, random_state=42))
svm_param_grid = {
    "clf__C": [0.1, 1, 3, 10],
    "clf__gamma": ["scale", "auto", 0.01, 0.001],
    "clf__kernel": ["rbf"]
}
svm_grid = GridSearchCV(
    svm_pipe, svm_param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1
)
svm_grid.fit(X_train, y_train)

logr_pipe = build_clf_pipeline(LogisticRegression(max_iter=2000, solver="liblinear"))
logr_param_grid = {
    "clf__C": [0.1, 0.5, 1, 3, 10],
    "clf__penalty": ["l1", "l2"]
}
logr_grid = GridSearchCV(
    logr_pipe, logr_param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=0
)
logr_grid.fit(X_train, y_train)

def evaluate_on_test(search_obj, name):
    best = search_obj.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "BestParams": search_obj.best_params_,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba)
    }

tuned_results = []
tuned_results.append(evaluate_on_test(rf_search, "RandomForest (Tuned)"))
tuned_results.append(evaluate_on_test(svm_grid, "SVM (Tuned)"))
tuned_results.append(evaluate_on_test(logr_grid, "LogisticRegression (Tuned)"))
tuned_df = pd.DataFrame(tuned_results)
tuned_df.to_csv("results/supervised_metrics_tuned.csv", index=False)
print("Tuned model metrics:\n", tuned_df)

best_row = tuned_df.sort_values("ROC_AUC", ascending=False).iloc[0]
best_name = best_row["Model"]
print(f"Best tuned model: {best_name}")

if "RandomForest" in best_name:
    best_pipeline = rf_search.best_estimator_
elif "SVM" in best_name:
    best_pipeline = svm_grid.best_estimator_
else:
    best_pipeline = logr_grid.best_estimator_

# --------------------------------
# 13) Export final pipeline (.pkl) + training columns
# --------------------------------
joblib.dump(best_pipeline, MODEL_PATH)

# Save original training columns so UI can reindex inputs safely
with open(TRAIN_COLS_PATH, "w") as f:
    json.dump({"train_columns": list(X.columns)}, f, indent=2)

with open("results/evaluation_metrics.txt", "w") as f:
    f.write("=== Baseline Models ===\n")
    f.write(metrics_df.to_string(index=False))
    f.write("\n\n=== Tuned Models ===\n")
    f.write(tuned_df.to_string(index=False))
    f.write(f"\n\nSelected Best Model: {best_name}\n")

print("Saved best full pipeline to models/final_model.pkl")
print("All artifacts written to ./results and ./models")

# --------------------------------
# 14) Optionally launch UI
# --------------------------------
if not args.no_ui:
    launch_ui()
