import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def make_preprocessor(X):
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", cat_pipe, categorical),
        ("num", num_pipe, numeric),
    ], remainder="drop")

def model_specs(seed=20260927, n_jobs=-1):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight=None, random_state=seed
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=10, random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, max_features="sqrt",
            class_weight=None, random_state=seed, n_jobs=n_jobs
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            objective="binary:logistic", eval_metric="logloss",
            random_state=seed, n_jobs=n_jobs, tree_method="hist"
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            objective="binary", random_state=seed, n_jobs=n_jobs,
            verbosity=-1
        ),
    }

def build_pipeline(model, X, scale=True):
    pre = make_preprocessor(X)
    return Pipeline([("preprocess", pre), ("model", model)])
