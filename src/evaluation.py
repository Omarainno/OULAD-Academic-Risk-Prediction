import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

def split_groups(df, seed, test_size=0.20):
    # A student may appear in multiple module-presentation records.
    # Grouping by student reduces direct student-level leakage between train and test.
    students = df["id_student"].drop_duplicates().to_numpy()
    train_students, test_students = train_test_split(
        students, test_size=test_size, random_state=seed
    )
    train = df[df["id_student"].isin(train_students)].copy()
    test = df[df["id_student"].isin(test_students)].copy()
    return train, test

def metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
    }

def fit_models(X_train, y_train, X_test, y_test, specs, seed, n_jobs=-1):
    rows, fitted, preds = [], {}, {}
    for name, model in specs.items():
        from .models import build_pipeline
        pipe = build_pipeline(model, X_train)
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        p = pipe.predict_proba(X_test)[:, 1]
        m = metrics(y_test, p)
        rows.append({"model": name, **m, "fit_time_s": elapsed})
        fitted[name] = pipe
        preds[name] = p
    return pd.DataFrame(rows), fitted, preds

def permutation_table(model, X_test, y_test, repeats=10, seed=20260927):
    # Permutation importance is calculated on original, human-readable columns.
    r = permutation_importance(
        model, X_test, y_test, scoring="roc_auc",
        n_repeats=repeats, random_state=seed, n_jobs=-1
    )
    return pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": r.importances_mean,
        "importance_sd": r.importances_std,
    }).sort_values("importance_mean", ascending=False)

def paired_bootstrap_diff(y, p1, p2, repeats=1000, seed=20260927):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    diffs = []
    n = len(y)
    for _ in range(repeats):
        idx = rng.integers(0, n, n)
        a = roc_auc_score(y[idx], p1[idx])
        b = roc_auc_score(y[idx], p2[idx])
        diffs.append(a - b)
    diffs = np.asarray(diffs)
    return {
        "mean_auc_difference": float(diffs.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_value": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
    }

def paired_student_bootstrap_diff(df_test, y, p_full, p_static, repeats=1000, seed=20260927):
    """Student-level cluster bootstrap for paired ROC-AUC difference (Full - Static)."""
    rng = np.random.default_rng(seed)
    students = df_test["id_student"].to_numpy()
    unique_students = np.unique(students)
    y = np.asarray(y)
    p_full = np.asarray(p_full)
    p_static = np.asarray(p_static)

    student_to_idx = {}
    for idx, sid in enumerate(students):
        student_to_idx.setdefault(sid, []).append(idx)

    diffs = []
    n_students = len(unique_students)
    for _ in range(repeats):
        sampled_sids = rng.choice(unique_students, size=n_students, replace=True)
        boot_idx = [i for sid in sampled_sids for i in student_to_idx[sid]]
        auc_full = roc_auc_score(y[boot_idx], p_full[boot_idx])
        auc_static = roc_auc_score(y[boot_idx], p_static[boot_idx])
        diffs.append(auc_full - auc_static)

    diffs = np.asarray(diffs)
    return {
        "mean_auc_diff_full_minus_static": float(diffs.mean()),
        "ci_low_95": float(np.quantile(diffs, 0.025)),
        "ci_high_95": float(np.quantile(diffs, 0.975)),
        "p_value": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
    }

def holm_adjustment(rows):
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p_value"].clip(0,1), method="holm")[1]
    out["significant_holm_0_05"] = out["p_holm"] < 0.05
    return out