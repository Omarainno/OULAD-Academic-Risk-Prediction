from pathlib import Path
import json, logging, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import load_config
from .data import load_oulad, download_oulad, raw_ready, validate_schema
from .features import build_dataset, feature_groups
from .models import model_specs
from .evaluation import (
    split_groups, fit_models, metrics, permutation_table,
    paired_bootstrap_diff, paired_student_bootstrap_diff, holm_adjustment
)

def setup_logging(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()]
    )

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def main():
    cfg = load_config()
    root = cfg["root"]
    raw = root / "data" / "raw"
    out = root / "results"
    setup_logging(out / "logs" / "pipeline.log")
    log = logging.getLogger("paper2")

    seed = int(cfg["seed"])
    np.random.seed(seed)
    t0 = time.perf_counter()

    if not raw_ready(raw):
        if cfg.get("download_if_missing", True):
            log.info("OULAD files missing; downloading the public UCI archive.")
            download_oulad(raw)
        else:
            raise FileNotFoundError("OULAD CSV files are missing from data/raw/.")

    d = load_oulad(raw)
    validate_schema(d)
    log.info("Loaded OULAD tables.")

    df = build_dataset(d, boundary_fraction=float(cfg["boundary_fraction"]))
    df.to_csv(root / "data" / "processed" / "paper2_50pct_features.csv", index=False)

    groups = feature_groups(df)
    dynamic = groups["dynamic"]
    static = groups["static"]
    target = "risk"
    id_cols = ["id_student", "code_module", "code_presentation", "module_presentation_length", "cutoff_day", "date_unregistration", target]

    X = df.drop(columns=id_cols, errors="ignore")
    y = df[target].astype(int)

    train, test = split_groups(df, seed, float(cfg["test_size"]))
    X_train = train.drop(columns=id_cols, errors="ignore")
    y_train = train[target].astype(int)
    X_test = test.drop(columns=id_cols, errors="ignore")
    y_test = test[target].astype(int)

    cohort = pd.DataFrame([{
        "boundary_fraction": cfg["boundary_fraction"],
        "cohort_records": len(df),
        "at_risk": int(y.sum()),
        "non_risk": int((1-y).sum()),
        "at_risk_percent": float(y.mean()*100),
        "students_train": int(train["id_student"].nunique()),
        "students_test": int(test["id_student"].nunique()),
        "train_records": len(train),
        "test_records": len(test),
        "feature_count_total": X.shape[1],
        "static_feature_count": len(static),
        "dynamic_feature_count": len(dynamic),
    }])
    save_csv(cohort, out/"tables"/"01_cohort_summary.csv")

    # Detailed Predictor Dictionary (Reviewer 1, Point 2)
    predictor_meta = [
        {"feature": "gender", "group": "Static", "source": "studentInfo", "calculation": "Binary category (M/F)", "temporal_rule": "Baseline", "missing_handling": "None (complete)"},
        {"feature": "region", "group": "Static", "source": "studentInfo", "calculation": "Categorical region", "temporal_rule": "Baseline", "missing_handling": "None (complete)"},
        {"feature": "highest_education", "group": "Static", "source": "studentInfo", "calculation": "Ordinal education level", "temporal_rule": "Baseline", "missing_handling": "None (complete)"},
        {"feature": "imd_band", "group": "Static", "source": "studentInfo", "calculation": "Index of Multiple Deprivation band", "temporal_rule": "Baseline", "missing_handling": "Imputed / Categorical"},
        {"feature": "age_band", "group": "Static", "source": "studentInfo", "calculation": "Categorical age group", "temporal_rule": "Baseline", "missing_handling": "None (complete)"},
        {"feature": "num_of_prev_attempts", "group": "Static", "source": "studentInfo", "calculation": "Integer count of previous attempts", "temporal_rule": "Baseline", "missing_handling": "None (complete)"},
        {"feature": "studied_credits", "group": "Static", "source": "studentInfo", "calculation": "Integer credit value", "temporal_rule": "Baseline", "missing_handling": "None (complete)"},
        {"feature": "disability", "group": "Static", "source": "studentInfo", "calculation": "Binary indicator (Y/N)", "temporal_rule": "Baseline", "missing_handling": "None (complete)"},
        {"feature": "total_clicks", "group": "Dynamic (Behavioural)", "source": "vle & studentVle", "calculation": "Sum of sum_click", "temporal_rule": "Bounded by date <= cutoff_day", "missing_handling": "Imputed as 0"},
        {"feature": "active_days", "group": "Dynamic (Behavioural)", "source": "studentVle", "calculation": "Count of unique active days", "temporal_rule": "Bounded by date <= cutoff_day", "missing_handling": "Imputed as 0"},
        {"feature": "unique_sites", "group": "Dynamic (Behavioural)", "source": "studentVle & vle", "calculation": "Count of unique id_site accessed", "temporal_rule": "Bounded by date <= cutoff_day", "missing_handling": "Imputed as 0"},
        {"feature": "active_weeks", "group": "Dynamic (Behavioural)", "source": "studentVle", "calculation": "Ceil((max_date + 1) / 7)", "temporal_rule": "Bounded by date <= cutoff_day", "missing_handling": "Imputed as 0"},
        {"feature": "mean_daily_clicks", "group": "Dynamic (Behavioural)", "source": "studentVle", "calculation": "Mean clicks per interaction day", "temporal_rule": "Bounded by date <= cutoff_day", "missing_handling": "Imputed as 0"},
        {"feature": "max_daily_clicks", "group": "Dynamic (Behavioural)", "source": "studentVle", "calculation": "Maximum clicks on any single day", "temporal_rule": "Bounded by date <= cutoff_day", "missing_handling": "Imputed as 0"},
        {"feature": "unique_activity_types", "group": "Dynamic (Behavioural)", "source": "studentVle & vle", "calculation": "Count of unique activity_type", "temporal_rule": "Bounded by date <= cutoff_day", "missing_handling": "Imputed as 0"},
        {"feature": "clicks_per_active_day", "group": "Dynamic (Behavioural)", "source": "studentVle", "calculation": "total_clicks / active_days", "temporal_rule": "Bounded by date <= cutoff_day", "missing_handling": "Imputed as 0"},
        {"feature": "active_day_ratio", "group": "Dynamic (Behavioural)", "source": "studentVle", "calculation": "active_days / cutoff_day", "temporal_rule": "Bounded by date <= cutoff_day", "missing_handling": "Imputed as 0"},
        {"feature": "max_daily_total_clicks", "group": "Dynamic (Behavioural)", "source": "studentVle", "calculation": "Max daily aggregated sum_click", "temporal_rule": "Bounded by date <= cutoff_day", "missing_handling": "Imputed as 0"},
        {"feature": "assessments_seen", "group": "Dynamic (Assessment)", "source": "assessments", "calculation": "Count of unique assessments due <= cutoff_day", "temporal_rule": "Assessment due date <= cutoff_day", "missing_handling": "Imputed as 0"},
        {"feature": "assessment_submissions", "group": "Dynamic (Assessment)", "source": "studentAssessment", "calculation": "Count of submissions recorded <= cutoff_day", "temporal_rule": "Submission date <= cutoff_day", "missing_handling": "Imputed as 0"},
        {"feature": "mean_assessment_score", "group": "Dynamic (Assessment)", "source": "studentAssessment", "calculation": "Mean score across submissions", "temporal_rule": "Submission recorded <= cutoff_day", "missing_handling": "Imputed via training median"},
        {"feature": "max_assessment_score", "group": "Dynamic (Assessment)", "source": "studentAssessment", "calculation": "Maximum score across submissions", "temporal_rule": "Submission recorded <= cutoff_day", "missing_handling": "Imputed via training median"},
        {"feature": "min_assessment_score", "group": "Dynamic (Assessment)", "source": "studentAssessment", "calculation": "Minimum score across submissions", "temporal_rule": "Submission recorded <= cutoff_day", "missing_handling": "Imputed via training median"},
        {"feature": "assessment_score_sd", "group": "Dynamic (Assessment)", "source": "studentAssessment", "calculation": "Standard deviation of scores", "temporal_rule": "Submission recorded <= cutoff_day", "missing_handling": "Imputed as 0"},
        {"feature": "mean_assessment_weight", "group": "Dynamic (Assessment)", "source": "assessments & studentAssessment", "calculation": "Mean weight of assessments seen/submitted", "temporal_rule": "Assessment due date <= cutoff_day", "missing_handling": "Imputed as 0"},
        {"feature": "on_time_submission_rate", "group": "Dynamic (Assessment)", "source": "assessments & studentAssessment", "calculation": "Proportion where date_submitted <= assessment date", "temporal_rule": "Assessment due date <= cutoff_day", "missing_handling": "Imputed as 0"},
    ]
    save_csv(pd.DataFrame(predictor_meta), out/"tables"/"02_feature_dictionary.csv")

    specs = model_specs(seed, int(cfg["n_jobs"]))
    perf, fitted, preds = fit_models(
        X_train, y_train, X_test, y_test, specs, seed, int(cfg["n_jobs"])
    )
    save_csv(perf, out/"tables"/"03_model_performance.csv")

    # Static-only and dynamic+static ablation.
    Xs_train = X_train[static].copy()
    Xs_test = X_test[static].copy()
    perf_static, _, preds_static = fit_models(
        Xs_train, y_train, Xs_test, y_test, specs, seed, int(cfg["n_jobs"])
    )
    perf_static["feature_set"] = "static_only"
    perf_full = perf.copy()
    perf_full["feature_set"] = "static_plus_behavioural"
    ab = pd.concat([perf_static, perf_full], ignore_index=True)
    save_csv(ab, out/"tables"/"04_ablation_static_vs_behavioural.csv")

    # Student-level paired bootstrap CI for Static vs Full models (Reviewer 1, Comment 3)
    boot_rows = []
    for idx_m, name in enumerate(specs.keys()):
        res_b = paired_student_bootstrap_diff(
            test, y_test.to_numpy(), preds[name], preds_static[name],
            repeats=int(cfg["bootstrap_repeats"]), seed=seed + 100 + idx_m
        )
        boot_rows.append({"model": name, **res_b})
    save_csv(pd.DataFrame(boot_rows), out/"tables"/"12_static_vs_full_student_bootstrap.csv")

    # Score-free assessment sensitivity ablation (Reviewer 1, Point 2)
    score_free_cols = static + groups["score_free_dynamic"]
    Xsf_train = X_train[score_free_cols].copy()
    Xsf_test = X_test[score_free_cols].copy()
    perf_score_free, _, _ = fit_models(
        Xsf_train, y_train, Xsf_test, y_test, specs, seed, int(cfg["n_jobs"])
    )
    perf_score_free["feature_set"] = "static_plus_behavioural_score_free"
    perf_score_free_all = pd.concat([ab, perf_score_free], ignore_index=True)
    save_csv(perf_score_free_all, out/"tables"/"11_sensitivity_score_free.csv")

    # SMOTE ablation: preprocessing first, then SMOTE on training data only.
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from .models import make_preprocessor
    smote_rows = []
    for name, model in specs.items():
        pre = make_preprocessor(X_train)
        pipe = ImbPipeline([
            ("preprocess", pre),
            ("smote", SMOTE(k_neighbors=int(cfg["smote_k_neighbors"]), random_state=seed)),
            ("model", model),
        ])
        t = time.perf_counter()
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:,1]
        sm = metrics(y_test, p)
        smote_rows.append({"model":name, "condition":"SMOTE", "fit_time_s":time.perf_counter()-t, **sm})
    base = perf.copy()
    base["condition"] = "No_SMOTE"
    sm = pd.concat([base, pd.DataFrame(smote_rows)], ignore_index=True)
    save_csv(sm, out/"tables"/"05_smote_ablation.csv")

    # Tuning ablation: student-grouped cross-validation search for tree ensembles (Reviewer 1, Comment 3).
    tuned_rows = []
    from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
    param_spaces = {
        "Random Forest": {
            "model__n_estimators":[300,500,700],
            "model__max_depth":[None,8,12,18],
            "model__min_samples_leaf":[1,3,5,10],
        },
        "XGBoost": {
            "model__n_estimators":[250,400,600],
            "model__max_depth":[3,5,7],
            "model__learning_rate":[0.03,0.05,0.1],
            "model__subsample":[0.75,0.9,1.0],
        },
        "LightGBM": {
            "model__n_estimators":[250,400,600],
            "model__num_leaves":[15,31,63],
            "model__learning_rate":[0.03,0.05,0.1],
            "model__min_child_samples":[10,20,40],
        }
    }
    cv = StratifiedGroupKFold(n_splits=int(cfg["cv_splits"]), shuffle=True, random_state=seed)
    for name in ["Random Forest","XGBoost","LightGBM"]:
        from .models import build_pipeline
        pipe = build_pipeline(specs[name], X_train)
        search = RandomizedSearchCV(
            pipe, param_spaces[name], n_iter=8, scoring="roc_auc",
            cv=cv, random_state=seed, n_jobs=int(cfg["n_jobs"]), refit=True
        )
        tt=time.perf_counter()
        search.fit(X_train, y_train, groups=train["id_student"])
        p=search.predict_proba(X_test)[:,1]
        tuned_rows.append({"model":name, "condition":"Tuned", "fit_time_s":time.perf_counter()-tt,
                           **metrics(y_test,p), "best_params":str(search.best_params_)})
    tuning = pd.concat([
        perf[perf["model"].isin(["Random Forest","XGBoost","LightGBM"])].assign(condition="Default"),
        pd.DataFrame(tuned_rows)
    ], ignore_index=True)
    save_csv(tuning, out/"tables"/"06_tuning_ablation.csv")

    # Permutation importance for the strongest ROC-AUC model.
    best_name = perf.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    imp = permutation_table(fitted[best_name], X_test, y_test,
                            repeats=int(cfg["permutation_repeats"]), seed=seed)
    imp.insert(0, "model", best_name)
    save_csv(imp, out/"tables"/"07_permutation_importance.csv")

    group_imp = []
    for group, cols in groups.items():
        vals = imp[imp["feature"].isin(cols)]["importance_mean"]
        group_imp.append({
            "feature_group": group,
            "sum_positive_importance": float(vals[vals>0].sum()),
            "mean_importance": float(vals.mean()) if len(vals) else 0,
            "feature_count": len(cols)
        })
    gi = pd.DataFrame(group_imp)
    denom = gi["sum_positive_importance"].sum()
    gi["percent_of_positive_importance"] = np.where(denom>0, gi["sum_positive_importance"]/denom*100, 0)
    save_csv(gi, out/"tables"/"08_feature_group_importance.csv")

    # Pairwise bootstrap ROC-AUC comparisons, followed by Holm correction.
    names=list(preds)
    rows=[]
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            a,b=names[i],names[j]
            r=paired_bootstrap_diff(y_test.to_numpy(), preds[a], preds[b],
                                    repeats=int(cfg["bootstrap_repeats"]), seed=seed+i+j)
            rows.append({"model_a":a,"model_b":b,**r})
    save_csv(holm_adjustment(rows), out/"tables"/"09_pairwise_holm.csv")

    # Figures.
    figdir=out/"figures"
    figdir.mkdir(parents=True,exist_ok=True)
    from sklearn.metrics import roc_curve, precision_recall_curve
    plt.figure(figsize=(8,6))
    for name,p in preds.items():
        fpr,tpr,_=roc_curve(y_test,p)
        plt.plot(fpr,tpr,label=name)
    plt.plot([0,1],[0,1],"--",label="Chance")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves — 50% semester boundary")
    plt.legend(); plt.tight_layout(); plt.savefig(figdir/"01_roc_curves.png",dpi=300); plt.close()

    plt.figure(figsize=(8,6))
    for name,p in preds.items():
        precision,recall,_=precision_recall_curve(y_test,p)
        plt.plot(recall,precision,label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–recall curves — 50% semester boundary")
    plt.legend(); plt.tight_layout(); plt.savefig(figdir/"02_pr_curves.png",dpi=300); plt.close()

    top=imp.head(15).sort_values("importance_mean")
    plt.figure(figsize=(8,6))
    plt.barh(top["feature"],top["importance_mean"])
    plt.xlabel("Permutation importance (ROC-AUC decrease)")
    plt.title(f"Top permutation features — {best_name}")
    plt.tight_layout(); plt.savefig(figdir/"03_top_permutation_importance.png",dpi=300); plt.close()

    plt.figure(figsize=(7,5))
    sns.barplot(data=gi,x="feature_group",y="percent_of_positive_importance")
    plt.ylabel("Share of summed positive permutation importance (%)")
    plt.xlabel("Feature group")
    plt.title("Relative contribution of static and behavioural features")
    plt.tight_layout(); plt.savefig(figdir/"04_feature_group_importance.png",dpi=300); plt.close()

    metadata = {
        "boundary_fraction": cfg["boundary_fraction"],
        "seed": seed,
        "best_model_by_test_roc_auc": best_name,
        "runtime_seconds": time.perf_counter()-t0,
        "data_source": "Open University Learning Analytics Dataset (OULAD), UCI dataset 349",
        "notes": [
            "All performance values were regenerated in this run.",
            "No previous-paper metrics are embedded in the analysis code.",
            "The final paper should report only values present in results/tables/.",
            "The prediction boundary is 50% of presentation duration for each presentation."
        ]
    }
    with open(out/"tables"/"10_run_metadata.json","w",encoding="utf-8") as f:
        json.dump(metadata,f,indent=2)

    log.info("Pipeline completed in %.1f seconds.", time.perf_counter()-t0)
    log.info("Best test ROC-AUC model: %s", best_name)

if __name__ == "__main__":
    main()