import numpy as np
import pandas as pd

RISK_RESULTS = {"Fail", "Withdrawn"}
SAFE_RESULTS = {"Pass", "Distinction"}

def _safe_div(a, b):
    return np.where(np.asarray(b) == 0, 0.0, np.asarray(a) / np.asarray(b))

def build_dataset(d, boundary_fraction=0.50):
    info = d["studentInfo"].copy()
    courses = d["courses"].copy()
    assess = d["assessments"].copy()
    sa = d["studentAssessment"].copy()
    sv = d["studentVle"].copy()
    vle = d["vle"].copy()
    reg = d["studentRegistration"].copy()

    # Keep only labelled final outcomes.
    info = info[info["final_result"].isin(RISK_RESULTS | SAFE_RESULTS)].copy()
    info["risk"] = info["final_result"].isin(RISK_RESULTS).astype(int)

    # Presentation-specific prediction boundary.
    pres = courses[["code_module", "code_presentation", "module_presentation_length"]].copy()
    pres["cutoff_day"] = np.ceil(pres["module_presentation_length"] * boundary_fraction).astype(int)

    base = info.merge(
        pres,
        on=["code_module", "code_presentation"],
        how="inner",
        validate="many_to_one"
    )

    # Merge registration data to obtain date_unregistration for cohort filtering (Reviewer 1, Point 1)
    base = base.merge(
        reg[["code_module", "code_presentation", "id_student", "date_unregistration"]],
        on=["code_module", "code_presentation", "id_student"],
        how="left",
        validate="one_to_one"
    )

    # Audit withdrawal timing relative to presentation cutoff_day
    total_raw = len(base)
    pre_cutoff_mask = (base["date_unregistration"].notna()) & (base["date_unregistration"] <= base["cutoff_day"])
    pre_count = pre_cutoff_mask.sum()
    
    print(f"[COHORT AUDIT] Raw labelled student-presentation records: {total_raw}")
    print(f"[COHORT AUDIT] Pre-cutoff or same-day withdrawals excluded (date_unregistration <= cutoff_day): {pre_count}")

    # Enforce strict early-warning eligibility: exclude pre-cutoff and same-day withdrawals
    base = base[~pre_cutoff_mask].copy()
    print(f"[COHORT AUDIT] Final active analytical cohort records: {len(base)}")

    # Static features: exclude final outcome and registration fields affected by withdrawal
    static_cols = [
        "code_module", "code_presentation", "id_student",
        "gender", "region", "highest_education", "imd_band",
        "age_band", "num_of_prev_attempts", "studied_credits", "disability"
    ]
    static = base[static_cols + ["risk", "final_result", "module_presentation_length", "cutoff_day", "date_unregistration"]].copy()

    # VLE records available by the 50% boundary only.
    sv = sv.merge(
        pres,
        on=["code_module", "code_presentation"],
        how="inner",
        validate="many_to_one"
    )
    sv = sv[sv["date"].notna() & (sv["date"] >= 0) & (sv["date"] <= sv["cutoff_day"])].copy()

    # Add activity type. Missing activity metadata is retained as 'unknown'.
    sv = sv.merge(
        vle[["id_site", "code_module", "code_presentation", "activity_type"]],
        on=["id_site", "code_module", "code_presentation"],
        how="left"
    )
    sv["activity_type"] = sv["activity_type"].fillna("unknown")
    sv["sum_click"] = pd.to_numeric(sv["sum_click"], errors="coerce").fillna(0)

    gkeys = ["code_module", "code_presentation", "id_student"]

    if len(sv):
        vg = sv.groupby(gkeys, observed=True)
        beh = vg.agg(
            total_clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
            unique_sites=("id_site", "nunique"),
            active_weeks=("date", lambda x: int(np.ceil((x.max() + 1) / 7)) if len(x) else 0),
            mean_daily_clicks=("sum_click", "mean"),
            max_daily_clicks=("sum_click", "max"),
            unique_activity_types=("activity_type", "nunique"),
        ).reset_index()
        beh["clicks_per_active_day"] = _safe_div(beh["total_clicks"], beh["active_days"])
        beh["active_day_ratio"] = _safe_div(
            beh["active_days"], np.maximum(beh["cutoff_day"] if "cutoff_day" in beh else 1, 1)
        )
        # A second daily aggregation gives a more meaningful maximum daily volume.
        daily = sv.groupby(gkeys + ["date"], observed=True)["sum_click"].sum().reset_index()
        daily_max = daily.groupby(gkeys, observed=True)["sum_click"].max().rename("max_daily_total_clicks")
        beh = beh.merge(daily_max.reset_index(), on=gkeys, how="left")
    else:
        beh = pd.DataFrame(columns=gkeys)

    # Assessment information available by the 50% boundary.
    assess2 = assess.merge(
        pres,
        on=["code_module", "code_presentation"],
        how="inner",
        validate="many_to_one"
    )
    assess2 = assess2[assess2["date"].notna() & (assess2["date"] >= 0) & (assess2["date"] <= assess2["cutoff_day"])].copy()

    sa2 = sa.merge(
        assess2[["id_assessment", "code_module", "code_presentation", "assessment_type", "weight", "date"]],
        on="id_assessment",
        how="inner",
        validate="many_to_one"
    )
    sa2["score"] = pd.to_numeric(sa2["score"], errors="coerce")
    sa2["weight"] = pd.to_numeric(sa2["weight"], errors="coerce")
    sa2["submitted_on_time"] = (sa2["date_submitted"] <= sa2["date"]).astype(int)
    sa2["score_norm"] = sa2["score"] / 100.0

    if len(sa2):
        ag = sa2.groupby(["code_module", "code_presentation", "id_student"], observed=True)
        assfeat = ag.agg(
            assessments_seen=("id_assessment", "nunique"),
            assessment_submissions=("id_assessment", "count"),
            mean_assessment_score=("score", "mean"),
            max_assessment_score=("score", "max"),
            min_assessment_score=("score", "min"),
            assessment_score_sd=("score", "std"),
            mean_assessment_weight=("weight", "mean"),
            on_time_submission_rate=("submitted_on_time", "mean"),
        ).reset_index()
    else:
        assfeat = pd.DataFrame(columns=gkeys)

    out = static.merge(beh, on=gkeys, how="left").merge(assfeat, on=gkeys, how="left")

    dynamic_cols = [
        "total_clicks", "active_days", "unique_sites", "active_weeks",
        "mean_daily_clicks", "max_daily_clicks", "unique_activity_types",
        "clicks_per_active_day", "active_day_ratio", "max_daily_total_clicks",
        "assessments_seen", "assessment_submissions", "mean_assessment_score",
        "max_assessment_score", "min_assessment_score", "assessment_score_sd",
        "mean_assessment_weight", "on_time_submission_rate"
    ]
    for c in dynamic_cols:
        if c in out:
            out[c] = out[c].fillna(0)

    # Recompute active_day_ratio using presentation-specific cutoff.
    out["active_day_ratio"] = _safe_div(out["active_days"], np.maximum(out["cutoff_day"], 1))

    # Remove identifiers and outcome text from modelling predictors.
    out = out.drop(columns=["final_result"], errors="ignore")
    return out

def feature_groups(df):
    static = [
        "gender", "region", "highest_education", "imd_band", "age_band",
        "num_of_prev_attempts", "studied_credits", "disability"
    ]
    dynamic = [
        "total_clicks", "active_days", "unique_sites", "active_weeks",
        "mean_daily_clicks", "max_daily_clicks", "unique_activity_types",
        "clicks_per_active_day", "active_day_ratio", "max_daily_total_clicks",
        "assessments_seen", "assessment_submissions", "mean_assessment_score",
        "max_assessment_score", "min_assessment_score", "assessment_score_sd",
        "mean_assessment_weight", "on_time_submission_rate"
    ]
    score_features = {"mean_assessment_score", "max_assessment_score", "min_assessment_score", "assessment_score_sd"}
    score_free_dynamic = [c for c in dynamic if c not in score_features]

    return {
        "static": [c for c in static if c in df.columns],
        "dynamic": [c for c in dynamic if c in df.columns],
        "score_free_dynamic": [c for c in score_free_dynamic if c in df.columns],
    }