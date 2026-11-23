# Paper 2 — 50% Boundary Reproducible Analysis

## Working title
**Early Student Academic-Risk Prediction at the Halfway Mark: Evaluating Behavioural Feature Engineering and Machine Learning Models**

This package is a fresh, reproducible implementation for the second OULAD study. It moves the prediction boundary from the first-quarter (25%) boundary to **50% of the course-presentation duration** and regenerates all performance results from the data.

The package is deliberately designed so that **no model-performance numbers from the earlier paper are hard-coded or copied**.

## Research focus
The experiment evaluates whether behaviourally engineered learning features available by the halfway point provide useful information for academic-risk prediction, and how much additional performance is associated with:
1. behavioural/assessment feature engineering,
2. class balancing with SMOTE, and
3. model tuning.

Five models are evaluated:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

## Outcome
`risk = 1` for final result `Fail` or `Withdrawn`; `risk = 0` for `Pass` or `Distinction`.

The final outcome is used only as the label. Post-boundary behavioural and assessment information is excluded from predictors.

## Important reproducibility rule
The script computes the 50% boundary separately for every course presentation:

`cutoff_day = ceil(0.50 × presentation_length)`

Only VLE and assessment records with dates on or before the cutoff are used.

## Folder structure
```text
oulad_paper2_50pct_reproducible/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/          # put OULAD CSVs here, or let the downloader fetch them
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   └── main.py
├── results/
│   ├── tables/
│   ├── figures/
│   └── logs/
└── tests/
    └── test_features.py
```

## Quick start

### Windows
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
```

### Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

If the raw OULAD files are absent, the pipeline attempts to download the public OULAD archive from UCI. The UCI repository identifies OULAD as a dataset covering seven selected courses and more than 30,000 students, with connected CSV tables. See the official dataset page: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics

## Expected raw files
The pipeline expects:
- `courses.csv`
- `assessments.csv`
- `studentAssessment.csv`
- `studentInfo.csv`
- `studentRegistration.csv`
- `vle.csv`
- `studentVle.csv`

The UCI distribution is the preferred source. If automatic download is unavailable, download OULAD manually and place the CSV files in `data/raw/`.

## Main outputs

After a successful run:

### Tables
- `01_cohort_summary.csv`
- `02_feature_dictionary.csv`
- `03_model_performance.csv`
- `04_ablation_static_vs_behavioural.csv`
- `05_smote_ablation.csv`
- `06_tuning_ablation.csv`
- `07_permutation_importance.csv`
- `08_feature_group_importance.csv`
- `09_pairwise_holm.csv`
- `10_run_metadata.json`

### Figures
- model ROC curves
- model precision-recall curves
- static-vs-behavioural comparison
- feature-group importance
- top permutation features

### Logs
- complete pipeline log
- model timing and run metadata

## Reproducibility safeguards
- fixed random seed;
- presentation-specific 50% cut-off;
- no post-cutoff VLE/assessment features;
- preprocessing fitted on training data only;
- SMOTE applied to training data only;
- grouped train/test split by `id_student` to reduce student-level leakage;
- model comparison based on held-out test predictions;
- pairwise bootstrap comparison with Holm correction;
- all tables/figures generated from the current run.

## Important interpretation rule
This package does **not** guarantee that the new paper will have better or worse performance than the earlier 25% study. The numerical results must be regenerated and reported exactly as produced by this run.

The final manuscript should use the actual output cohort counts, feature counts, model metrics, ablation results and statistical comparisons from `results/tables/`.

## Citation
Dataset:
Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2015). Open University Learning Analytics dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5KK69
