import sys
packages = [
    "numpy", "pandas", "scipy", "sklearn", "imblearn",
    "xgboost", "lightgbm", "matplotlib", "yaml", "statsmodels"
]
failed=[]
for p in packages:
    try:
        __import__(p)
        print(f"[OK] {p}")
    except Exception as e:
        print(f"[FAIL] {p}: {e}")
        failed.append(p)
if failed:
    sys.exit(1)
print("Environment validation passed.")
