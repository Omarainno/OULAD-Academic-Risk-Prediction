#!/usr/bin/env bash
set -e
echo "=========================================="
echo "OULAD Paper 2 - 50% Boundary Pipeline"
echo "=========================================="
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
echo "Pipeline finished. Check results/tables and results/figures."
