$ErrorActionPreference = "Stop"
python -m pip install -e .
python -m pip install -r requirements.txt
python scripts/make_v3_figures.py
