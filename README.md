# Knowledge-Based AI for Digital Transformation Readiness

Research prototype accompanying the proposed short communication: **A Knowledge-Based AI Decision-Support Model for Predicting Digital Transformation Readiness from Enterprise Innovation Data**.

## What it does
A leakage-controlled train/test split; calibrated logistic-regression baseline; explicit knowledge rules about data quality, skills and leadership; interpretable recommended actions; comparable metrics (AUROC, average precision, Brier score, balanced accuracy); reproducible CSV/JSON output.

## Run
```bash
python -m pip install -r requirements.txt
python -m unittest -v test_model.py
python model.py --output outputs
python model.py --data your_enterprise_data.csv --output outputs
```

## Data contract
CSV columns: `innovation_spend_ratio`, `digital_skills`, `data_quality`, `leadership_support`, `process_digitization`, `innovation_collaboration`, and independently measured binary `ready`. All six predictors must be numeric and scaled to [0,1], with documented definitions. The `ready` label MUST be obtained independently of the six feature scores and knowledge rules (e.g., independently assessed deployment milestone), otherwise evaluation is circular. All columns must be present, nonmissing; at least 100 records and 10 per class.

## Scientific limitations
**Synthetic mode is only a software demonstration, not evidence of real enterprise readiness.** Knowledge thresholds and penalty are illustrative and must be elicited/validated before empirical claims. The adjusted score is a heuristic and may worsen calibration. No claimed superiority without independently sourced enterprise observations, preregistered hypotheses, independent holdout evaluation, confidence intervals, and appropriate baselines. Beware temporal leakage and survey sampling bias. Readiness is distinct from innovation activity and cannot be assumed from innovation expenditure alone.

## Compact repository
Only `model.py`, `test_model.py`, `requirements.txt`, `README.md`. Generated `outputs/` should not be committed. Add a `.gitignore` if necessary when publishing. License and data-use permissions must be determined before release.