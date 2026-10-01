# CustomerChurn-Decision-Tree-and-Random-Forest

This repository contains a reproducible Telco customer-churn analysis. The
assignment report is in [`REPORT.md`](REPORT.md), including the answers to
Part I and Part II, assumptions, limitations, model comparison, evaluation
framework, and ethical-sustainability assessment.

## Reproduce the analysis

Install the pinned dependencies, then run the analysis against the supplied
`WA_Fn-UseC_-Telco-Customer-Churn.csv` file:

```bash
python -m pip install -r requirements.txt
python analyze_churn.py /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The script performs leakage-safe preprocessing and trains logistic regression,
k-nearest neighbours, a decision tree, and a random forest on the same
stratified split. It writes `metrics.json` and the class-distribution, ROC,
confusion-matrix, and decision-tree figures to `outputs/`.

Run focused tests with:

```bash
python -m pytest
```
