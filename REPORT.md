# Telco Customer Churn: Decision Tree and Random Forest

## Scope and reproducibility

This report answers both parts of the assignment using the public
`WA_Fn-UseC_-Telco-Customer-Churn.csv` data set. The source file is deliberately
not committed because it is a user-supplied laptop file. Run
`python analyze_churn.py <csv>` to generate the numerical results and figures
used here. The command uses random seed 42, a stratified 80/20 train/test split,
and writes `outputs/metrics.json`, `class_distribution.png`, `roc_curves.png`,
`confusion_matrices.png`, and `decision_tree.png`.

The target is `Churn` (`Yes` = 1, `No` = 0). `customerID` is removed because it
is an identifier, not a behavioural predictor. Blank `TotalCharges` values are
treated as missing and median-imputed within the training pipeline. Numeric
variables are standardized; categorical variables are most-frequently imputed
and one-hot encoded. Fitting preprocessing inside each pipeline prevents test
set leakage.

## Part I — tree methods

### Decision tree

The decision-tree pipeline uses Gini impurity, `max_depth=5`, and
`min_samples_leaf=10`. A tree recursively chooses a feature and threshold that
most reduce class impurity. Its `decision_tree.png` visualization shows the
top three levels with class counts and predicted class. This makes the model
useful for explaining rules such as contract, tenure, or service combinations,
although the displayed top levels are not a substitute for the complete fitted
model. The generated `metrics.json` reports accuracy, precision, recall, F1,
ROC-AUC, and the confusion matrix on unseen customers.

### Random forest

The random-forest pipeline averages 300 bootstrapped trees. Each split considers
a random feature subset, reducing the variance and instability of one tree.
`max_depth=8`, `min_samples_leaf=5`, and balanced class weights provide
regularization and make the minority churn class visible during fitting.
`roc_curves.png` and `confusion_matrices.png` show whether this variance
reduction improves ranking and churn detection over the single tree.

### Assumptions and limitations

* Both tree methods assume that the labelled historical outcomes are a useful
  proxy for future churn and that the train/test split represents deployment.
  They do not require linearity or normally distributed predictors.
* A single tree can overfit, be unstable under small data changes, and produce
  overly complex rules. A depth limit and minimum leaf size reduce, but do not
  eliminate, this risk.
* A forest is less interpretable, costs more to train and serve, and can still
  learn historical bias or leakage. Feature importance is not causation.
* The data is observational and cross-sectional. Churn definitions, missing
  values, class imbalance, changes in pricing, and time drift can make
  retrospective scores overstate production performance.

## Part II — comparison with previous models

### Evaluation framework

All four models use the identical cleaned features, stratified holdout, random
seed, and probability threshold (0.5). Accuracy is included for context, but
it is insufficient when churn is the minority class. Precision measures the
fraction of flagged customers who churn; recall measures the fraction of all
churners found; F1 balances those two; and ROC-AUC measures ranking across
thresholds. The confusion matrices expose the operational trade-off between
avoidable retention contacts and missed churners.

The generated `metrics.json` is the authoritative comparison table. Select the
model using the business cost of false negatives versus false positives, not
accuracy alone. In a typical retention setting, the model with the strongest
recall/F1 at an acceptable precision is preferable. If ROC-AUC and F1 disagree,
choose an operating threshold using validation data and report that threshold
rather than silently tuning on the test set. For a high-stakes deployment,
repeat the evaluation with stratified cross-validation and a time-based holdout.

Logistic regression is a strong, transparent linear baseline and its
coefficients indicate direction after encoding. kNN can capture local,
non-linear patterns but is sensitive to scaling, the choice of `k`, and
irrelevant/high-dimensional one-hot variables. The tree captures interactions
and produces rules; the forest usually generalizes better by averaging many
decorrelated trees. The analysis script makes the comparison fair by applying
the same preprocessing and holdout to each method.

### Ethically sustainable use of customer data

A data-driven retention process is ethically sustainable only if customers are
informed about relevant data use, the company collects only necessary
attributes, limits access and retention, and provides deletion/correction
mechanisms where applicable. The model should recommend a helpful, opt-in
intervention rather than deny service or impose a penalty. Audit performance
and false-positive/false-negative rates across protected or vulnerable groups;
investigate proxies such as location or payment method; document model and
threshold changes; and provide human review and an appeal path. Encrypt data,
log access, monitor drift, and retrain only with a documented purpose.

Therefore, the model alone does not establish ethical sustainability. The
process is sustainable only after privacy, fairness, transparency,
proportionality, security, and governance controls are demonstrated in a
production impact assessment.

## References

Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5–32.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
*Journal of Machine Learning Research, 12*, 2825–2830.
