# Online Shoppers Purchasing Intention

This project predicts whether an e-commerce browsing session ends in a purchase (`Revenue = True/False`) using the UCI Online Shoppers Purchasing Intention dataset.

## a. Problem statement

Given session-level browsing behaviour (pages visited, time spent, bounce/exit rates, visitor type, etc.), the goal is to classify each session as purchase or no-purchase. Five models are trained on the same data, compared with Accuracy, AUC, Precision, Recall, F1 and MCC, and exposed through a Streamlit app that scores held-out test rows.

## b. Dataset description

| Item | Detail |
|------|--------|
| Name | UCI Online Shoppers Purchasing Intention |
| Source | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Rows | 12,330 |
| Features | 17 predictors + target `Revenue` |
| Task | Binary classification |

Feature groups:

- Activity: `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`
- Engagement: `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`
- Context: `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`

Numeric columns are scaled; `Month`, `VisitorType` and `Weekend` are one-hot encoded inside each model pipeline.

## c. GitHub Repository Link

https://github.com/ganeshkumar-chandrasekaran/online-shoppers-classifier

## Live Streamlit app

https://ganeshkumar-chandrasekaran-online-shoppers-classifier.streamlit.app

### What goes on GitHub

The **GitHub repo root** should contain these files directly
(do **not** nest another `online-shopping/` folder inside the repo):

```text
online-shoppers-classifier/          ← GitHub repository
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── data/
│   └── online_shoppers_intention.csv
└── model/
    ├── online_shoppers_classification.py
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest_ensemble.joblib
    ├── label_encoder.joblib
    └── metrics.json
```

On this laptop the same files currently live under a local folder named `online-shopping/`
only for convenience. When creating the GitHub repo, push the **contents** of that folder
as the repository root so Streamlit Cloud can find `app.py` at the top level.

## d. Models used

Training used an 80/20 stratified split (`random_state=42`). Metrics below are on the **held-out test set** (2,466 rows), also saved as `test_data.csv`. Models use class weighting / distance weights where helpful, and decision thresholds are tuned on a validation slice of the training data to improve F1/recall without leaking the test set.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8779 | 0.8962 | 0.5990 | 0.6414 | 0.6195 | 0.5473 |
| Decision Tree | 0.8658 | 0.8863 | 0.5484 | 0.7565 | 0.6359 | 0.5670 |
| kNN | 0.8625 | 0.8118 | 0.5616 | 0.5131 | 0.5363 | 0.4564 |
| Naive Bayes | 0.7113 | 0.7932 | 0.3171 | 0.7487 | 0.4455 | 0.3404 |
| Random Forest (Ensemble) | 0.8889 | 0.9255 | 0.6274 | 0.6963 | 0.6600 | 0.5950 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Balanced class weights + tuned threshold raise purchase recall/F1 a lot vs a plain 0.5 cutoff. |
| Decision Tree | Strongest recall among tree/linear models; accuracy dips a little versus the untuned tree. |
| kNN | Distance-weighted neighbors improve F1/recall over uniform kNN, but still trail ensembles. |
| Naive Bayes | High recall continues; precision/accuracy remain the weak spot even after threshold tuning. |
| Random Forest (Ensemble) | Best AUC, F1 and MCC; much higher purchase recall than the earlier RF, with only a small accuracy trade-off. |
| Overall Winner | Random Forest (Ensemble) |

## How to run

### Train

From the project root:

```bash
pip install -r requirements.txt
python model/online_shoppers_classification.py
```

This trains the five models and writes `model/*.joblib`, `model/metrics.json`, and `test_data.csv`.

### Streamlit app

From the **project root** (folder that contains `app.py`):

```bash
pip install -r requirements.txt
streamlit run app.py
```

App behaviour:

- Default evaluation file = `test_data.csv` (**held-out TEST split**, not training data)
- Optional sidebar upload = score a custom evaluation CSV
- Live metrics + confusion matrix = current evaluation file only
- Saved comparison table = original test-split scores from the training run

### Streamlit Community Cloud

Live app: https://ganeshkumar-chandrasekaran-online-shoppers-classifier.streamlit.app

Deployed from this repository (`main` / `app.py`) on https://streamlit.io/cloud.

## Dependencies

See `requirements.txt`: streamlit, scikit-learn, numpy, pandas, matplotlib, seaborn, joblib.
