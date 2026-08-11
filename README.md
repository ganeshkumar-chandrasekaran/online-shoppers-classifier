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

Training used an 80/20 stratified split (`random_state=42`). Metrics below are on the **held-out test set** (2,466 rows), also saved as `test_data.csv`.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8812 | 0.8876 | 0.7432 | 0.3560 | 0.4814 | 0.4603 |
| Decision Tree | 0.8824 | 0.8480 | 0.6456 | 0.5340 | 0.5845 | 0.5199 |
| kNN | 0.8747 | 0.7998 | 0.6714 | 0.3743 | 0.4807 | 0.4389 |
| Naive Bayes | 0.6736 | 0.7932 | 0.2937 | 0.7880 | 0.4279 | 0.3234 |
| Random Forest (Ensemble) | 0.9015 | 0.9209 | 0.7565 | 0.5366 | 0.6279 | 0.5842 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong accuracy and AUC, but recall is low — many real purchases are missed. |
| Decision Tree | Better recall than logistic regression; accuracy stays close, AUC is a bit lower. |
| kNN | Decent accuracy, but AUC and F1 lag behind the tree-based models. |
| Naive Bayes | Highest recall, weakest precision/accuracy — too many false purchase flags. |
| Random Forest (Ensemble) | Best Accuracy, AUC, Precision, F1 and MCC on this dataset. |
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
