# Home Credit Default Risk: Credit Scoring Pipeline

An end-to-end credit risk project that predicts whether a loan applicant will have difficulty repaying, using the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. The workflow follows how a lender builds a scorecard: data cleaning, feature engineering, Weight of Evidence / Information Value (WoE/IV) analysis, model training, and a cost-based decision threshold.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Methodology](#methodology)
- [Evaluation](#evaluation)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Problem Statement
Given an applicant's profile and prior credit history, predict `TARGET`:

| Value | Meaning |
|---|---|
| `0` | Customer repaid on time |
| `1` | Customer had difficulty repaying (default) |

The data is heavily imbalanced (about 8% defaults), so accuracy is not a useful metric. The project uses ranking and risk metrics and ends with a business decision rule.

## Dataset
Source: [Kaggle, Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)

| File | Description | Used |
|---|---|---|
| `application_train.csv` | One row per applicant, includes `TARGET` | Yes |
| `bureau.csv` | Applicant's previous credits at other institutions | Yes |
| `HomeCredit_columns_description.csv` | Column documentation | Yes |
| `application_test.csv` | Test set without labels (Kaggle submission) | No |
| `bureau_balance.csv`, `previous_application.csv`, `installments_payments.csv`, `credit_card_balance.csv`, `POS_CASH_balance.csv` | Additional history tables | No (see [Future Work](#future-work)) |

> The data is not included in this repository. Download it from Kaggle (see [Setup](#setup)).

## Project Structure
```
.
├── home_credit_risk_complete.ipynb   # Full pipeline, cell by cell
├── requirements.txt                  # Python dependencies
├── README.md
└── models/                           # Created automatically when the notebook is run
    ├── lgbm_credit_risk.joblib
    ├── logreg_credit_risk.joblib
    └── metadata.joblib
```
Everything lives in the notebook. Helper functions for unzipping and memory reduction are defined inside it, so no extra modules are needed. The `data/` and `models/` folders are generated at run time and should not be committed.

## Setup

**1. Create and activate a virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Get the data from Kaggle**
1. Log in to Kaggle and open the [competition page](https://www.kaggle.com/c/home-credit-default-risk/data).
2. Click **Join Competition** and accept the rules (required, otherwise downloads fail with a 403 error).
3. Create an API token under **Settings, API**, then authenticate in one of these ways:
   - Run `kagglehub.login()` and enter your username and token, or
   - Set the environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`, or
   - Download the zip manually, extract it, and set `DATA_DIR` in the notebook to that folder.

> Never commit your Kaggle token to version control.

## How to Run
1. Open `home_credit_risk_complete.ipynb` in Jupyter or VS Code and select the `.venv` kernel.
2. Run the cells from top to bottom (each cell depends on the ones before it).
3. Optional settings at the top of the notebook:
   - `SHOW_EDA = False` skips the EDA plots.
   - `COST_FN` / `COST_FP` set the business cost of a missed defaulter vs a rejected good customer.

## Methodology

### 1. Data preparation
- Memory optimisation by downcasting numeric types.
- Columns with more than 50% missing values are dropped.
- Columns with 10-50% missing values get a NaN indicator flag, then imputation.
- Remaining nulls: median (numeric) and mode (categorical).
- `DAYS_*` columns converted to years; the `365243` placeholder in `DAYS_EMPLOYED` is flagged and replaced.
- Rows with `CODE_GENDER = XNA` removed.

### 2. Feature engineering
- Ratios: `INCOME_BY_CREDIT`, `ANNUITY_BY_INCOME`, `GOODS_AMT_BY_CREDIT`.
- Social-circle default percentages: `DEF_30_RATIO`, `DEF_60_RATIO`.
- Age bands (`AGE_BIN`).
- Bureau aggregates per applicant: total/mean credit and debt, mean debt-to-credit ratio, overdue and active counts, credit recency, and a `HAS_BUREAU_RECORD` flag.
- Redundant features removed (document flags, duplicate building statistics, sparse inquiry counts).

### 3. Feature analysis (WoE / IV)
Each feature is binned and scored with Information Value, the standard credit-scoring way to rank predictors.

| IV | Predictive power |
|---|---|
| < 0.02 | Useless |
| 0.02 - 0.1 | Weak |
| 0.1 - 0.3 | Medium |
| 0.3 - 0.5 | Strong |
| > 0.5 | Suspicious, check for leakage |

### 4. Modelling
| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline (quantile scaling, one-hot encoding, balanced class weights) |
| LightGBM | Main model (early stopping, `scale_pos_weight` for imbalance) |

An 80/20 stratified train/test split is used. A validation slice of the training data drives early stopping and threshold selection, so the test set is only used for final reporting.

### 5. Decision threshold
The cut-off is chosen to minimise expected cost, where a missed defaulter costs `COST_FN` and a rejected good customer costs `COST_FP` (default 5:1, which should be replaced with real figures).

## Evaluation
- **ROC-AUC**: overall ranking ability
- **PR-AUC**: performance on the minority class
- **KS statistic**: separation between good and bad customers (credit-industry standard)
- **5-fold cross-validation**: stability of the AUC estimate
- **Risk deciles**: default rate and lift from the lowest to highest risk band
- **Confusion matrix** at the cost-optimal threshold: recall, precision, approval rate

## Results
Run the notebook to populate this table with your own numbers.

| Model | ROC-AUC | PR-AUC | KS |
|---|---|---|---|
| Logistic Regression | _fill in_ | _fill in_ | _fill in_ |
| LightGBM | _fill in_ | _fill in_ | _fill in_ |

## Limitations
- Only 2 of the 7 source tables are used, which limits achievable AUC.
- Missing-value imputation is performed before the train/test split, a minor leakage risk. Moving it into the sklearn pipeline would remove it.
- No hyperparameter tuning has been done.
- No fairness analysis has been done (for example, on `CODE_GENDER`), which matters for lending models.
- Cost ratios for the threshold are assumptions.

## Future Work
- Aggregate features from `previous_application`, `installments_payments`, `credit_card_balance`, `POS_CASH_balance`, and `bureau_balance`.
- Hyperparameter search with Optuna.
- SHAP explanations for individual decisions.
- WoE-based logistic regression scorecard with score scaling.
- Probability calibration and fairness / bias review.
- Model deployment (for example, a FastAPI scoring endpoint).

## Acknowledgements
Data provided by Home Credit via Kaggle. Use of the dataset is subject to the competition rules.