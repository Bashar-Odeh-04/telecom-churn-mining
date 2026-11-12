# Telecom Customer Churn Prediction

Predicting which telecom customers are likely to churn so a retention team can prioritize limited retention offers.

**Course:** Applied Data Mining, Phase 1 (Project Proposal, Data Sourcing Plan & Stack Setup)
**Team:** Kinan Sanajleh 2110269 Bashar Odeh 2210221

## Business Problem

A telecom retention team has a limited budget for discounts, upgrades and calls, so it cannot offer something to every customer. This project builds a model that flags customers likely to leave, using account, service and billing information, and identifies which factors drive churn.

## Research Question

Can a classification model predict churn with a ROC-AUC of at least 0.80 and recall of at least 0.70 on the churn class, and does it clearly beat a logistic regression baseline? Which features contribute most?

## Dataset

- **Source:** IBM Telco Customer Churn dataset (Kaggle, "Telco Customer Churn" by blastchar)
- **Size:** about 7,043 rows x 21 columns (confirm with `df.shape`)
- **Target:** `Churn` (Yes/No), roughly 26% positive
- **File location:** `data/telco_churn.csv` (rename the downloaded `WA_Fn-UseC_-Telco-Customer-Churn.csv`)

| Group | Columns |
| --- | --- |
| Identifier | customerID (dropped before modelling) |
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Billing | MonthlyCharges, TotalCharges |
| Target | Churn |

**Known data issues:** `TotalCharges` loads as text with blank values for brand-new customers; classes are imbalanced; the data is a single snapshot with no time dimension.

## Approach

1. Clean data (convert `TotalCharges` to numeric, handle missing rows)
2. Encode categoricals, scale numerics for linear models
3. Engineer features: tenure groups, number of add-on services, average monthly spend, month-to-month contract flag
4. Handle imbalance by comparing class weights and SMOTE
5. Compare Logistic Regression (baseline), Decision Tree, Random Forest and Gradient Boosting/XGBoost using stratified splits and cross-validation
6. Evaluate with ROC-AUC, precision, recall, F1 and a confusion matrix
7. Interpret feature importance
8. Rebuild the pipeline in PySpark MLlib and compare with scikit-learn

## Repository Structure

```
telecom-churn-mining/
  README.md
  requirements.txt
  data/            raw CSV
  notebooks/
    01_baseline.ipynb
  src/
  reports/
```

## Setup

**Local**

```bash
git clone https://github.com/<your-username>/telecom-churn-mining.git
cd telecom-churn-mining
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

**Google Colab**

1. Open `notebooks/01_baseline.ipynb` in Colab (File > Open notebook > GitHub).
2. Upload `telco_churn.csv` to a `data/` folder in the Colab file browser.
3. Run all cells. PySpark may need `!pip install pyspark` in the first cell.

Python 3.11 is recommended.

## Baseline

`notebooks/01_baseline.ipynb` loads the CSV, cleans `TotalCharges`, one-hot encodes features, makes a stratified 80/20 split and fits a class-weighted logistic regression, printing ROC-AUC and a classification report.

## Timeline

1. Exploratory analysis and cleaning
2. Model training and tuning
3. Evaluation and interpretation
4. Final report and presentation

## References

1. G. Sam, P. Asuquo and B. Stephen, "Customer Churn Prediction using Machine Learning Models," *Journal of Engineering Research and Reports*, vol. 26, no. 2, pp. 181-193, 2024.
2. L. N. Wakhidah, A. K. Zyen and B. B. Wahono, "Evaluation of Telecommunication Customer Churn Classification with SMOTE Using Random Forest and XGBoost Algorithms," *Journal of Applied Informatics and Computing*.
3. M. Imani, Z. Ghaderpour, M. Joudaki and A. Beikmohammadi, "The Impact of SMOTE and ADASYN on Random Forest and Advanced Gradient Boosting Techniques in Telecom Customer Churn Prediction," Preprints.org, 2024.
