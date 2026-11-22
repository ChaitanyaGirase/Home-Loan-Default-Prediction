# Home Loan Default Prediction

A machine learning project that analyzes home loan application data and predicts which customers may face payment difficulties. The project covers data cleaning, exploratory data analysis, preprocessing without data leakage, model building, model comparison, and customer risk segmentation.

> **Note:** The model predicts historical payment difficulties. It does not directly decide whether a customer is eligible for a loan.

---

## Project Objective

- Prepare a complete data analysis of the loan application data.
- Build and compare models to identify customers with a higher risk of payment difficulties.
- Identify the key factors and customer risk segments.
- Recommend the best model for production.

**Target column:** `TARGET`
- `0` = Customer did not face payment difficulties
- `1` = Customer faced payment difficulties

---

## Dataset

- **Source:** Home Credit loan application data (Banking domain)
- **Main file used:** `application_train.csv` (307,511 rows, 122 columns)
- **Class balance:** only 8.07% of customers have payment difficulties (highly imbalanced)

The other six files (`bureau.csv`, `bureau_balance.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, `previous_application.csv`, `installments_payments.csv`) are part of the original dataset but were **not used** in this project.

The dataset is not included in this repository because of its size. Download it from the link in the project requirement and place the files in the `dataset/` folder.

---

## Project Structure

```text
Home_Loan_Default_Prediction/
│
├── dataset
│
├── Home_Loan_Default_Prediction.ipynb
└── README.md
```

---

## Notebook Structure

1. Project Introduction
2. Business Understanding
3. Dataset Description
4. Importing Libraries
5. Data Loading
6. Data Understanding
7. Data Cleaning
8. Exploratory Data Analysis (EDA)
9. Data Preprocessing
10. Model Building
11. Model Evaluation
12. Model Comparison
13. Customer Risk Segmentation (including key factors)
14. Challenges Faced
15. Conclusion

---

## Workflow

**Data cleaning**
- Dropped 17 columns with more than 60% missing values.
- Replaced the unusual `365243` value in `DAYS_EMPLOYED` and created `AGE` and `YEARS_EMPLOYED`.
- Handled `XNA` values in `CODE_GENDER` and `ORGANIZATION_TYPE`.
- Kept genuine extreme values in the financial columns.

**Preprocessing (no data leakage)**
- Split the data first (80% train, 20% test, stratified).
- Fitted the imputation, scaling and encoding on the training data only, using a Scikit-learn pipeline.
- Numerical columns: median imputation and `StandardScaler`.
- Categorical columns: most-frequent imputation and one-hot encoding.
- Removed `DAYS_BIRTH` and `DAYS_EMPLOYED` because they duplicate `AGE` and `YEARS_EMPLOYED`.

**Handling class imbalance**
- Stratified train-test split.
- `class_weight='balanced'` in all models.
- Evaluation with Precision, Recall, F1-score and ROC-AUC instead of accuracy alone.

---

## Key Findings from EDA

- The payment difficulty rate decreases as education level increases (Lower secondary 10.93%, Higher education 5.36%).
- Income, credit amount, annuity and goods price show only small differences between the two groups.
- The external credit scores (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`) have the strongest relationship with `TARGET`.
- Age and employment length show a weak negative relationship with payment difficulties.
- All individual relationships are weak, so the models need to combine many features.

---

## Model Results

All models were tested on the same untouched test set (61,503 customers).

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.6894 | 0.1610 | **0.6763** | **0.2601** | **0.7483** |
| Random Forest | 0.7069 | 0.1618 | 0.6292 | 0.2574 | 0.7350 |
| Decision Tree | 0.6763 | 0.1496 | 0.6427 | 0.2427 | 0.7165 |

**Recommended model: Logistic Regression**
- It has the highest recall, F1-score and ROC-AUC.
- It found 3,358 of the 4,965 customers with payment difficulties (about 68%).
- It is simple, fast and easy to explain.

Accuracy was not used to choose the best model, because a model that predicts "no difficulty" for every customer would get about 92% accuracy without finding any defaulter.

---

## Customer Risk Segmentation

The Logistic Regression probabilities were used to split the test customers into four equal groups:

| Risk Group | Actual Payment Difficulty Rate |
|---|---|
| Low | 1.97% |
| Medium | 4.01% |
| High | 7.88% |
| Very High | 18.44% |

- The Very High group has a rate about 2.3 times higher than the overall rate (8.07%) and holds 57.1% of all customers with payment difficulties.
- These groups show relative risk. Even in the Very High group, about 82% of customers did not face difficulties.

---

## Key Factors

- The external credit scores (especially `EXT_SOURCE_3` and `EXT_SOURCE_2`) are the strongest factors in the Random Forest and Logistic Regression models.
- Employment length, age and education level also matter.
- These are patterns in historical data and do not prove cause.

---

## Challenges Faced (Summary)

- Many columns with a high percentage of missing values.
- Placeholder value `365243` in `DAYS_EMPLOYED`, which created an artificial spike after median filling.
- Data leakage from filling missing values before the split. Fixed with a pipeline fitted on training data only.
- Highly imbalanced target.
- Low precision, and probabilities pushed higher by the balanced class weights. Used ranked risk groups instead of fixed cut-offs.
- Similar features (`AMT_CREDIT` and `AMT_GOODS_PRICE`) gave unstable Logistic Regression coefficients.

The full table of challenges, techniques and reasons is in Section 14 of the notebook.

---

## Limitations

- Only `application_train.csv` was used. The other tables may improve the results.
- The best ROC-AUC is 0.7483, which is moderate. The model should not be used alone for real lending decisions.
- Precision is low, so many flagged customers are false alarms.
- The models were not tuned, and the default 0.5 cut-off was used.
- The predicted probabilities are not exact percentages. They should be used for ranking customers.

---

## Future Improvements

- Add features from the other tables (credit bureau, previous applications, payment history), with checks to avoid data leakage.
- Tune the models and choose the cut-off based on the cost of missed defaulters versus false alarms.
- Test SMOTE as an extra experiment on the training data only.
- Check the model for unfair differences between customer groups.

---

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

---

## How to Run

1. Clone the repository:
   ```bash
   git clone [your-repository-link]
   cd Home_Loan_Default_Prediction
   ```
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Download the dataset and place the files in the `dataset/` folder.
4. Open the notebook and run all cells:
   ```bash
   jupyter notebook Home_Loan_Default_Prediction.ipynb
   ```

---

## Author

**Chaitanya Girase**
- GitHub: [your-github-link]
- LinkedIn: [your-linkedin-link]
