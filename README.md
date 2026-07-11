# CreditWise: Intelligent Loan Approval & Risk Prediction System

An end-to-end supervised machine learning pipeline that predicts loan approval outcomes (Approved / Rejected) for a simulated bank — **SecureTrust Bank** — using applicant financial, credit, and demographic data.

## Problem Statement

SecureTrust Bank, a mid-sized financial company offering personal and home loans across urban and rural India, relies on a manual verification process where loan officers evaluate applications by checking income proofs, employment details, and credit history. This process is time-consuming, inconsistent, and prone to bias, leading to two key issues:

1. **Good customers get rejected** → loss of business
2. **High-risk customers get approved** → financial losses

This project builds a machine learning system to automatically analyze applicant data and predict loan approval, supporting faster and more consistent decisions ahead of final human verification.

## Dataset

Each row represents a loan applicant with personal, financial, and credit attributes.

| Column | Description |
|---|---|
| Applicant_ID | Unique applicant ID |
| Applicant_Income | Monthly income of applicant |
| Coapplicant_Income | Monthly income of co-applicant |
| Employment_Status | Salaried / Self-Employed / Business |
| Age | Applicant age |
| Marital_Status | Married / Single |
| Dependents | Number of dependents |
| Credit_Score | Credit bureau score |
| Existing_Loans | Number of already running loans |
| DTI_Ratio | Debt-to-Income ratio |
| Savings | Savings balance |
| Collateral_Value | Value of collateral provided |
| Loan_Amount | Loan amount requested |
| Loan_Term | Loan duration (months) |
| Loan_Purpose | Home / Education / Personal / Business |
| Property_Area | Urban / Semi-Urban / Rural |
| Education_Level | Graduate / Postgraduate / Undergraduate |
| Gender | Male / Female |
| Employer_Category | Govt / Private / Self |
| **Loan_Approved (Target)** | 1 = Approved, 0 = Rejected |

## Approach

1. **Exploratory Data Analysis (EDA)** — examined distributions, missing values, and relationships between applicant attributes and loan approval.
2. **Data Cleaning & Feature Engineering** — handled missing values, encoded categorical variables, engineered derived features, and scaled numerical features (28 final features after encoding).
3. **Train/Test Split** — split data into training and test sets for unbiased evaluation.
4. **Model Training** — trained three binary classification models:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Gaussian Naive Bayes
5. **Model Evaluation** — since accuracy alone is a poor metric for loan approval (class imbalance risk), models were evaluated using **Precision, Recall, F1-Score, and Confusion Matrix**.

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | 0.880 | 0.785 | 0.836 | 0.810 |
| KNN (k=5) | 0.785 | 0.673 | 0.574 | 0.619 |
| Naive Bayes | 0.860 | 0.811 | 0.705 | 0.754 |

**Key takeaway:** Logistic Regression achieved the best overall balance of precision and recall (highest F1-Score), making it the strongest candidate model. Naive Bayes achieved the highest precision, which is valuable in scenarios where minimizing false approvals (high-risk customers wrongly approved) is the priority.

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, seaborn, matplotlib
- **Environment:** Jupyter Notebook / Google Colab

## Project Structure

creditwise-loan-approval-ml/
├── CreditWise_Loan_System.ipynb   # Main notebook: EDA, feature engineering, modeling
├── loan_approval_data.csv         # Dataset
├── README.md
└── requirements.txt

## How to Run

1. Clone the repository:
```bash
   git clone https://github.com/sumitstat07/creditwise-loan-approval-ml.git
   cd creditwise-loan-approval-ml
```
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Launch the notebook:
```bash
   jupyter notebook CreditWise_Loan_System.ipynb
```

## Future Improvements

- Add ensemble models (Random Forest, XGBoost, Gradient Boosting) for comparison
- Hyperparameter tuning via GridSearchCV / RandomizedSearchCV
- Handle class imbalance using SMOTE or class weighting
- Deploy as a web app (Flask/Streamlit) for interactive loan approval predictions
- Add SHAP/feature importance analysis for model explainability

## Author

**Sumit Sana**




