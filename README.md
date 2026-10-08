# Credit Risk Prediction System

A simple machine learning project developed for **Machine Learning 600**. The system predicts whether a loan applicant is **Low Risk (0)** or **High Risk (1)** based on their financial and personal information.

The project includes data analysis, preprocessing, model training, evaluation, optimisation, and a Streamlit web application for visualising results and testing predictions.

## Project Structure

```text
Credit-Risk-Prediction/
│
├── main.py
├── credit_risk_prediction.py
├── credit_risk_dataset.csv
├── credit_risk_presentation.pptx
├── model/
│   └── DecisionTreeClassifier.pkl
└── README.md
```

## Machine Learning Models

Three classification algorithms are used:

1. **Logistic Regression** – Predicts the probability of an applicant belonging to a risk category.
2. **K-Nearest Neighbours (KNN)** – Classifies applicants based on similar records.
3. **Decision Tree Classifier** – Uses decision rules to classify applicants.

The models are evaluated both **before and after PCA** to compare their performance.

## Machine Learning Pipeline

The project follows these steps:

1. **Data Acquisition** – Load and inspect the Credit Risk Dataset.
2. **Exploratory Data Analysis (EDA)** – Explore distributions, relationships, and outliers.
3. **Data Cleaning** – Remove duplicates and handle missing values.
4. **Outlier Handling** – Use the Interquartile Range (IQR) method to limit extreme values.
5. **Encoding** – Convert categorical features into numerical values.
6. **Feature Scaling** – Standardise features using StandardScaler.
7. **Data Splitting** – Split the dataset into 80% training and 20% testing.
8. **PCA** – Reduce dimensionality while retaining at least 95% of the explained variance.
9. **Model Training** – Train Logistic Regression, KNN, and Decision Tree models.
10. **Model Evaluation** – Compare the models using classification metrics.
11. **Hyperparameter Tuning** – Optimise the Decision Tree using GridSearchCV.
12. **Model Saving** – Save the optimised Decision Tree using Joblib.

## Streamlit Web Application

The project includes a simple web application built using **Streamlit**.

### Dashboard

The Dashboard displays:

- Dataset preview and information
- Total records, features, and missing values
- Loan status distribution
- Correlation heatmap
- Boxplots for outlier detection
- Feature distribution histograms
- Model performance before and after PCA
- Confusion matrices displayed in a 2 × 3 graph layout
- Classification reports for all three models
- Best Decision Tree parameters and evaluation scores

### Test Model

The Test Model page allows users to enter applicant information, including:

- Age and annual income
- Loan amount and interest rate
- Home ownership and employment length
- Loan purpose and grade
- Loan percentage of income
- Previous default history
- Credit history length

The application uses the saved Decision Tree model to predict the applicant's credit risk.

**Prediction Output:**

- **0 – Low Risk**
- **1 – High Risk**

The predictions are demonstrations of the trained model and should not be used as real lending decisions without further validation.

## Model Evaluation

The models are evaluated using the following metrics:

| Metric | Description |
|---|---|
| Accuracy | Percentage of correct predictions |
| Precision | How many predicted high-risk applicants are actually high-risk |
| Recall | How many actual high-risk applicants were identified |
| F1-Score | Balance between precision and recall |
| Confusion Matrix | Shows correct and incorrect classifications |
| Classification Report | Summarises precision, recall, F1-score, and support |

The project compares these metrics before and after applying PCA.

## Technologies and Libraries

- **Python** – Main programming language
- **Pandas** – Data manipulation
- **NumPy** – Numerical operations
- **Matplotlib** – Data visualisation
- **Scikit-learn** – Machine learning and evaluation
- **Streamlit** – Web application
- **Joblib** – Model saving and loading

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mindlos777/Credit-Risk-Prediction
cd YOUR-REPOSITORY
```

Replace the URL with your GitHub repository link.

### 2. Install Dependencies

Make sure Python is installed, then run:

```bash
pip install pandas numpy matplotlib scikit-learn streamlit joblib
```
OR:
```uv package manager
uv add -r requiremnts.txt
```

### 3. Prepare the Dataset

Make sure `credit_risk_dataset.csv` is in the same directory as the Python files.

### 4. Run the Machine Learning Program

```bash
python credit_risk_prediction.py
```

This runs the machine learning workflow, including preprocessing, PCA, model evaluation, and hyperparameter tuning.

The optimised Decision Tree model is saved in the `model` folder.

### 5. Run the Streamlit Application

```bash
streamlit run main.py
```

Streamlit will open the application in your browser, usually at:

`http://localhost:8501`

Use the sidebar to navigate between **Dashboard** and **Test Model**.

## Dataset

The project uses the **Credit Risk Dataset** available on Kaggle.

The dataset contains applicant information such as age, income, employment length, home ownership, loan details, and credit history.

**Target Variable:** `loan_status`

- `0` – Low Risk
- `1` – High Risk

## Project Purpose

The aim of this project is to demonstrate how supervised machine learning can be used to classify credit risk.

It also explores how data preprocessing, dimensionality reduction, and hyperparameter tuning affect model performance.

## Future Improvements

- Save the complete preprocessing pipeline with the trained model.
- Improve the model testing interface and input validation.
- Add additional classification algorithms.
- Compare models using ROC curves and AUC scores.
- Deploy the Streamlit application online.

---

**Machine Learning 600 | Credit Risk Prediction Project**
