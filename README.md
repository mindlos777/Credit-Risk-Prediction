# Credit Risk Prediction System

This is a simple Machine Learning 600 project that predicts whether a loan applicant is high-risk or low-risk.

## Project Files

- `credit_risk_prediction.py` - Python machine learning code
- `credit_risk_dataset.csv` - dataset used by the program
- `credit_risk_presentation.pptx` - project presentation

## Machine Learning Models

The project uses three classification models:

1. Logistic Regression
2. K-Nearest Neighbours (KNN)
3. Decision Tree Classifier

The models are trained before and after applying Principal Component Analysis (PCA).

## Main Steps

- Load and inspect the dataset
- Check missing values
- Perform Exploratory Data Analysis
- Clean the dataset
- Handle missing values and outliers
- Encode categorical variables
- Scale the features
- Split the data into training and testing sets
- Apply PCA
- Train three machine learning models
- Compare model performance
- Tune the Decision Tree using GridSearchCV

## Libraries Used

- pandas
- numpy
- matplotlib
- scikit-learn

## How to Run

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Make sure these files are in the same folder:

```text
credit_risk_prediction.py
credit_risk_dataset.csv
```

Run the project:

```bash
python credit_risk_prediction.py
```

The program will display dataset information, graphs, PCA results, accuracy, precision, recall, F1-score, confusion matrices, classification reports and tuning results.

## Dataset

The assignment requires the Credit Risk Dataset from Kaggle.

Dataset name:

```text
Credit Risk Dataset
```