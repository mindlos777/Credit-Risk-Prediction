import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# PART A: DATA ACQUISITION
df = pd.read_csv("credit_risk_dataset.csv")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nInfo:")
df.info()
print("\nDescription:\n", df.describe(include="all"))
print("\nMissing values:\n", df.isnull().sum())

# PART B: EDA
plt.figure()
df["loan_status"].value_counts().sort_index().plot(kind="bar")
plt.title("Target Variable Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
# Interpretation: 0 means lower risk and 1 means higher risk/default.

numeric = df.select_dtypes(include=np.number)
corr = numeric.corr()
plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
# Interpretation: the heatmap shows positive and negative relationships between numerical variables.

plt.figure()
df[["person_income", "loan_amnt", "loan_int_rate"]].boxplot()
plt.title("Boxplots for Outliers")
plt.tight_layout()
plt.show()
# Interpretation: extreme values can be seen outside the main range of the boxes.

plt.figure()
df[["person_age", "person_income", "loan_amnt"]].hist(figsize=(8, 6))
plt.tight_layout()
plt.show()
# Interpretation: histograms show how the selected numerical features are distributed.

# PART C: CLEANING AND PREPROCESSING
df = df.drop_duplicates()

for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(include=np.number).columns:
    if col != "loan_status":
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)

X = df.drop("loan_status", axis=1)
y = df["loan_status"]
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Scaling is needed because models such as KNN 
# and PCA are affected by different feature sizes.

# PART D: PCA
pca_full = PCA()
pca_full.fit(X_train_scaled)
print("\nExplained variance ratio:\n", pca_full.explained_variance_ratio_)

cumulative = np.cumsum(pca_full.explained_variance_ratio_)
plt.figure()
plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o")
plt.axhline(0.95, linestyle="--")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.title("PCA Cumulative Variance")
plt.tight_layout()
plt.show()

n_components = np.argmax(cumulative >= 0.95) + 1
print("Components for at least 95% variance:", n_components)

pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# PART E: MODEL DEVELOPMENT
def evaluate(name, model, Xtr, Xte):
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, zero_division=0))
    print("Recall:", recall_score(y_test, pred, zero_division=0))
    print("F1-score:", f1_score(y_test, pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print("Classification Report:\n", classification_report(y_test, pred, zero_division=0))
    return accuracy_score(y_test, pred)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

print("\nBEFORE PCA")
before = {}
for name, model in models.items():
    before[name] = evaluate(name, model, X_train_scaled, X_test_scaled)

print("\nAFTER PCA")
after = {}
for name, model in models.items():
    after[name] = evaluate(name, model, X_train_pca, X_test_pca)

print("\nComparison before PCA:", before)
print("Comparison after PCA:", after)

# PART F: MODEL OPTIMISATION
params = {
    "max_depth": [2, 3, 4, 5, None],
    "min_samples_split": [2, 4, 6]
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=3)
grid.fit(X_train_scaled, y_train)

print("\nBest parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)

best_tree = grid.best_estimator_
pred = best_tree.predict(X_test_scaled)
print("Tuned Decision Tree Accuracy:", accuracy_score(y_test, pred))
print("Tuned Decision Tree Classification Report:\n", classification_report(y_test, pred, zero_division=0))

# Tuning controls model complexity. A simpler tree can reduce overfitting (variance),
# while a tree that is too simple can increase underfitting (bias).
