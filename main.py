
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder

from credit_risk_prediction import (
    data_acquisition,
    before_after_pca,
    X,
    scaler
)


st.set_page_config(
    page_title="Credit Risk Prediction",
    layout="wide"
)


# STYLE
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
        color: black;
    }

    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #e2e8f0;
    }

    h1, h2, h3, h4, p, label {
        color: black !important;
    }

    [data-testid="stSidebar"] * {
        color: black;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
    }

    [data-testid="stMetric"] * {
        color: black !important;
    }

    [data-testid="stCode"] {
        background-color: #1e293b;
    }

    [data-testid="stCode"] code,
    [data-testid="stCode"] code * {
        color: white !important;
    }

    .stButton > button[kind="primary"] {
        background-color: #2563eb;
        color: white !important;
        width: 100%;
        border-radius: 8px;
    }

    .stButton > button[kind="primary"] * {
        color: white !important;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# SIDEBAR
st.sidebar.title("Credit Risk")
st.sidebar.caption("Machine Learning Dashboard")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Test Model"],
    label_visibility="collapsed"
)


# FUNCTIONS
@st.cache_data
def load_data():
    return data_acquisition()


@st.cache_resource
def load_results():
    return before_after_pca()


def results_table(results):
    rows = []

    for name, values in results.items():
        rows.append({
            "Model": name,
            "Accuracy": values[1],
            "Precision": values[2],
            "Recall": values[3],
            "F1 Score": values[4]
        })

    return pd.DataFrame(rows)


def plot_confusion_matrices(before, after):

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    for row, results in enumerate([before, after]):

        for col, (name, values) in enumerate(results.items()):

            ax = axes[row, col]
            cm = values[5]

            ax.imshow(cm, cmap="Blues")

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black"
                    )

            ax.set_title(
                f"{name}\n{'Before PCA' if row == 0 else 'After PCA'}"
            )

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])

    plt.tight_layout()
    return fig


# DASHBOARD
if page == "Dashboard":

    st.title("Credit Risk Prediction")

    st.write("""
    This application uses machine learning to analyse
    loan applicants and predict their credit risk.
    """)

    st.divider()

    # DATA ACQUISITION
    st.header("Exploratory Data Analysis")

    data = load_data()
    df = data[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", df.shape[0])

    with col2:
        st.metric("Total Features", df.shape[1])

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)

    with st.expander("Dataset Information"):
        st.write("Shape:", data[1])
        st.write("Columns:", data[2])
        st.write("Missing Values:")
        st.write(data[5])
        st.write("Statistical Description:")
        st.dataframe(data[4], use_container_width=True)

    st.divider()

    # EDA GRAPHS
    st.header("Data Visualizations")

    # LOAN STATUS
    st.subheader("Loan Status Distribution")

    fig, ax = plt.subplots()

    df["loan_status"].value_counts().sort_index().plot(
        kind="bar",
        ax=ax,
        color="#2563eb"
    )

    ax.set_xlabel("Loan Status")
    ax.set_ylabel("Count")
    ax.set_title("Target Variable Distribution")

    st.pyplot(fig)
    plt.close(fig)

    # CORRELATION
    st.subheader("Correlation Heatmap")

    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(
        corr,
        cmap="coolwarm",
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # BOXPLOTS
    st.subheader("Boxplots for Outliers")

    fig, ax = plt.subplots(figsize=(8, 4))

    df[
        ["person_income", "loan_amnt", "loan_int_rate"]
    ].boxplot(ax=ax)

    ax.set_title("Boxplots for Outliers")

    st.pyplot(fig)
    plt.close(fig)

    # HISTOGRAMS
    st.subheader("Feature Distributions")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    columns = [
        "person_age",
        "person_income",
        "loan_amnt"
    ]

    for i, column in enumerate(columns):

        axes[i].hist(
            df[column].dropna(),
            bins=20,
            color="#2563eb",
            edgecolor="white"
        )

        axes[i].set_title(column)
        axes[i].set_ylabel("Frequency")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # MODEL EVALUATION
    st.header("Model Evaluation")

    results = load_results()

    before = results[0]
    after = results[1]

    tab1, tab2 = st.tabs(["Before PCA", "After PCA"])

    with tab1:
        st.dataframe(
            results_table(before),
            use_container_width=True
        )

    with tab2:
        st.dataframe(
            results_table(after),
            use_container_width=True
        )

    st.divider()

    # CONFUSION MATRICES
    st.header("Confusion Matrices")

    st.write(
        "Compare the confusion matrices before and after PCA."
    )

    fig = plot_confusion_matrices(before, after)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "First row: Before PCA | Second row: After PCA"
    )

    st.divider()

    # CLASSIFICATION REPORTS
    st.header("Classification Reports")

    report_tab1, report_tab2 = st.tabs(
        ["Before PCA", "After PCA"]
    )

    with report_tab1:
        for name, values in before.items():
            st.subheader(name)
            st.code(values[6], language=None)

    with report_tab2:
        for name, values in after.items():
            st.subheader(name)
            st.code(values[6], language=None)

    st.divider()

    # BEST MODEL
    st.header("Best Model Results")

    with st.container(border=True):

        st.write("**Best Parameters:**")
        st.code(str(results[2]), language=None)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Best Cross-Validation Score",
                f"{results[3]:.4f}"
            )

        with col2:
            st.metric(
                "Test Accuracy",
                f"{results[4]:.4f}"
            )

    st.subheader("Optimized Decision Tree Report")
    st.code(results[5], language=None)


# TEST MODEL
elif page == "Test Model":

    st.title("Test Model")

    st.write(
        "Enter applicant details to predict their credit risk."
    )

    st.divider()

    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)

    with col1:

        person_age = st.slider("Age", 18, 100, 30)

        person_income = st.number_input(
            "Annual Income", min_value=0, value=50000
        )

        loan_amnt = st.number_input(
            "Loan Amount", min_value=1000, value=10000
        )

        loan_int_rate = st.slider(
            "Interest Rate", 0.0, 30.0, 5.0
        )

        person_home_ownership = st.selectbox(
            "Home Ownership",
            ["RENT", "OWN", "MORTGAGE", "OTHER"]
        )

        person_emp_length = st.slider(
            "Employment Length (Years)", 0.0, 40.0, 5.0
        )

    with col2:

        loan_intent = st.selectbox(
            "Loan Purpose",
            [
                "PERSONAL", "EDUCATION", "MEDICAL",
                "VENTURE", "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION"
            ]
        )

        loan_grade = st.selectbox(
            "Loan Grade",
            ["A", "B", "C", "D", "E", "F", "G"]
        )

        loan_percent_income = st.slider(
            "Loan Percent of Income", 0.0, 1.0, 0.20
        )

        cb_person_default_on_file = st.selectbox(
            "Previous Default", ["N", "Y"]
        )

        cb_person_cred_hist_length = st.slider(
            "Credit History Length (Years)", 0, 40, 5
        )

    input_data = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }])

    with st.expander("View Applicant Information"):
        st.dataframe(input_data, use_container_width=True)

    st.divider()

    st.subheader("Prediction Result")

    if st.button("Predict Risk", type="primary"):

        model = joblib.load(
            "model/DecisionTreeClassifier.pkl"
        )

        processed_input = input_data.copy()
        original_df = pd.read_csv("credit_risk_dataset.csv")

        for column in processed_input.select_dtypes(
            include="object"
        ).columns:

            encoder = LabelEncoder()

            encoder.fit(
                original_df[column].dropna().astype(str)
            )

            processed_input[column] = encoder.transform(
                processed_input[column].astype(str)
            )

        processed_input = processed_input.reindex(
            columns=X.columns,
            fill_value=0
        )

        processed_scaled = scaler.transform(
            processed_input
        )

        prediction = model.predict(processed_scaled)

        if prediction[0] == 0:
            st.success(
                "The applicant is predicted to be LOW RISK."
            )
        else:
            st.error(
                "The applicant is predicted to be HIGH RISK."
            )
