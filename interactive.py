import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
import plotly.express as px
import os

# Set Streamlit page title and icon
st.set_page_config(page_title="Loan Default Dashboard", page_icon="📊", layout="wide")

# Title and subtitle
st.title("📊 Loan Default Prediction Dashboard")
st.markdown("A powerful tool for understanding model performance and feature impact.")

import os
import pandas as pd
import streamlit as st
import zipfile

# File path for the saved dataset
DATA_FILE = "trainnn.csv"
ZIP_FILE = "trainnn.csv.zip"

# Load the dataset without asking the user to upload
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    st.write("### ✅ Loaded Saved Dataset (`trainnn.csv`)")
elif os.path.exists(ZIP_FILE):
    # Extract the dataset if only the ZIP is present
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(".")
    df = pd.read_csv(DATA_FILE)
    st.write("### ✅ Loaded Extracted Dataset (`trainnn.csv.zip`)")
else:
    st.error("🚫 Dataset file 'trainnn.csv' or 'trainnn.csv.zip' not found. Please make sure it is available.")

# Preview the dataset
if "df" in locals():
    st.write(df.head())

# Proceed if the dataset is loaded
if "df" in locals():
    # Split features/target
    X = df.drop(columns="Default")
    y = df["Default"]

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Transformers
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numerical_transformer = StandardScaler()

    # Preprocessor
    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    # Model setup
    model = RandomForestClassifier(random_state=42)
    pipe = ImbPipeline([
        ("preprocess", preprocessor),
        ("clf", model)
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Model training
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"### 🏆 Model Accuracy: {acc:.4f}")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=["Non-Default", "Default"], 
                       y=["Non-Default", "Default"],
                       title="Confusion Matrix",
                       color_continuous_scale="Blues")
    st.plotly_chart(fig_cm)

    # Feature importance (if available)
    if hasattr(pipe["clf"], "feature_importances_"):
        # Extract numerical and categorical feature names
        categorical_names = pipe.named_steps["preprocess"].transformers_[1][1].fit(X[categorical_cols]).get_feature_names_out(categorical_cols)
        feature_names = numerical_cols + list(categorical_names)

        # Get feature importances
        feature_importances = pipe["clf"].feature_importances_
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
        feat_df = feat_df.sort_values(by="Importance", ascending=False).head(10)

        # Plot feature importances
        st.write("### 🔍 Top 10 Feature Importances")
        fig_feat = px.bar(feat_df, x="Importance", y="Feature", orientation="h", title="Top 10 Features", color="Importance",
                          color_continuous_scale=px.colors.sequential.Teal)
        st.plotly_chart(fig_feat)

        # Add comments about feature strength
        st.markdown("### 📌 Key Feature Insights")
        for i, row in feat_df.iterrows():
            if row["Importance"] > 0.05:
                st.markdown(f"**{row['Feature']}** is a **strong** predictor of loan default.")
            elif row["Importance"] > 0.02:
                st.markdown(f"**{row['Feature']}** has a **moderate** impact on default prediction.")
            else:
                st.markdown(f"**{row['Feature']}** has a **minor** influence on the model's decision.")

    # Category distribution
    st.write("### 📊 Category Distribution")
    st.markdown("The loan default dataset contains a mix of categorical and numerical features that describe the financial and demographic characteristics of loan applicants. Understanding the distribution of these categorical features is crucial for accurate model training and meaningful insights.")
    st.markdown("such as Employemnt type,Education Level, and Martial status. ")



    # Pie chart for Age distribution (if available)
    st.write("### 📊 Age Insights")
    st.markdown("Younger borrowers may have higher default rates due to less financial stability and limited credit history. ")
    st.markdown("Middle-aged borrowers might be more financially stable but have higher financial commitments (e.g., mortgages and family expenses).")
    
        

# Footer
st.markdown("---")
st.markdown("© 2025 **Loan Default Predictor**")

