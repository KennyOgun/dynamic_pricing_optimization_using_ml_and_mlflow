import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
import mlflow
import mlflow.pyfunc
import requests

# MLflow Tracking Server Configuration
MLFLOW_TRACKING_URI = "http://ec2-13-42-33-239.eu-west-2.compute.amazonaws.com:5000/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Streamlit Configuration
st.set_page_config(
    page_title="Jewelry Price Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and Introduction
st.title("Jewelry Price Prediction App")
st.write("""
This app predicts jewelry prices based on selected features using a machine learning model integrated with MLflow.
- Upload your dataset for analysis.
- Train models or use pre-trained models for predictions.
- Compare actual and predicted prices.
""")

# Sidebar for Navigation
st.sidebar.header("App Navigation")
page = st.sidebar.radio(
    "Choose a page", 
    ["Overview", "Data Analysis", "Model Training and Evaluation", "Predict Jewelry Price"]
)

# Helper Functions
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    return None

def create_pipeline():
    categorical_features = ['Category', 'Target_Gender', 'Main_Color', 'Main_Metal', 'Main_Gem']
    numerical_features = ['Brand_ID']
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ])
    return preprocessor

def preprocess_data(X_train, X_test, preprocessor):
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def evaluate_model(y_test, preds, model_name):
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {"Model": model_name, "MAE": mae, "MSE": mse, "R2": r2}

@st.cache_resource(allow_output_mutation=True)
def load_mlflow_model():
    model_uri = "models:/jewelry_price_model/production"  # Replace with your registered model path
    return mlflow.pyfunc.load_model(model_uri)

# Main Functionality
def main():
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
    df = load_data(uploaded_file)

    if page == "Overview":
        st.subheader("Overview")
        st.write("""
        This app leverages machine learning to predict jewelry prices. 
        Upload your dataset to explore the features, train models, or make predictions.
        """)
        if df is not None:
            st.write("### Dataset Preview")
            st.dataframe(df.head())
        else:
            st.info("Upload a dataset to proceed.")

    elif page == "Data Analysis":
        if df is not None:
            st.subheader("Exploratory Data Analysis")
            st.write("### Distribution of Jewelry Categories")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(data=df, x='Category', hue='Target_Gender', ax=ax, palette="viridis")
            ax.set_title("Jewelry Category Distribution by Gender")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
        else:
            st.warning("Please upload a dataset to analyze.")

    elif page == "Model Training and Evaluation":
        if df is not None:
            st.subheader("Train and Evaluate Models")
            target_column = 'Price_USD'
            test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.3)
            
            model_options = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "LightGBM": LGBMRegressor(random_state=42)
            }
            selected_models = st.sidebar.multiselect("Select Models", list(model_options.keys()))

            X = df.drop(columns=[target_column])
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            preprocessor = create_pipeline()
            X_train, X_test = preprocess_data(X_train, X_test, preprocessor)

            results = []
            for model_name in selected_models:
                model = model_options[model_name]
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                metrics = evaluate_model(y_test, preds, model_name)
                results.append(metrics)

            if results:
                results_df = pd.DataFrame(results)
                st.write("### Model Performance")
                st.dataframe(results_df)
        else:
            st.warning("Please upload a dataset to train models.")

    elif page == "Predict Jewelry Price":
        st.subheader("Predict Jewelry Price")
        category = st.selectbox("Category", ["Jewelry.Pendant", "Jewelry.Necklace", "Jewelry.Earring", 
                                             "Jewelry.Ring", "Jewelry.Brooch", "Jewelry.Bracelet", 
                                             "Jewelry.Souvenir"])
        main_color = st.selectbox("Main Color", ["yellow", "white", "red"])
        main_metal = st.selectbox("Main Metal", ["gold", "silver"])
        main_gem = st.selectbox("Main Gem", ["sapphire", "diamond", "amethyst", "fianit", "pearl", 
                                             "topaz", "garnet"])

        input_data = pd.DataFrame({
            "Category": [category],
            "Main_Color": [main_color],
            "Main_Metal": [main_metal],
            "Main_Gem": [main_gem]
        })

        if st.button("Predict Price"):
            model = load_mlflow_model()
            with st.spinner("Predicting..."):
                try:
                    predicted_price = model.predict(input_data)[0]
                    st.success(f"Predicted Price: ${predicted_price:.2f}")
                except Exception as e:
                    st.error(f"Error in prediction: {e}")

        if st.checkbox("Compare Predicted and Actual Prices"):
            actual_price = st.number_input("Actual Price (USD)", min_value=0.0)
            if "predicted_price" in locals():
                difference = actual_price - predicted_price
                st.write(f"Actual Price: ${actual_price:.2f}")
                st.write(f"Predicted Price: ${predicted_price:.2f}")
                st.write(f"Difference: ${difference:.2f}")

if __name__ == "__main__":
    main()
