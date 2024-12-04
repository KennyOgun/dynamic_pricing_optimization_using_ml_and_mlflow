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
import mlflow.sklearn
import requests

# # MLflow Setup
mlflow.set_tracking_uri("http://localhost:5000")  # Replace with your server's URI if deployed elsewhere
mlflow.set_experiment("Jewelry Price Optimization-Streamlit_mlflow") # name your experiment

#Add a check in your code to confirm the URI is reachable
import requests

tracking_uri = "http://localhost:5000"
try:
    response = requests.get(tracking_uri)
    if response.status_code == 200:
        print("MLflow Tracking Server is reachable.")
        mlflow.set_tracking_uri(tracking_uri)
    else:
        print("MLflow Tracking Server is not reachable.")
except Exception as e:
    print(f"Error connecting to MLflow Tracking Server: {e}")


# Initialize the variable to avoid UnboundLocalError
selected_models = None

# Streamlit Configuration
st.set_page_config(
    page_title="Jewellery Price Optimization",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize the variable to avoid UnboundLocalError
selected_models = None

# Helper Functions
def split_data(df, target_column='Price_USD', test_size=0.3, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_pipeline():
    categorical_features = ['Category', 'Target_Gender', 'Main_Color', 'Main_Metal', 'Main_Gem']
    numerical_features = ['Brand_ID']

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_features),
            ('cat', cat_pipeline, categorical_features)
        ]
    )
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

def log_model_with_mlflow(model, model_name, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        preds_train = model.predict(X_train)
        preds_test = model.predict(X_test)

        mlflow.log_metric("train_R2", r2_score(y_train, preds_train))
        mlflow.log_metric("test_R2", r2_score(y_test, preds_test))
        mlflow.log_metric("train_MAE", mean_absolute_error(y_train, preds_train))
        mlflow.log_metric("test_MAE", mean_absolute_error(y_test, preds_test))
        mlflow.log_metric("train_MSE", mean_squared_error(y_train, preds_train))
        mlflow.log_metric("test_MSE", mean_squared_error(y_test, preds_test))

        mlflow.sklearn.log_model(model, model_name)

# Main App
def main():
    st.title("Jewellery Price Optimization")
    st.sidebar.header("App Navigation")
    page = st.sidebar.radio("Choose a page", ["Overview", "Data Analysis", "Model Training and Evaluation"])

    # Initialize selected_models
    selected_models = []

    # Define `target_column` early to avoid UnboundLocalError
    target_column = 'Price_USD'

    # Load Dataset
    @st.cache_data
    def load_data(uploaded_file):
        if uploaded_file:
            return pd.read_csv(uploaded_file)
        else:
            return None

    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
    df = load_data(uploaded_file)

    if df is not None:
        if page == "Overview":
            st.subheader("Project Overview")
            st.write("This app optimizes jewelry pricing using machine learning, integrated with MLflow.")
            st.dataframe(df.head())
            st.subheader("Dataset Shape")
            st.write(f"Shape: {df.shape}")
            st.subheader("Dataset Description")
            st.write("1. Brand ID: Identifier for jeweler brand.")
            st.write("2. Price in USD: Jewelry price in US Dollars.")
            st.write("3. Product gender (for male/female) (Target gender for jewelry piece).")
            st.write("4. Main Color: Overall color of jewelry piece.")
            st.write("5. Main metal: Main metal used for mounting.")
            st.write("6. Main gem: Main gem mounted on jewelry piece.")
            st.write("7. Category: Name of jewelry category e.g., earring.")

        elif page == "Data Analysis":
            st.subheader("Exploratory Data Analysis")
            category_gender_order = (
                df.groupby('Category')
                .size()
                .sort_values(ascending=False)
                .index
            )
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(
                data=df,
                x='Category',
                hue='Target_Gender',
                ax=ax,
                palette="viridis",
                order=category_gender_order
            )
            ax.set_title("Jewelry Category Distribution by Gender")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)

            category_order = (
                df['Category']
                .value_counts()
                .index
            )
            st.write("### Jewellery Category Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(
                data=df,
                x='Category',
                ax=ax,
                palette="viridis",
                order=category_order
            )
            ax.set_title("Bar Chart: Jewellery Category Distribution")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)

            st.write("### Please select models in the sidebar before proceeding to 'Model Training and Evaluation'.")

        elif page == "Model Training and Evaluation":
            st.subheader("Model Training and Evaluation")

            random_state = 42
            test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.2, 0.3)

            model_options = {
                "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200, max_depth=30),
                "XGBoost": XGBRegressor(random_state=42, learning_rate=0.2, max_depth=10, n_estimators=300),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42, learning_rate=0.2, max_depth=7),
                "LightGBM": LGBMRegressor(random_state=42, max_depth=31, learning_rate=0.3)
            }
            selected_models = st.sidebar.multiselect("Select Models for Training", list(model_options.keys()))

            if not selected_models:
                st.warning("No models selected. Please select at least one model from the sidebar.")
            else:
                st.info("MLflow Model Tracking is running...")
                mlflow_server_url = "http://localhost:5000"
                st.markdown(f"[Explore the MLflow Tracking Server]({mlflow_server_url})", unsafe_allow_html=True)

            # Split and Preprocess Data
            X_train, X_test, y_train, y_test = split_data(df, target_column, test_size=test_size)
            preprocessor = create_pipeline()
            X_train, X_test = preprocess_data(X_train, X_test, preprocessor)

            results = []
            if selected_models:
                for model_name in selected_models:
                    model = model_options[model_name]
                    with mlflow.start_run(run_name=model_name):
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        mlflow.sklearn.log_model(model, artifact_path="model")
                        mlflow.log_param("Model Type", model_name)
                        mlflow.log_param("Test Size", test_size)
                        r2 = r2_score(y_test, preds)
                        mae = mean_absolute_error(y_test, preds)
                        mse = mean_squared_error(y_test, preds)
                        mlflow.log_metric("R2", r2)
                        mlflow.log_metric("MAE", mae)
                        mlflow.log_metric("MSE", mse)
                        results.append({
                            "Model": model_name,
                            "R2": r2,
                            "MAE": mae,
                            "MSE": mse
                        })

            if results:
                results_df = pd.DataFrame(results)
                st.write("### Model Performance Metrics")
                st.dataframe(results_df)

                st.write("### R² Metric for Models")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Model', y='R2', data=results_df, palette="viridis", ax=ax)
                ax.set_title("R² Metric by Model")
                ax.set_ylabel("R² Score")
                ax.set_xlabel("Model")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                st.pyplot(fig)
# Run the App
if __name__ == "__main__":
    main()

