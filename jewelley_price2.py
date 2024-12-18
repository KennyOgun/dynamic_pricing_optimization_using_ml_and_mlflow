import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import mlflow
import mlflow.pyfunc

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit Page Configuration
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

# MLflow Tracking Server Configuration (DagsHub)
dagshub_mlflow_url = "https://dagshub.com/KennyOgun/jewellery_price_optimization.mlflow"
mlflow.set_tracking_uri(dagshub_mlflow_url)

# Authenticate with DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["DAGSHUB_USERNAME"]  # Store these in Streamlit secrets
os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["DAGSHUB_TOKEN"]

# Display MLflow Tracker
st.markdown("""
### MLflow Experiment Tracking
Explore the experiment tracking hosted on DagsHub:
""")
st.markdown(f"[Open MLflow Tracking Server]({dagshub_mlflow_url})", unsafe_allow_html=True)
st.components.v1.iframe(dagshub_mlflow_url, width=1000, height=600)

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

@st.cache_resource
def load_mlflow_model():
    model_uri = "models:/jewelry_price_model/production"  # Replace with your registered model path
    return mlflow.pyfunc.load_model(model_uri)

# Sidebar for Navigation
st.sidebar.header("App Navigation")
page = st.sidebar.radio(
    "Choose a page", 
    ["Overview", "Data Analysis", "Model Training and Evaluation", "Predict Jewelry Price"]
)

# Main Functionality
def main():
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])
    df = load_data(uploaded_file)

    if page == "Overview":
        st.subheader("Overview")
        st.write("This app leverages machine learning to predict jewelry prices.")
        if df is not None:
            st.write("### Dataset Preview")
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
        else:
            st.info("Upload a dataset to proceed.")

    elif page == "Data Analysis":
        if df is not None:
            st.subheader("Exploratory Data Analysis")
            st.write("### Distribution of Jewelry Categories")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(data=df, x='Category', hue='Target_Gender', ax=ax, palette="viridis")
            ax.set_title("Jewelry Category Distribution by Gender")
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


            
        else:
            st.warning("Please upload a dataset to analyze.")

    elif page == "Model Training and Evaluation":
        if df is not None:
            st.subheader("Train and Evaluate Models")
            target_column = 'Price_USD'
            test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.3)
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            preprocessor = create_pipeline()
            X_train, X_test = preprocess_data(X_train, X_test, preprocessor)
            
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate_model(y_test, preds, "Random Forest")
            st.write("### Model Performance")
            st.json(metrics)
        else:
            st.warning("Please upload a dataset to train models.")

    elif page == "Predict Jewelry Price":
        st.subheader("Predict Jewelry Price")
        input_data = {"Category": ["Jewelry.Ring"], "Main_Color": ["white"], "Main_Metal": ["gold"], "Main_Gem": ["diamond"]}
        if st.button("Predict Price"):
            model = load_mlflow_model()
            prediction = model.predict(pd.DataFrame(input_data))
            st.success(f"Predicted Price: ${prediction[0]:.2f}")

if __name__ == "__main__":
    main()
