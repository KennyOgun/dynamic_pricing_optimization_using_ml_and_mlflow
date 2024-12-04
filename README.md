# 🛠️ Jewelry Price Optimization - Streamlit with MLflow Integration

This repository contains a Streamlit application that demonstrates the integration of machine learning models for optimizing jewelry pricing. The project uses MLflow for model tracking, parameter logging, and performance evaluation.


🚀 **Features**

  * Interactive **Streamlit Dashboard** for data visualization and model training.
    
  * Integration with **MLflow** for experiment tracking.
    
  * Support for multiple regression models:
    
      * Random Forest Regressor
      * XGBoost Regressor
      * Gradient Boosting Regressor
      * LightGBM Regressor
        
  * **Data Preprocessing Pipelines:** Handles categorical and numerical features efficiently.

  * Real-time evaluation of models with metrics like MAE, MSE, and R².

🧰 **Prerequisites**

Ensure you have the following installed:

  * Python 3.8+
    
  * Streamlit (pip install streamlit)
    
  * MLflow (pip install mlflow)
    
  * Machine Learning Libraries:
    
     * scikit-learn
       
     * XGBoost (pip install xgboost)
       
     * LightGBM (pip install lightgbm)
       
     * matplotlib, seaborn, pandas, numpy
       
  * MLflow Tracking Server running locally or on a remote server.

    
📂 **File Structure**

├── jewelley_price.py          # Main Streamlit application

├── requirements.txt           # List of dependencies

├── README.md                  # Documentation

📦 **Installation**

1. Clone this repository:

   git clone [(https://github.com/KennyOgun/jewellery_price_optimization.git]
   cd jewelry-price-optimization

2. Install the required Python packages:

   pip install -r requirements.txt

3. Start the MLflow Tracking Server (optional if not already running):

   mlflow ui --backend-store-uri sqlite:///mlflow.db

4. Run the Streamlit app:

   streamlit run jewelley_price.py
   
🏗️ **Usage**

**1. Upload Dataset**

  * The app requires a CSV dataset with columns such as Price_USD, Category, Target_Gender, Main_Color, Main_Metal, Main_Gem, and Brand_ID.
    
  * Use the Upload Dataset section in the app's sidebar to load your dataset.
    
**2. Navigate the App**

  * **Overview:** Understand the dataset and project scope.
    
  * **Data Analysis:** Visualize key insights like jewelry category distributions.
    
   * **Model Training and Evaluation:**
     
     * Select models in the sidebar.
       
     * Adjust the test size and train selected models.
       
     * View real-time MLflow metrics for the selected models.
       
**3. Access MLflow Tracking**

  * The app logs all model parameters, metrics, and artifacts to the MLflow server.
    
  * Navigate to the MLflow Tracking UI at http://localhost:5000 to explore detailed model performance and comparisons.

🛠️ **Troubleshooting**

**Common Errors**

  * **MLflow Server Unreachable:** Verify the MLflow tracking URI is correctly set in the script (http://localhost:5000 by default) and that the server is running.
    
  * **Dependencies Issues:** Ensure all libraries are installed using the requirements.txt.
    
**Logs**

Streamlit app logs can be viewed in the terminal. MLflow logs are available in the MLflow UI.
