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
  * 
📂 **File Structure**

├── jewelley_price.py          # Main Streamlit application
├── requirements.txt           # List of dependencies
├── README.md                  # Documentation

📦 **Installation**

1. Clone this repository:

   git clone https://github.com/YourGitHubUsername/jewelry-price-optimization.git
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





**Dataset Overview**

  * Source: Gemineye jewellery sales data.
  * Size: 95,910 rows with features such as category, price, target gender, main metal, main gem, and main color.
    
**Data Quality Issues:**

  * 16% of data (~15,452 rows) were corrupt and removed.
  * Presence of outliers (~804 rows), trimmed using the Isolation Forest algorithm.
  * Duplicates rows(2,589) were dropped.
  * Final dataset: 77,322 rows with 7 key features.

**Key Features Used**

**Category:** Type of jewellery (e.g., earrings, rings, pendants).

**Price (USD):** Target variable for optimization.

**Target Gender:** Primary buyer demographic (male/female).

**Main Color:** Color of the jewellery (e.g., red, yellow).

**Main Metal:** Material used (e.g., gold, silver).

**Main Gem:** Gem type (e.g., diamond, sapphire).

**Exploratory Data Analysis (EDA)**

**Price Distribution:**

  * Highly skewed with prices ranging from £30.99 to £34,448.60.
  * Mean price: £362.21; Standard deviation: £444.16.
    
**Buyer Demographics:**

  * 99.2% of buyers are female; males account for only 0.8%.
    
**Popular Items:**

  * Earrings, rings, and pendants account for 87% of sales.
  * Most common gem: **Diamond**.
  * Most common metal: **Gold**.

**Data Preprocessing**

**1. Cleaning:**
  * Removed corrupt and irrelevant data.
  * Handled missing values using simple imputation (mean for numerical, mode for categorical).
  * Dropped Duplicates rows.
    
**2. Feature Selection:**
  * Retained critical features (e.g., price, category, metal, gem).
  * Dropped irrelevant columns (e.g., Order ID, SKU Quantity).
   
**3. Outlier Removal:**
  * Detected and removed outliers using the Isolation Forest algorithm.
    
**4. Feature Correlation:**
  * Identified relationships using Phik correlation to guide feature selection.
   
**Modeling and Evaluation**

**Models Used:**

**1. Random Forest Regressor**

**2. XGBoost Regressor**

**3. Gradient Boosting Regressor**

**4. LightGBM Regressor**
    
**Hyperparameter Tuning:**

  * Performed using grid search.
  * MLflow was utilized to track experiments, hyperparameters, and metrics.
   
**Evaluation Metrics:**

**R² (Goodness of Fit):** ~0.29 across models.

**MAE (Mean Absolute Error):** Ranges between ~152 and ~153.

**MSE (Mean Squared Error):** Ranges between ~53104 and ~53333.

**Performance Summary:**

**Best-performing model: XGBoost and Gradient Boosting Regressor.**
Close results from Random Forest, and LightGBM.

**Recommendations for Business**

**1. Invest in Data Quality and Expansion:**

  * Address gaps in data such as customer demographics, purchasing patterns, and product details.
  * Collect data on marketing influences (e.g., ads, promotions) and seasonal trends.
    
**2. Improve Feature Engineering:**

  * Include variables like jewellery weight, purity levels, and certifications.
  * Analyze regional and seasonal demand patterns.
   
**3. Leverage Insights for Strategy:**
    
  * Focus on female buyers and popular items (earrings, rings, pendants).
  * Develop campaigns to target male buyers in niche categories like rings.
  * Increase inventory of high-demand items (e.g., gold jewellery with diamonds).
    
**Future Work**

  * Incorporate additional data sources for better feature diversity.
  * Explore advanced deep learning models for price prediction.
  * Perform market segmentation for targeted pricing strategies.

**How to Run the Project**

  * Clone the repository.
  * Install dependencies from requirements.txt.
  * Run the Jupyter Notebook or the Python scripts to preprocess data and train models.
  * Use MLflow for model tracking and experiment monitoring.
    
**Technologies Used**

  * Programming: Python (Pandas, NumPy, Scikit-learn, XGBoost, LightGBM)
  * Visualization: Matplotlib, Seaborn, Plotly
  * Experiment Tracking: MLflow
  * Outlier Detection: Isolation Forest Algorithm

**Conclusion**

This project demonstrates how machine learning can be used for price optimization in the jewellery industry. By addressing data quality issues, applying advanced regression models, and leveraging insights, businesses can make informed pricing decisions and maximize profitability.

