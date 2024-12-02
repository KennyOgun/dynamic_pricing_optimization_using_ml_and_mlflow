#** Jewellery Price Optimization**

**Project Objective**
The goal is to develop an effective pricing strategy for jewellery by analyzing sales data and predicting optimal prices using machine learning. This involves:

  * Cleaning and preprocessing data to improve quality.
  * Identifying significant features that influence pricing.
  * Applying machine learning models to predict jewellery prices.
  * Providing insights to improve sales and profitability.

**Dataset Overview**
  * Source: Gemineye jewellery sales data.
  * Size: 95,910 rows with features such as category, price, target gender, main metal, main gem, and main color.
**Data Quality Issues:**
  * 16% of data (~15,452 rows) were corrupt and removed.
  * Presence of outliers (~804 rows), trimmed using the Isolation Forest algorithm.
  * Final dataset: 79,654 rows with 7 key features.

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
**2. Feature Selection:**
  * Retained critical features (e.g., price, category, metal, gem).
  * Dropped irrelevant columns (e.g., Order ID, SKU Quantity).
**3. Outlier Removal:**
  * Detected and removed outliers using the Isolation Forest algorithm.
**4. Feature Correlation:**
  * Identified relationships using Phik correlation to guide feature selection.
  * 
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
**MSE (Mean Squared Error):** Monitored for consistency.

**Performance Summary:**
**Best-performing model: Gradient Boosting Regressor.**
Close results from Random Forest, XGBoost, and LightGBM.

**Recommendations for Business**
**1. Invest in Data Quality and Expansion:*8

  * Address gaps in data such as customer demographics, purchasing patterns, and product details.
  * Collect data on marketing influences (e.g., ads, promotions) and seasonal trends.
  * 
**2. Improve Feature Engineering:**

  * Include variables like jewellery weight, purity levels, and certifications.
  * Analyze regional and seasonal demand patterns.
  * 
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

