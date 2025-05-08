# Dynamic Price Optimization Using ML and Mlflow

![image](https://github.com/user-attachments/assets/82ccb2ea-2ebc-43f5-b73e-f2444f92cc06)

# Project Overview
The Dynamic Price Optimization Using ML and MLflow project focuses on building machine learning models to optimize pricing strategies while leveraging MLflow for model tracking and deployment. It involves data analysis, feature engineering, and understanding customer pricing behavior, with a strong emphasis on integrating pricing models into business workflows for actionable insights.

# Business Introduction

The focal company is a luxury jewelry retailer known for its craftsmanship, quality, and innovation. With a global presence, the brand aims to cater to a diverse clientele. The company has consistently leveraged technology to enhance customer experiences, offering online customization and seamless e-commerce options. However, its pricing strategies currently rely on manual adjustments, leading to inefficiencies in capturing optimal revenue.

 Achievements:
  - Leading in bespoke jewelry sales.
  - Recognition for excellent customer service.
    
Unique Aspects:
  - Seasonal pricing variations.
  - Customization options influencing price variability.

# Business Problem

1. Overpricing risks losing price-sensitive customers.
2. Under-pricing reduces profit margins.
3. Lack of dynamic adjustments based on market trends, preferences, and competition.
4. Inconsistent pricing strategies across regions and product lines.
5. Absence of data-driven demand prediction.

# Project Objective

  * Maximized Revenue: Data-driven pricing ensures optimal revenue generation by balancing volume and margins.
  * Competitive Edge: Dynamic pricing enables quick adaptation to market changes.
  * Improved Customer Retention: Personalized pricing strategies cater to different customer segments.
  * Efficient Decision-Making: Automating pricing decisions reduces reliance on manual intervention.
  * Actionable Insights: An ML-driven approach provides deeper insights into customer behavior and demand patterns.

# Data Description

The features contained in the dataset are:
1. Order datetime: The time at which the order was placed.
2. Order ID: Identifiers for the different orders placed.
3. Purchased product ID: Identifiers for the different product ordered for.
4. Quantity of SKU in the order: Quantity of jewelry pieces ordered for.
5. Category ID: Identifier for the jewelry category.
6. Category alias: Name of jewelry category e.g. earring.
7. Brand ID: Identifier for jeweler brand
8. Price in USD: Jewelry price in US Dollars- target variable for optimization.
9. User ID: Identifier for user/customer
10. Product gender: (for male/female) 
11. Main Color: Overall color of jewelry piece -e.g., red, yellow.
12. Main metal: Main metal used for mounting- e.g., gold, silver
13. Main gem: Main gem mounted on jewelry piece e.g. diamond, sapphire.

# Data Preprocessing 

1. Missing Values: About nine(9) features had missing values varying from 5% - 50%(Target_Gender (50%) and Main_Gem (35%).
2. Duplicates: The data had 2.6%  duplicate rows.
3. Feature Variety: Low variety in SKU_Quantity, Main_Color, and Main_Metal.
4. Outliers in Price: Price ranges from $0.99 to $34,448.60, indicating possible outliers or premium items.
5. Data Quality Issues: Incorrect values in Category (e.g., '451.10', '283.49') need correction.

## Next Steps for Data Preparation
1. Data Exploration 
2. Address missing values and duplicates
3. Correct inconsistent or corrupt entries(about 16%).
4. Handle outliers in the price feature.
5. Normalize categorical and numerical features for modelling.
6. Retained critical features like price, category, metal, gem and dropped irrelevant columns e.g., Order ID, SKU Quantity

# Exploratory Data Analysis (EDA)

1. Jewelry Category Distribution

 <img width="597" alt="image" src="https://github.com/user-attachments/assets/0fcfdc0d-9e89-46ca-b8c2-59dce7ccd596" />

Earrings, rings, and pendants account for 87% of sales.

2. Buyer Demographics

<img width="515" alt="image" src="https://github.com/user-attachments/assets/2c8acb96-8819-4828-91b8-a19341aaa749" />


The majority of their customers are 99.2% female and males account for only 0.8%.

# Feature Engineering, Selection, and Dimensionality Reduction

1. Correlation Matrix Heatmap 

<img width="431" alt="image" src="https://github.com/user-attachments/assets/c2ad3f89-5451-46bd-83f9-9276cadcccac" />

The heatmap shows the following  associations of Price;

   * Moderate: with Main_Metal(0.40) and Category(0.31),
   * low: with  Main_Metal(0.19) and Main_Colour(0.22)
   * extemely low: with Brand_ID(0.11) and Target_Gender (0.06).

# Model Selection, Training, Evaluation and MLflow set-up 

1. **Models Chosen(ML):**
   
Selected for their strong performance in regression tasks and ability to handle nonlinear relationships.
These models are known for their robustness, scalability, and hyperparameter tuning flexibility.

<img width="724" alt="image" src="https://github.com/user-attachments/assets/858a5e21-d95d-4850-8a5b-8efbd8a3808f" />


     
2. **Model Training:** Hyperparameter tuning using GridSearchCV with 5-fold cross-validation was applied to ML models.
 
  * Best parameters identified:
   
     * RandomForestRegressor: max_depth=30, min_samples_split=2, n_estimators=200.
     * XGBRegressor: learning_rate=0.2, max_depth=10, n_estimators=300.
     * GradientBoostingRegressor: learning_rate=0.2, max_depth=7, n_estimators=300.
     * LGBMRegressor: learning_rate=0.3, num_leaves=31, n_estimators=300.


 3. **MLflow Model Tracking Experiment Snapshot:**

<img width="742" alt="image" src="https://github.com/user-attachments/assets/0c1fb135-e8c3-4b0c-bbb5-f26c92cbea92" />


 * Inspection of the models using MLflow tracking reveals the following key patterns:
  * The number of estimators ranges from 100 to 300 across all models.
  * The maximum depth varies between 7 and 31.


# Evaluation Metrics and Results 

The metrics align with regression task objectives and provide a holistic evaluation.

1. R² (Coefficient of Determination): Measures goodness of fit.
2. MAE (Mean Absolute Error): Evaluates average prediction error.
3. MSE (Mean Squared Error): Penalizes larger errors to assess overall variance.

<img width="308" alt="image" src="https://github.com/user-attachments/assets/a64fb25a-1c04-4cd9-8b62-7e725c8ffe36" />


These similarities in hyperparameters explain why the models deliver closely matching results:
    * R² (Goodness of Fit): ~0.29 for all models.
    * MAE (Mean Absolute Error) and MSE(Mean Squared Error): Ranges between ~152 and ~153 and ~53104 and ~53333 respectively,  indicating similar average 
      prediction errors.

**Results:**

XXGBoost & Gradient Boosting Regressor with the highest R² (0.2982), other models had comparable but slightly lower R² scores (~0.29).

The results indicate limited predictive power, suggesting that the current dataset lacks sufficient features and data quality to model jewellery prices effectively.

# Recommendations:

1. **Invest in Data Quality and Expansion**
   
   The dataset faced a 20% reduction due to errors. It also highlight significant limitations such as:
   
     * Lack of diverse and detailed features (e.g. customer demographics, promotional campaigns, and purchasing patterns).
     * Insufficient data samples for certain categories (e.g., male buyers and less common gems/metals).
     * Jewellery-specific attributes: Weight, purity levels, and certifications.
     * Marketing data: Ad campaigns, promotions, and customer engagement metrics.
     * External factors: Seasonality and regional trends impacting demand.

* Action Plan:
  
    * Expand Customer Data: Collect information on age, location, and income level.
    * Enrich Product Features: Include design specifications, customization options, and seasonal trends.
    * Enhance Sales History: Track discounts, bundling strategies, and cross-sale patterns..

2. Leverage Insights for Business Strategy: Utilize data-driven insights to refine business operations;
   
   * Target Female Customers: Focus on earrings, rings, and pendants, which dominate sales.
   * Optimize Inventory and Promotions: Prioritize popular items like gold jewellery with diamonds.
   * Engage Male Buyers: Develop targeted campaigns to drive purchases in niche categories like rings.

In conclusion, investing in data quality, expanding feature diversity, and leveraging insights for strategic decision-making will empower the company to optimize pricing strategies, improve operational efficiency, and enhance profitability.




################




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

