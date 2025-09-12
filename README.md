# Advanced House Price Prediction: A Regression Project with the Ames Dataset
# Project Overview
This project presents a complete, end-to-end machine learning workflow to solve a complex regression problem: predicting house sale prices in Ames, Iowa. The goal was not just to build an accurate model, but to demonstrate a robust, professional, and reproducible data science process.

The Ames Housing dataset was chosen for its real-world complexity, featuring 79 explanatory variables, a rich mix of data types, and significant missing values. This project showcases a systematic approach to data cleaning, feature engineering, and model optimization to turn this messy data into an accurate and interpretable predictive tool.

The primary success metric is the Root Mean Squared Error (RMSE), which measures the average prediction error in dollars.

# Key Features & Skills Demonstrated
**Deep Exploratory Data Analysis (EDA):** Identified and corrected a skewed target variable (SalePrice) using a log transformation and systematically analyzed the most influential numerical and categorical features.

**Intelligent Missing Value Imputation:** Implemented a multi-stage strategy based on domain knowledge, treating missing values differently based on their context (e.g., NaN in GarageType means 'No Garage').

**Advanced Feature Engineering:** Created new, high-impact features such as TotalSF (total square footage), HouseAge, and various binary "Has..." features to provide clearer signals to the model.

**Robust Preprocessing with Pipelines:** Utilized Scikit-learn's Pipeline and ColumnTransformer to build a professional, leak-proof workflow that handles data scaling and one-hot encoding consistently.

**Systematic Model Comparison & Tuning:** Trained and evaluated multiple models (from Linear Regression to XGBoost) using a "qualifying laps" approach. The top contenders were then optimized using GridSearchCV to find their best hyperparameters.

**Advanced Model Interpretation:** Went beyond performance metrics to explain the "why" behind the champion model's decisions using SHAP (SHapley Additive exPlanations) plots.

# Project Workflow
**Data Cleaning & EDA:** The project began with a thorough analysis of the target variable, SalePrice, correcting its positive skew with a log transformation. A deep dive into the 79 features using correlation analysis and visualizations identified key price drivers and revealed multicollinearity.

**Missing Value Imputation:** A comprehensive strategy was developed to handle the extensive missing data, distinguishing between values missing because a feature was absent (e.g., no pool) and those requiring statistical imputation (e.g., LotFrontage).

**Modeling & Tuning:** A baseline was established with a LinearRegression model. More complex models, including RandomForestRegressor and XGBRegressor, were then tested. After identifying these two as the top contenders, GridSearchCV was used to meticulously tune their hyperparameters, specifically to control for overfitting and minimize the cross-validated RMSE.

# Final Results & Champion Model
After a rigorous comparison of all baseline and tuned models, the Tuned XGBoost Regressor emerged as the definitive champion.

**Final Test RMSE:** $25,151.53

**Final Test R²:** 0.9093

This means the final model is able to predict house prices with an average error of approximately $25,150 and can explain about 91% of the variance in sale prices on unseen data.

# Model Performance Visualized
**Actual vs. Predicted Prices:** The plot shows a tight clustering of predictions around the line of perfect accuracy, visually confirming the model's high performance.


**Residuals Plot:** The residuals are randomly scattered around the zero-error line, indicating that the model's errors are unbiased and well-behaved.


**Key Insights:** What Drives House Prices in Ames?
The final, tuned XGBoost model was interpreted using its feature importance plot, revealing the most significant drivers of a house's value:

**Overall Quality (OverallQual):** The material and finish of the house is the single most important factor.

**Total Square Footage (TotalSF):** The overall size of the house, including the basement, is a critical predictor.

**Above Ground Living Area (GrLivArea):** The primary living space is a key component of the total size.

**Age of the House (HouseAge):** Newer houses consistently command higher prices.


# Technologies Used

Python

Pandas & NumPy for data manipulation

Matplotlib & Seaborn for data visualization

Scikit-learn for preprocessing, modeling, and evaluation

XGBoost for the final champion model

SHAP for advanced model interpretation
