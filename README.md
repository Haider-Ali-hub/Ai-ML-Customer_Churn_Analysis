Customer Churn Prediction

Week 1: Exploratory Data Analysis
Dataset

Source: Telco Customer Churn (Kaggle)

Size: 7,043 customers, 21 features

Target: Predict customer churn (Yes/No)

Key Findings
Contract Impact: Month-to-month contract holders have the highest churn rate (~42.7%), whereas 2-year contract holders show strong retention (>97%).

Tenure Influence: High churn is concentrated among new customers (tenure < 10 months). Churn risk decreases significantly as tenure grows (Correlation: -0.35).

High Charges & Services: Customers paying higher monthly charges ($70–$100+) and those subscribed to Fiber Optic internet (~41.8% churn rate) exhibit significantly higher drop-offs.

Payment Method Risk: Customers using Electronic check churn at a rate of ~45%, compared to ~15–16% for automated payment methods.

Setup
Open the Kaggle notebook or run locally:
pip install pandas numpy matplotlib seaborn

Bash
pip install pandas numpy matplotlib seaborn
