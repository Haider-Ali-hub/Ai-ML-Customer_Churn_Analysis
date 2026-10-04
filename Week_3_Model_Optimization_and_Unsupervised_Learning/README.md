## Week 3: Model Optimization and Unsupervised Learning

This week focused on replacing single train/test split estimates with more reliable cross-validation, hyperparameter tuning, XGBoost optimization, customer segmentation, and PCA.

### Key Results

* **Split-to-split accuracy range:** 78.0% to 82.8% across 20 random seeds
* **Split standard deviation:** 0.0104
* **5-fold CV AUC — Logistic Regression:** 0.846 ± 0.013
* **5-fold CV AUC — Random Forest:** 0.844 ± 0.011
* **Tuned Logistic Regression:** best C = 10.0
* **Random Forest Grid Search:** best AUC = 0.8468
* **Random Forest Random Search:** best AUC = 0.8464
* **Tuned XGBoost CV AUC:** 0.8502 ± 0.0117
* **Final model:** Tuned XGBoost
* **Final test AUC:** 0.8483
* **Final test recall:** 0.521
* **Final test precision:** 0.659
* **Customer segments:** k = 4
* **Highest-risk segment:** High-Value At-Risk Customers, with 43% churn
* **Lowest-risk segment:** Stable Low-Cost Customers, with 5% churn
* **PCA:** 15 of 30 components explain 90% of the variance

### Biggest Lesson

A single accuracy or AUC value can be misleading because model performance changes with the data split. Cross-validation provides a more reliable estimate of model performance and uncertainty, while the untouched test set should be used only once for final evaluation.

