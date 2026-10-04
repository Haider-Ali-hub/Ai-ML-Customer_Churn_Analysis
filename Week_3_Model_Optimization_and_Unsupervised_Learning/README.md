## Week 3: Model Optimization and Unsupervised Learning

## Executive Summary & Key Highlights
This repository documents Week 3 of the ML-Powered Customer Analytics project using the **Telco Customer Churn** dataset (7,043 rows, 30 encoded features). The core focus of this lab is addressing data leakage and random variance, executing hyperparameter optimization across multiple models, applying unsupervised clustering and dimensionality reduction, and evaluating final model generalization on an untouched test set.

* **Split Noise Measurement:** Evaluating a single model across 20 random train/validation splits revealed a **4.8% accuracy variance** ($0.780$ to $0.828$), proving that single train/validation split comparisons are unreliable due to random sampling noise.
* **Stratified 5-Fold Cross-Validation:** Built leak-free pipelines using `StratifiedKFold`. Baseline Logistic Regression (AUC $0.846 \pm 0.013$) and Random Forest (AUC $0.844 \pm 0.011$) were statistically tied within standard error limits.
* **Hyperparameter Tuning:** Compared Grid Search and Random Search for Random Forest and XGBoost. Random Search achieved equivalent AUC ($0.8464$) in fewer computation fits compared to Grid Search.
* **XGBoost Optimization:** Implemented early stopping (stopping at tree $247$, validation AUC $0.8541$) to prevent overfitting.
* **Customer Segmentation (K-Means):** Built $k=4$ distinct customer personas based on tenure, spend, and service usage. Identified a high-risk group (**Mid-tenure, high spend**) exhibiting a **43% churn rate**.
* **Dimensionality Reduction (PCA):** Demonstrated that **15 out of 30 principal components** capture 90% of total variance. Identified perfect multicollinearity in dummy dummy columns (`No internet service` redundancy).
* **Final Holdout Evaluation:** Tuned XGBoost selected via CV was tested once on the locked 20% holdout set ($1,409$ rows), yielding an honest Test AUC of **0.8483**.

---

## 1. Split Noise Analysis & Test Set Locking 

### Test Set Preservation
To ensure zero data leakage and unbiased generalization evaluation, $20\%$ of the dataset ($1,409$ rows) was stratified and locked away until Part 7. All preprocessing (imputation of blank `TotalCharges` with 0, one-hot encoding) was conducted without statistical aggregation to prevent feature leakage.

### Measurement of Split Noise (Task 1.3)
To measure the random variance introduced by single data splits, an unregularized Logistic Regression pipeline was executed across **20 distinct random seeds** ($random\_state \in [0, 19]$) on the training set.

* **Observed Accuracy Range:** $0.780$ to $0.828$ (Spread: $0.048$ / $4.8\%$)
* **Observed Standard Deviation ($\sigma$):** $0.0104$
* **Theoretical Standard Error ($SE = \sqrt{p(1-p)/n}$):** $0.0107$ ($p \approx 0.803, n = 1127$)
* **95% Confidence Interval:** $\pm 2.1\%$ ($\pm 0.021$)

**Key Takeaway:** Model performance fluctuates by up to 4.8 percentage points purely due to row assignment chance. Because model architecture differences between Logistic Regression and Random Forest are often $< 1.0\%$, single-split comparisons can easily lead to false conclusions. Cross-validation reporting ($mean \pm std$) is strictly required.

---

## 2. 5-Fold Cross-Validation & Model Baselines 

A leak-free 5-fold cross-validation pipeline was constructed using `StratifiedKFold`. Standard scaling was embedded within sklearn `Pipeline` objects to ensure scaling statistics were calculated exclusively on training folds.

### Cross-Validation Results

| Model Pipeline | ROC-AUC (Mean ± Std) | Recall (Mean ± Std) | F1-Score (Mean ± Std) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | **0.846 ± 0.013** | **0.545 ± 0.042** | **0.594 ± 0.030** |
| **Random Forest** | 0.844 ± 0.011 | 0.496 ± 0.019 | 0.573 ± 0.020 |

**Insight:**  
1. **Statistical Tie:** The ROC-AUC difference ($0.002$) is roughly 6 times smaller than the cross-validation standard deviation ($\approx 0.012$), confirming both models perform identically on ranking quality.
2. **Threshold Gap:** At default $0.5$ classification threshold, Logistic Regression achieves higher Recall ($54.5\%$) than Random Forest ($49.6\%$). Both models miss nearly half of actual churners, highlighting the necessity for threshold tuning or class weighting.

---

## 3. Hyperparameter Tuning 

### Validation Curve for Logistic Regression ($C$)
Systematically evaluated $13$ values of inverse regularization strength $C \in [10^{-4}, 10^2]$ on log scale.
* **Underfitting Region:** Very small values ($C < 0.01$) penalize coefficients excessively, lowering both Train and CV AUC.
* **Optimal Region:** $C = 10.0$ achieved highest CV AUC ($0.8464$), though performance plateaus between $C=1.0$ and $C=10.0$. Minimal variance between Train and CV curves indicates absence of overfitting.

### Grid Search vs. Random Search (Random Forest)
* **Grid Search:** Evaluated $24$ hyperparameter combinations ($4 \times 3 \times 2$) over 5 folds ($120$ total fits).  
  * *Best Params:* `max_depth=8`, `max_features='sqrt'`, `min_samples_leaf=20`
  * *Result:* Best CV AUC = **0.8468** (Execution Time: 69s)
* **Random Search:** Sampled $24$ random configurations from continuous ranges over 5 folds ($120$ total fits).  
  * *Best Params:* `max_depth=15`, `max_features≈0.21`, `min_samples_leaf=15`
  * *Result:* Best CV AUC = **0.8464** (Execution Time: 74s)

**Insight:** Score difference between Grid and Random search ($0.0004$) is far below the CV standard deviation ($\approx 0.011$). Random Search is superior for higher-dimensional spaces as it samples distinct values across influential parameters without exponential grid growth.

---

## 4. XGBoost Optimization & Early Stopping 

### Early Stopping Dynamics (Task 4.1 & 4.2)
Trained XGBoost with up to $2,000$ trees, `learning_rate=0.03`, and `early_stopping_rounds=100` on an $80/20$ inner split using `scale_pos_weight=2.77`.
* **Optimal Stopping Point:** Iteration **247** (Validation Log-Loss minimum reached).
* **Validation AUC:** **0.8541**
* **Overfitting Analysis:** Post round 247, training loss continued decreasing while validation loss began rising. Early stopping successfully truncated training before noise memorization occurred.

### Random Search Hyperparameter Tuning (Task 4.3)
Executed 30 random configurations across 5 folds ($150$ fits) tuning `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, and `reg_lambda`.
* **Optimal Hyperparameters:** `max_depth=2`, `learning_rate≈0.034`, `n_estimators=476`, `subsample=0.6`, `colsample_bytree≈0.561`, `min_child_weight=1`, `reg_lambda≈1.97`.
* **Tuned CV AUC:** **0.8502 ± 0.0117**

**Insight:** The optimal `max_depth=2` indicates shallow, decision-stump boosting trees perform best. This confirms the underlying churn signal is largely additive, explaining why linear Logistic Regression remains competitive.

---

## 5. K-Means Customer Segmentation 

Customer segmentation was performed on 4 numerical features: `tenure`, `MonthlyCharges`, `TotalCharges`, and `n_services`. Features were standardized using `StandardScaler` to prevent `TotalCharges` ($\sigma \approx 2267$) from dominating low-scale features like `n_services` ($\sigma \approx 1.8$).

### Segment Profiling Table ($k = 4$)

| Segment Name | Cluster | Size | Avg Tenure | Avg Spend | Avg Services | Churn Rate | Strategic Retention Action |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mid-tenure, High Spend** | Cluster 1 | 2,157 | 18.4 mos | $80.41 | 3.3 | **43%** | Offer 1-2 year contract price locks & free security/tech-support add-ons. |
| **New, Low Spend** | Cluster 3 | 1,918 | 9.0 mos | $37.71 | 1.2 | **32%** | Implement 90-day onboarding check-ins & first-year discount incentives. |
| **Loyal, Fully Bundled** | Cluster 2 | 1,938 | 59.8 mos | $92.09 | 5.1 | **14%** | Reward with VIP loyalty perks and priority support; avoid over-discounting. |
| **Long-tenure, Minimal** | Cluster 0 | 1,030 | 53.6 mos | $30.96 | 1.5 | **5%** | Maintain low-touch stable service; soft cross-sell economical add-ons. |

**Key Business Insight:** **Cluster 1 (Mid-tenure, High Spend)** represents the primary churn risk driver, accounting for roughly half of total churners (~930 out of ~1,870). These customers pay high monthly charges without long-term contracts or full service bundling.

---

## 6. Principal Component Analysis 

### Scree Plot & Variance Analysis
PCA was applied to the 30 standardized training features:
* **Dimensionality Reduction:** **15 out of 30 components** explain **90%** of total cumulative variance, showing 50% feature redundancy.
* **PC1 Feature Loadings:** Identical highest loadings ($0.302$) were assigned to `InternetService_No` and six dummy features representing `No internet service`.
* **Multicollinearity Discovery:** Dummy feature creation produces identical zero-variance dummy duplicates for non-internet users, causing severe multicollinearity in unregularized linear models.

### 2D Visualization Analysis
Plotting PC1 vs PC2 revealed extensive spatial overlap between churned and non-churned customers. This confirms churn boundary non-linearity and explains the empirical performance ceiling of $\approx 0.85$ ROC-AUC across all model families.

---

## 7. Model Selection & Holdout Evaluation 

### Model Comparison Summary

| Tuned Model | 5-Fold CV AUC | CV Std | Model Selected |
| :--- | :---: | :---: | :---: |
| Logistic Regression ($C=10.0$) | 0.8464 | 0.0129 | No |
| Random Forest (Tuned) | 0.8464 | 0.0114 | No |
| **XGBoost (Tuned)** | **0.8502** | **0.0117** | **YES** |

### Single Final Test Evaluation
The winning tuned XGBoost model was refit on the complete training set ($5,634$ rows) and evaluated **exactly once** on the locked test set ($1,409$ rows).

* **Test ROC-AUC:** **0.8483**
* **Test Recall (0.5 threshold):** **0.521**
* **Test Precision (0.5 threshold):** **0.659**

**Honest Assessment:** The holdout Test AUC ($0.8483$) sits comfortably within the 95% CV confidence interval ($0.8502 \pm 0.0234 \rightarrow [0.8268, 0.8736]$), validating cross-validation honesty. The performance gain of complex XGBoost over plain Logistic Regression is marginal ($+0.0038$), proving feature engineering and decision threshold tuning are more vital than model complexity.

---

**Repository Maintained by:** Haider Ali Asghar  
**Contact / Profile:** [GitHub Profile](https://github.com/Haider-Ali-hub)
