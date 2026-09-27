# Week 2: Building Machine Learning Models

This project focuses on building, evaluating, and comparing machine learning models for **customer churn prediction**. The analysis uses the Telco Customer Churn dataset and explores baseline modeling, Logistic Regression, Decision Trees, Random Forest, class imbalance, threshold optimization, feature importance, and feature engineering.

## 📊 Model Results

| Model                          | Accuracy | Precision | Recall | F1-Score |   AUC |
| ------------------------------ | -------: | --------: | -----: | -------: | ----: |
| Baseline                       |    73.5% |     0.000 |  0.000 |    0.000 | 0.500 |
| Logistic Regression            |    80.7% |     0.658 |  0.567 |    0.609 | 0.842 |
| Logistic Regression (Balanced) |    73.9% |     0.505 |  0.781 |    0.613 | 0.841 |
| Decision Tree (Depth 5)        |    79.6% |     0.632 |  0.551 |    0.589 | 0.829 |
| Random Forest                  |    80.7% |     0.673 |  0.529 |    0.593 | 0.842 |

## 🎯 Baseline

The baseline model predicts every customer as **"Stay"**.

* Accuracy: **73.5%**
* Churn Recall: **0.0%**
* AUC: **0.500**

This demonstrates that accuracy alone can be misleading when the target classes are imbalanced.

## 📈 Logistic Regression

Logistic Regression achieved:

* Accuracy: **80.7%**
* AUC: **0.842**
* Precision: **65.8%**
* Recall: **56.7%**
* F1-Score: **60.9%**

The model provides useful churn discrimination while also offering interpretable coefficients and odds ratios.

## 🌳 Decision Tree

A depth analysis was performed to investigate model complexity and overfitting.

As tree depth increased, training accuracy continued to rise while test accuracy eventually decreased. For example:

* Depth 8: Training Accuracy = **83.5%**, Test Accuracy = **78.0%**
* Depth 15: Training Accuracy = **96.9%**, Test Accuracy = **73.7%**
* Unlimited Depth: Training Accuracy = **99.8%**, Test Accuracy = **74.2%**

This demonstrates how an overly complex Decision Tree can overfit the training data.

## 🌲 Random Forest

The Random Forest achieved:

* OOB Accuracy: **80.3%**
* Test Accuracy: **80.7%**
* AUC: **0.842**

The close relationship between OOB and test accuracy indicates reasonably consistent generalization on this dataset.

## ⚖️ Class Imbalance

Using `class_weight='balanced'` changed the model's behavior toward the minority churn class.

### Default Logistic Regression

* Precision: **65.8%**
* Recall: **56.7%**
* F1-Score: **60.9%**

### Balanced Logistic Regression

* Precision: **50.5%**
* Recall: **78.1%**
* F1-Score: **61.3%**

The balanced model substantially increased churn recall while reducing precision, illustrating the trade-off between identifying more churners and generating more false positives.

## 🎯 Threshold Optimization

The business cost assumptions were:

* False Negative Cost: **PKR 6,000**
* False Positive Cost: **PKR 1,000**

The theoretical decision threshold was approximately **0.14**, while the empirical cost analysis identified **0.15** as the minimum-cost threshold on the evaluated test set.

This demonstrates that the default classification threshold of 0.50 does not necessarily match the business objective.

## 🔍 Feature Importance

The top features according to Random Forest permutation importance were:

1. **tenure — 0.0404**
2. **TotalCharges — 0.0206**
3. **Contract_Two year — 0.0164**

Permutation importance measures the change in test-set performance when a feature's values are randomly shuffled.

## 🛠️ Feature Engineering

The following engineered features were created:

* `n_services`
* `is_new`
* `charge_per_mo`
* `price_jump`

Random Forest AUC before feature engineering:

**0.8422**

Random Forest AUC after feature engineering:

**0.8420**

The engineered features did not improve AUC on the evaluated test split, suggesting that much of the relevant information was already captured by the original features.

## 💡 Key Learning

The main lesson from this experiment is that **model accuracy alone is not sufficient for evaluating an imbalanced classification problem**.

A complete evaluation should consider:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* Model complexity and overfitting
* Feature importance
* Business costs
* Classification threshold

Threshold selection should be aligned with the relative cost of false positives and false negatives.

## 🧰 Tools & Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Kaggle Notebook
* Logistic Regression
* Decision Tree
* Random Forest

## 📁 Project

This notebook is part of my **Week 2 Machine Learning journey**, building on the exploratory data analysis performed in Week 1.
