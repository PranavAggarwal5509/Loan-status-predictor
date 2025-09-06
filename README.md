# Loan-status-predictor
💰 Loan Approval Classification using Ensemble Models  
This project presents a machine learning approach to classify loan approval status based on applicant data. It compares two powerful classifiers: **Random Forest** and **XGBoost**, while handling real-world issues like class imbalance using **SMOTE** and optimizing performance through **GridSearchCV**.

📌 **Problem Statement**  
Given an applicant dataset with demographic and financial features, predict whether a loan should be approved (`loan_status` = 1) or not (`loan_status` = 0).

🧠 **Modeling Approach**

Two ensemble learning models:
- **Random Forest Classifier**: A bagging-based tree ensemble.
- **XGBoost Classifier**: A boosting-based gradient ensemble tuned via GridSearch.

Both models are trained and compared using accuracy and ROC-AUC metrics.

⚙️ **Workflow**

**Data Preprocessing**
- Encoded categorical variables using `OrdinalEncoder`
- Normalized numerical features with `StandardScaler`
- Dropped/handled missing values if any

**Class Imbalance Handling**
- Applied **SMOTE** to balance target classes in training set

**Model Training**
- `RandomForestClassifier`: default settings
- `XGBClassifier`: tuned with `GridSearchCV` on:
  - `n_estimators`, `max_depth`, `learning_rate`, `subsample`

**Evaluation**
- Accuracy
- Feature Importance Comparison
- ROC Curve & AUC Score

**Visualizations**
- Feature Importance Bar Chart (RF vs XGBoost)
- ROC Curve Comparison

📊 **Results**  
_Test metrics (example — update with actual values):_

- **Random Forest Accuracy**: 92.73%
- **XGBoost Accuracy**: 91.49%
- **ROC-AUC RF**: ~0.973
- **ROC-AUC XGBoost**: ~0.9689

**Visualizations:**
- Feature Importance (horizontal bar plot)
- ROC Curve (line plot with AUC)

📁 **Dataset**

- Format: `.csv` file
- Features:
  - `person_income`, `loan_amnt`, `person_home_ownership`, `person_education`, etc.
- Target: `loan_status` (binary)

🛠️ **Libraries Used**

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`, `imblearn`


📌 **Key Takeaways**

- Handled imbalanced data effectively using **SMOTE**
- Compared bagging and boosting models side by side
- Performed model tuning with **GridSearchCV**
- Used **feature importance and ROC-AUC** for evaluation
- Created **clear visualizations** to interpret results

---
