import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc


# Load dataset
df= pd.read_csv("C:/Users/prana/OneDrive/Desktop/loan_data.csv")

# Encode target variable
from sklearn.preprocessing import OrdinalEncoder
oe= OrdinalEncoder()
for col in ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']:
    df[col]= oe.fit_transform(df[[col]])

# Normalize the feature
from sklearn.preprocessing import StandardScaler
sk= StandardScaler()
for col in ['person_income', 'loan_amnt']:
    df[col]= sk.fit_transform(df[[col]])

# Split into features and target
y= df['loan_status']
X= df.drop('loan_status', axis=1)

# Split the data into training(80%) and testing(20%) sets
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size= 0.2, random_state=42)

# Handle class imbalance using SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)



# RandomForest Classifier Model
rf= RandomForestClassifier()
rf.fit(X_train, y_train)
y_pr= rf.predict(X_test)

# Accuracy using RandomForest Classifier Model
accuracy_score(y_test, y_pr)*100



# XGBoost Classifier Model
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2], 'subsample': [0.7, 1] }
# Grid search with 3-fold CV
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=2)
grid_search.fit(X_train_res, y_train_res)
# Best model from grid search
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)

# Accuracy using XGBoost Classifier Model
accuracy_score(y_test, y_pred)*100



#Feature Importance Comparison: Random Forest vs XGBoost 
# Get feature importances from both models
rf_importance = rf.feature_importances_
xgb_importance = best_xgb.feature_importances_

# Create a DataFrame for both
feature_names = X.columns
comparison_df = pd.DataFrame({
    'Feature': feature_names,
    'Random Forest': rf_importance,
    'XGBoost': xgb_importance
})
# Sort by average importance for better visualization
comparison_df['Mean'] = comparison_df[['Random Forest', 'XGBoost']].mean(axis=1)
comparison_df = comparison_df.sort_values(by='Mean', ascending=True)

# Plot
plt.figure(figsize=(10, 8))
bar_width = 0.4
y = np.arange(len(comparison_df))
plt.barh(y, comparison_df['Random Forest'], height=bar_width, label='Random Forest', color='skyblue')
plt.barh(y + bar_width, comparison_df['XGBoost'], height=bar_width, label='XGBoost', color='salmon')
plt.yticks(y + bar_width / 2, comparison_df['Feature'])

# Formatting
plt.xlabel('Feature Importance Score')
plt.title('Feature Importance Comparison: Random Forest vs XGBoost')
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



#ROC Curve Comparison: Random Forest vs XGBoost
# 1. Predict probabilities
rf_probs = rf.predict_proba(X_test)[:, 1]
xgb_probs = best_xgb.predict_proba(X_test)[:, 1]

# 2. Get ROC curve and AUC for both models
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
auc_rf = auc(fpr_rf, tpr_rf)
auc_xgb = auc(fpr_xgb, tpr_xgb)

# 3. Print AUC Scores
print("ROC-AUC Score for Random Forest: {:.4f}".format(auc_rf))
print("ROC-AUC Score for XGBoost: {:.4f}".format(auc_xgb))

# 4. Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = {:.2f})'.format(auc_rf), color='skyblue')
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost (AUC = {:.2f})'.format(auc_xgb), color='salmon')

# Reference line
plt.plot([0, 1], [0, 1], 'k--', lw=1)

# Formatting
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()