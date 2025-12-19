#!/usr/bin/env python3
"""
Quick Model Training Script
Trains the essential models needed for the Flask app
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import bz2
import shap
from lifelines import CoxPHFitter
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("QUICK MODEL TRAINING - CUSTOMER CHURN PREDICTION")
print("="*80)

# Load dataset
print("\n[1/4] Loading dataset...")
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
print(f"✓ Loaded {len(df):,} rows")

# Prepare data for churn model
print("\n[2/4] Training Random Forest Churn Model...")
df_model = df.drop(['customerID'], axis=1)
df_model['Churn'] = df_model['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical features
categorical_cols = df_model.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
df_encoded = df_model.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Split data
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train Random Forest with hyperparameter tuning
print("  - Performing grid search...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', verbose=0, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

print(f"✓ Best Random Forest trained")
print(f"  - Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"  - Precision: {precision_score(y_test, y_pred):.4f}")
print(f"  - Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"  - F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"  - ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)
print("✓ Saved model.pkl")

# Create and save SHAP explainer
print("\n[3/4] Creating SHAP explainer...")
explainer = shap.TreeExplainer(best_rf)
with bz2.BZ2File('explainer.bz2', 'w') as f:
    pickle.dump(explainer, f)
print("✓ Saved explainer.bz2")

# Train Cox Proportional Hazards Model
print("\n[4/4] Training Cox Proportional Hazards Model...")
df_cox = df.copy()
df_cox['SeniorCitizen'] = df_cox['SeniorCitizen'].astype(str)
df_cox['duration'] = df_cox['tenure']
df_cox['event'] = (df_cox['Churn'] == 'Yes').astype(int)

# Encode for Cox model
binary_vars = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for var in binary_vars:
    df_cox[var] = (df_cox[var] == df_cox[var].unique()[0]).astype(int)

# One-hot encoding for multi-category variables
categorical_features = ['SeniorCitizen', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
df_cox = pd.get_dummies(df_cox, columns=categorical_features, drop_first=True)

# Select features for Cox model
feature_cols = [col for col in df_cox.columns if col not in ['customerID', 'Churn', 'duration', 'event', 'tenure']]
cox_df = df_cox[feature_cols + ['duration', 'event']].copy()

# Fit Cox model
try:
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col='duration', event_col='event', show_progress=False)
    
    with open('survivemodel.pkl', 'wb') as f:
        pickle.dump(cph, f)
    print(f"✓ Cox model trained (Concordance Index: {cph.concordance_index_:.4f})")
    print("✓ Saved survivemodel.pkl")
except Exception as e:
    print(f"⚠️  Cox model training failed: {str(e)}")
    print("  Creating dummy survival model...")
    # Create a minimal dummy Cox model
    simple_cox = cox_df[['MonthlyCharges', 'TotalCharges', 'duration', 'event']].copy()
    cph = CoxPHFitter()
    cph.fit(simple_cox, duration_col='duration', event_col='event', show_progress=False)
    with open('survivemodel.pkl', 'wb') as f:
        pickle.dump(cph, f)
    print("✓ Saved survivemodel.pkl (simplified version)")

# Save preprocessing artifacts
artifacts = {
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_names': X.columns.tolist(),
    'numerical_cols': numerical_cols,
    'categorical_cols': categorical_cols
}

with open('preprocessing_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)
print("✓ Saved preprocessing_artifacts.pkl")

print("\n" + "="*80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nFiles created:")
print("  ✓ model.pkl (Random Forest classifier)")
print("  ✓ survivemodel.pkl (Cox Proportional Hazards model)")
print("  ✓ explainer.bz2 (SHAP explainer)")
print("  ✓ preprocessing_artifacts.pkl (scalers and encoders)")
print("\nThe Flask app is ready to run with real trained models!")
print("\nTo start the app:")
print("  ./run.sh")
print("  or")
print("  python app.py")
