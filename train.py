#!/usr/bin/env python3
"""
Customer Churn Prediction - Training Script
Author: Ashish Jha
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """Load and preprocess the data"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df = df.dropna()
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Churn':
            df[col] = le.fit_transform(df[col])
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = le.fit_transform(df['Churn'])
    
    return X, y

def engineer_features(X):
    """Feature engineering"""
    # Add custom features here
    return X

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models"""
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test)
        
        print(f"\n{name} Results:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred):.4f}")
        
        results[name] = {
            'model': model,
            'auc': roc_auc_score(y_test, y_pred)
        }
    
    return results

if __name__ == "__main__":
    # Load data (download from Kaggle: telco-customer-churn)
    X, y = load_and_preprocess_data('data/telco_churn.csv')
    
    # Feature engineering
    X = engineer_features(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train models
    results = train_models(X_train, y_train, X_test, y_test)
    
    # Save best model
    best_model_name = max(results, key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['model']
    
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print(f"\nBest model: {best_model_name}")
    print(f"AUC Score: {results[best_model_name]['auc']:.4f}")
    print("\nModels saved successfully!")
