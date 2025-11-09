# Customer Churn Prediction

## Overview
Production-ready machine learning project for predicting customer churn using advanced classification algorithms including Logistic Regression, Random Forest, and XGBoost. This project demonstrates end-to-end ML pipeline development with proper data preprocessing, feature engineering, model training, evaluation, and deployment capabilities.

## Features
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature Engineering**: Advanced feature creation and selection
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and AUC-ROC
- **Hyperparameter Tuning**: GridSearchCV for optimal model performance
- **Model Persistence**: Save and load trained models
- **Production Ready**: Modular code structure with configuration management

## Performance
- **Recall Improvement**: 12% increase through feature engineering and SMOTE
- **Best Model**: XGBoost with optimized hyperparameters
- **Validation Strategy**: Stratified K-Fold Cross-Validation

## Dataset
Using Telco Customer Churn dataset from Kaggle
- **Source**: https://www.kaggle.com/blastchar/telco-customer-churn
- **Samples**: 7,043 customers
- **Features**: 20 features including demographics, services, and account information
- **Target**: Churn (Yes/No)

## Project Structure
```
customer-churn-prediction/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned and preprocessed data
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── data_preprocessing.py   # Data cleaning and preprocessing
│   ├── feature_engineering.py  # Feature creation
│   ├── model_training.py       # Model training pipeline
│   ├── model_evaluation.py     # Model evaluation utilities
│   └── predict.py              # Inference script
├── models/                     # Saved trained models
├── config/
│   └── config.yaml            # Configuration parameters
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

```bash
# Clone the repository
git clone https://github.com/ASHISH311001/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements
- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- imbalanced-learn >= 0.9.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- joblib >= 1.1.0

## Usage

### 1. Data Preprocessing
```python
from src.data_preprocessing import preprocess_data

# Load and preprocess data
X_train, X_test, y_train, y_test = preprocess_data('data/raw/telco_churn.csv')
```

### 2. Feature Engineering
```python
from src.feature_engineering import engineer_features

# Create advanced features
X_train_eng = engineer_features(X_train)
X_test_eng = engineer_features(X_test)
```

### 3. Train Models
```python
from src.model_training import train_all_models

# Train and compare multiple models
models, results = train_all_models(X_train_eng, y_train, use_smote=True)
```

### 4. Make Predictions
```python
from src.predict import load_model, predict_churn

# Load trained model and make predictions
model = load_model('models/xgboost_model.pkl')
predictions = predict_churn(model, X_test_eng)
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|----------|
| Logistic Regression | 0.78 | 0.72 | 0.65 | 0.68 | 0.82 |
| Random Forest | 0.82 | 0.78 | 0.73 | 0.75 | 0.88 |
| **XGBoost** | **0.85** | **0.82** | **0.77** | **0.79** | **0.91** |

## Key Features Engineered
1. **Tenure Groups**: Categorization of customer tenure
2. **Total Charges per Month**: Ratio feature
3. **Service Adoption Rate**: Number of services used
4. **Contract Value**: Interaction between contract type and charges
5. **Customer Lifetime Value**: Estimated CLV

## Hyperparameters (XGBoost)
```python
{
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'min_child_weight': 1,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

## Future Improvements
- [ ] Implement deep learning models (Neural Networks)
- [ ] Add real-time prediction API using Flask/FastAPI
- [ ] Create interactive dashboard with Streamlit
- [ ] Implement A/B testing framework
- [ ] Add model monitoring and drift detection
- [ ] Deploy to cloud (AWS/GCP/Azure)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
**Ashish Jha**
- Email: jha.ashu3110@gmail.com
- LinkedIn: [Profile](https://www.linkedin.com/in/ashish-jha)
- GitHub: [@ASHISH311001](https://github.com/ASHISH311001)

## Acknowledgments
- Dataset provided by Kaggle
- Inspired by best practices in ML engineering
- Thanks to the open-source community
