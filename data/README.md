# Dataset Information

## Required Dataset: Telco Customer Churn

### Download Sources:
1. **Kaggle**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. **IBM**: https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113

### Expected Features (23 total):

#### Demographics:
- `gender`: Male/Female
- `SeniorCitizen`: 0 or 1
- `Partner`: Yes/No
- `Dependents`: Yes/No

#### Services:
- `PhoneService`: Yes/No
- `MultipleLines`: Yes/No/No phone service
- `InternetService`: DSL/Fiber optic/No
- `OnlineSecurity`: Yes/No/No internet service
- `OnlineBackup`: Yes/No/No internet service
- `DeviceProtection`: Yes/No/No internet service
- `TechSupport`: Yes/No/No internet service
- `StreamingTV`: Yes/No/No internet service
- `StreamingMovies`: Yes/No/No internet service

#### Account:
- `tenure`: Number of months (0-72)
- `Contract`: Month-to-month/One year/Two year
- `PaperlessBilling`: Yes/No
- `PaymentMethod`: Electronic check/Mailed check/Bank transfer/Credit card
- `MonthlyCharges`: Numeric (18-118)
- `TotalCharges`: Numeric

#### Target:
- `Churn`: Yes/No (what we're predicting)

### File Format:
- Type: CSV
- Encoding: UTF-8
- Separator: comma (,)
- Expected rows: ~7000

### Preprocessing Steps:
1. Convert categorical variables to numeric (0/1)
2. One-hot encode multi-category features (InternetService, Contract, PaymentMethod)
3. Handle missing values in TotalCharges
4. Scale numeric features if needed

### Place Dataset Here:
```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

After downloading, the training script will handle the preprocessing.
