# Loan-Default-Prediction-Model
## This project implements a deep learning prediction model to assess loan default risk using Lending Club's historical loan data. 

# <span style="font-size:30px">&#128181; Lending Club Loan Default Prediction Model </span>  

**Author**: _Dzmitry Kandrykinski_ 
**Date**: _2025-08-25_

---

## Overview
This deep learning model predicts loan default risk for Lending Club loans using a neural network built with TensorFlow/Keras. The model achieves **79.49% accuracy** and **0.7949 AUC-ROC** on test data.

---

## Model Architecture
- **Type**: Deep Neural Network (Sequential)
- **Layers**: 4 hidden layers with BatchNormalization and Dropout
- **Output**: Binary classification (Default/No Default)
- **Framework**: TensorFlow/Keras
- **Total Parameters**: 45,712

---

## <span style="font-weight:bold"> Required Input Data Format

### Column Names and Order (CRITICAL)
### The input data must contain exactly **13 columns**  in this specific order: 
<span style="font-size:30px">&#128073; </span> 

| Column # | Column Name | Data Type | Description |
|----------|-------------|-----------|-------------|
| 1 | `credit.policy` | int | Credit policy compliance (0 or 1) |
| 2 | `purpose` | object/string | Loan purpose category |
| 3 | `int.rate` | float | Interest rate |
| 4 | `installment` | float | Monthly installment amount |
| 5 | `log.annual.inc` | float | Natural log of annual income |
| 6 | `dti` | float | Debt-to-income ratio |
| 7 | `fico` | int | FICO credit score |
| 8 | `days.with.cr.line` | float | Days with credit line |
| 9 | `revol.bal` | int | Revolving credit balance |
| 10 | `revol.util` | float | Revolving line utilization rate |
| 11 | `inq.last.6mths` | int | Credit inquiries in last 6 months |
| 12 | `delinq.2yrs` | int | Delinquencies in past 2 years |
| 13 | `pub.rec` | int | Number of public records |

### Purpose Categories
The `purpose` column must contain one of these **7 categories**:
- `credit_card`
- `debt_consolidation` 
- `educational`
- `home_improvement`
- `major_purchase`
- `other`
- `small_business`

#### <span style="color: red; font-weight:bold">  **Unknown categories will be automatically removed from predictions**

### Target Column
- **Target column `not.fully.paid` is <span style="color: red; font-weight:bold">  NOT </span> required** for prediction
- Model outputs probability of default (0-1 scale)
- Threshold: >0.5 = Default (1), ≤0.5 = No Default (0)

## Required Files for Deployment

### 1. Model Files
- `Lending_Club_best_model.keras` - Trained neural network model
- `encoder.pkl` - Label encoder for categorical features
- `scaler.pkl` - RobustScaler for continuous features
- `preprocessing.py` - Preprocessing pipeline

### 2. Dependencies
```python

Name: scikit-learn
Version: 1.6.1
Name: pandas
Version: 2.2.3
Name: numpy
Version: 1.26.4
Name: tensorflow
Version: 2.19.0
Name: joblib
Version: 1.5.1
```

## Usage Example

```python

import pandas as pd
from preprocessing import preprocess_new_data
import joblib
from tensorflow.keras.models import load_model

# Load your data (without target column)
df = pd.read_csv('customer_loan_data.csv')

# Load trained model
model = load_model('Lending_Club_best_model.keras')

# Run preprocessor with a new data and make predictions
try:
    processed_data = preprocess_new_data(df)
    predictions = model.predict(processed_data)
except Exception as e:
    print(f"Preprocessing failed: {e}")

# Convert the predictions to binary integers using a 'threshold'
threshold = 0.5
binary_predictions = (predictions > threshold).astype('int')

# Add results to dataframe
df['default_probability'] = predictions.flatten()
df['predicted_default'] = binary_predictions.flatten()

# Check the distribution of predictions
print(f" Predicted Default Distribution: {df['predicted_default'].value_counts(1) * 100} ")

# Save the new file
df.to_csv('customer_loan_data_predicted.csv')

```

## Model Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 79.49% |
| **AUC-ROC** | 0.7949 |
| **Precision (No Default)** | 0.83 |
| **Recall (No Default)** | 0.74 |
| **Precision (Default)** | 0.76 |
| **Recall (Default)** | 0.85 |
| **F1-Score (No Default)** | 0.78 |
| **F1-Score (Default)** | 0.81 |

## Data Preprocessing Details

### Scaling
- **Continuous features**: Scaled using RobustScaler
- **Categorical features**: Label encoded (passthrough scaling)
- **Missing values**: Not handled (ensure clean input data)

### Feature Engineering
- Original dataset had severe class imbalance (5.25:1 ratio)
- Training used upsampling to balance classes
- No feature selection applied

## Important Notes

1. **Column Order**: Input data must match the exact column order specified above
2. **Data Quality**: Ensure no missing values in input data
3. **Categories**: Unknown purpose categories will be excluded from predictions
4. **Scaling**: All preprocessing must use the saved scaler.pkl file
5. **Performance**: Model works best on data similar to Lending Club loan characteristics

## Support

For questions or issues with model deployment, ensure:
- Input data format matches specifications exactly
- All required files are available
- Dependencies are properly installed
- Data preprocessing follows the example pipeline

---
*Model trained on Lending Club historical loan data with balanced classes (50/50 split)*
