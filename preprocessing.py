import pandas as pd
import joblib

def preprocess_new_data(df_new, scaler_path="scaler.pkl", encoder_path="encoder.pkl"):
    """
    Preprocess new loan data with proper feature handling and validation.
  
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    df = df_new.copy()
    
    # 1. COLUMN VALIDATION
    expected_columns = [
        'credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 
        'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 
        'inq.last.6mths', 'delinq.2yrs', 'pub.rec'
    ]
    
    # Check for missing columns
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f" Missing required columns: {missing_cols}")
    
    # Check for extra columns (warning only)
    extra_cols = set(df.columns) - set(expected_columns)
    if extra_cols:
        print(f" Extra columns found (will be ignored): {extra_cols}")
        df = df[expected_columns]  # Keep only expected columns
    
    print(f"Column validation passed. Processing {len(df)} samples.")
    
    # 2. REMOVE DUPLICATES
    initial_rows = len(df)
    df = df.drop_duplicates(keep='first')
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # 3. LOAD PRE-FITTED TRANSFORMERS
    try:
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
        print("Transformers loaded successfully")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Transformer file not found: {e}")

    # 4. ENCODE CATEGORICAL FEATURES
    # Identify unknown categories
    unknown_mask = ~df['purpose'].isin(label_encoder.classes_)
    
    if unknown_mask.any():
        # Calculate percentage
        unknown_count = unknown_mask.sum()
        total_rows = len(df)
        percentage = (unknown_count / total_rows) * 100
        
        print(f"{unknown_count} rows with unknown categories will be deleted.")
        print(f"This represents {percentage:.2f}% of total data ({unknown_count}/{total_rows})")
        
        # Drop rows with unknown categories
        df = df[~unknown_mask].copy()
        
        # Transform the cleaned data (no unknown categories left)
        df['purpose'] = label_encoder.transform(df['purpose'])
        print(f"Categorical encoding completed. Remaining rows: {len(df)}")
    else:
        try:
            df['purpose'] = label_encoder.transform(df['purpose'])
            print("Categorical encoding completed")
        except ValueError as e:
            print(f"Warning: Unknown categories in 'purpose': {e}")
       
    # 5. ENSURE CORRECT COLUMN ORDER FOR SCALER
    # This is CRITICAL - must match training order!
    continuous_features = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico',
                          'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 
                          'delinq.2yrs', 'pub.rec']
    other_features = ['credit.policy', 'purpose']
    
    # Reorder columns to match training
    correct_order = continuous_features + other_features
    df_ordered = df[correct_order]

    # Value ranges (basic sanity checks)
    if (df['fico'] < 300).any() or (df['fico'] > 850).any():
        print("FICO scores outside normal range (300-850)")
    
    if (df['int.rate'] < 0).any() or (df['int.rate'] > 1).any():
        print("Interest rates outside normal range (0-1)")
    
    # 6. APPLY SCALING
    try:
        df_scaled_array = scaler.transform(df_ordered)
        df_scaled = pd.DataFrame(
            df_scaled_array, 
            columns=correct_order, 
            index=df.index
        )
        print("Scaling completed successfully")
        return df_scaled
        
    except Exception as e:
        raise ValueError(f"Scaling failed: {e}")