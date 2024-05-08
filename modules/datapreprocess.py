import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 

def preprocess(df):
    """Performs data preprocessing steps.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """

    # Handle missing values (using median imputation)
    imputer = SimpleImputer(strategy='median')  
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Convert target column to correct data type (if necessary)
    df_imputed['class'] = df_imputed['class'].astype(int)  # Ensure 'class' is of integer type

    # Standardization
    scaler = StandardScaler()
    feature_columns = df_imputed.columns.difference(['class'])  # Exclude 'class'
    scaled_features = scaler.fit_transform(df_imputed[feature_columns])
    df_scaled = pd.DataFrame(scaled_features, columns=feature_columns)
    df_scaled['class'] = df_imputed['class']  # Add 'class' column back

    return df_scaled
