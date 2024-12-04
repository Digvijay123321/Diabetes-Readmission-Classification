import pandas as pd
import numpy as np
import os

def load_data(ids_file, data_file):
    """_summary_

    Args:
        ids_file (_type_): _description_
        data_file (_type_): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        _type_: _description_
    """
    
    if not os.path.exists(ids_file) or not os.path.exists(data_file):
        raise FileNotFoundError(f"One or more input files not found: {ids_file}, {data_file}")

    ids_df = pd.read_csv(ids_file)
    df = pd.read_csv(data_file)
    print(df.head())  # Print the first few rows to check if the file is loaded correctly
    print(f"IDS DataFrame shape: {ids_df.shape}")
    print(f"Main DataFrame shape: {df.shape}")
    return ids_df, df


def split_ids_mapping(ids_df):
    admission_type_id_df = ids_df.iloc[:8]
    discharge_disposition_id_df = ids_df.iloc[10:40].rename(columns={'admission_type_id': 'discharge_disposition_id'})
    admission_source_id_df = ids_df.iloc[42:].rename(columns={'admission_type_id': 'admission_source_id'})
    return admission_type_id_df, discharge_disposition_id_df, admission_source_id_df


def clean_data(df):
    # Drop high null value columns
    df = df.drop(df[df['gender'] == 'Unknown/Invalid'].index)
    df.drop(['max_glu_serum', 'A1Cresult', 'weight', 'payer_code', 'medical_specialty', 'encounter_id', 'race'], axis=1, inplace=True)
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)

    # Check if the DataFrame is empty
    if df.empty:
        print("DataFrame is empty after cleaning. Please check the data source.")
        return None  # Return None to indicate an issue
    return df


# def drop_low_variance_columns(df, threshold=0.999999999):
#     categorical_columns = df.select_dtypes(include=['object', 'category']).columns
#     low_variance_cols = [col for col in categorical_columns if df[col].value_counts(normalize=True).max() > threshold]
#     print(f"Low-variance columns: {low_variance_cols}")
#     return df.drop(low_variance_cols, axis=1)

def drop_low_variance_columns(df, threshold=0.999999999):
    """
    Drops low-variance categorical columns from the DataFrame and returns the modified DataFrame
    along with the list of dropped columns.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        threshold (float): The maximum proportion of the most common category value 
                           to consider as low variance.
    
    Returns:
        pd.DataFrame: DataFrame with low-variance columns removed.
        list: List of dropped column names.
    """
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    # Find low-variance columns
    low_variance_cols = [
        col for col in categorical_columns 
        if df[col].value_counts(normalize=True).max() > threshold
    ]
    
    print(f"Low-variance columns: {low_variance_cols}")
    
    # Drop low-variance columns from the DataFrame
    df = df.drop(low_variance_cols, axis=1)
    
    return df, low_variance_cols



def aggregate_service_utilization(df):
    required_columns = ['number_outpatient', 'number_emergency', 'number_inpatient']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns for aggregation: {missing_columns}")
    
    df['service_utilization'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    df.drop(['number_outpatient', 'number_emergency', 'number_inpatient'], axis=1, inplace=True)
    return df


def encode_drug_changes(df, keys):
    for col in keys:
        colname = f"{col}temp"
        df[colname] = df[col].apply(lambda x: 0 if (x == 'No' or x == 'Steady') else 1)
    
    df['numchange'] = 0
    for col in keys:
        colname = f"{col}temp"
        df['numchange'] += df[colname]
        df.drop(colname, axis=1, inplace=True)
    return df

def map_admission_and_discharge(df):
    # Map `admission_type_id`
    admission_type_mapping = {2: 1, 7: 1, 6: 5, 8: 5}
    df['admission_type_id'] = df['admission_type_id'].replace(admission_type_mapping)
    
    # Map `discharge_disposition_id`
    discharge_mapping = {
        6: 1, 8: 1, 9: 1, 13: 1, 
        3: 2, 4: 2, 5: 2, 14: 2, 22: 2, 23: 2, 24: 2, 
        12: 10, 15: 10, 16: 10, 17: 10, 
        25: 18, 26: 18
    }
    df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(discharge_mapping)
    return df

def map_admission_source(df):
    admission_source_mapping = {
        2: 1, 3: 1, 
        5: 4, 6: 4, 10: 4, 22: 4, 25: 4, 
        15: 9, 17: 9, 20: 9, 21: 9, 
        13: 11, 14: 11
    }
    df['admission_source_id'] = df['admission_source_id'].replace(admission_source_mapping)
    return df

def encode_categorical_columns(df):
    df['change'] = df['change'].replace({'Ch': 1, 'No': 0})
    df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0})
    # df = df.drop(df[df['gender'] == 'Unknown/Invalid'].index)
    df['diabetesMed'] = df['diabetesMed'].replace({'Yes': 1, 'No': 0})
    return df

def encode_drugs(df, keys):
    for col in keys:
        df[col] = df[col].replace({'No': 0, 'Steady': 1, 'Up': 1, 'Down': 1})
    return df

def encode_age(df):
    for i in range(0, 10):
        df['age'] = df['age'].replace(f'[{10*i}-{10*(i+1)})', i+1)
    return df

def preprocess_diagnosis(df):
    for diag in ['diag_1', 'diag_2', 'diag_3']:
        level1_col = f'level1_{diag}'
        level2_col = f'level2_{diag}'
        df[level1_col] = df[diag].replace(r'[VE].*', 0, regex=True).astype(float)
        df[level2_col] = df[level1_col]
    df.drop(['diag_1', 'diag_2', 'diag_3'], axis=1, inplace = True)
    return df

def main():
    ids_file = "/Users/amoghagadde/Desktop/Amogha/Northeastern/SEM_3/ML/Project/Project/Diabetes-Readmission-Classification/Experimentation/IDS_mapping.csv"
    data_file = "/Users/amoghagadde/Desktop/Amogha/Northeastern/SEM_3/ML/Project/Project/Diabetes-Readmission-Classification/Experimentation/diabetic_data.csv"
    
    # Load data
    ids_df, df = load_data(ids_file, data_file)

    # Split mappings
    _, _, _ = split_ids_mapping(ids_df)

    # Clean and preprocess data
    print(f"Initial DataFrame shape: {df.shape}")
    df = clean_data(df)
    print(f"After cleaning: {df.shape}")

    df, dropped_columns = drop_low_variance_columns(df)
    print(f"Before aggregation, df type: {type(df)}, shape: {df.shape if df is not None else 'N/A'}")
    df = aggregate_service_utilization(df)
    print(f"After aggregation, df type: {type(df)}, shape: {df.shape if df is not None else 'N/A'}")
    # # Drug change encoding
    # keys = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
    #         'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
    #         'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide', 
    #         'metformin-pioglitazone', 'glimepiride-pioglitazone', 'glipizide-metformin', 
    #         'troglitazone', 'tolbutamide', 'acetohexamide']
    
    # Update keys based on remaining columns
    all_drug_columns = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone', 'change', 'diabetesMed'] 
    keys = [col for col in all_drug_columns if col in df.columns]
    print(f"Updated drug keys: {keys}")
    
    df = encode_drug_changes(df, keys)

    # Map IDs
    df = map_admission_and_discharge(df)
    df = map_admission_source(df)

    # Encode categorical columns
    df = encode_categorical_columns(df)
    df = encode_drugs(df, keys)
    df = encode_age(df)

    # Process diagnosis columns
    df = preprocess_diagnosis(df)

    # Save preprocessed data to 'data' folder
    output_path = os.path.join("data", "preprocessed_data.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create the folder if it doesn't exist
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}")

if __name__ == "__main__":
    main()