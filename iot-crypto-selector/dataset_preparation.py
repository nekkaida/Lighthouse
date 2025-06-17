#!/usr/bin/env python3
"""
dataset_preparation.py
Prepare and process the IoT cryptographic algorithm dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(filepath):
    """Load the raw dataset and perform initial cleaning"""
    df = pd.read_csv(filepath)
    
    # Remove any duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    # Drop rows with missing critical values
    df = df.dropna(subset=['Algorithm_Clean', 'CPU_Freq_MHz', 'RAM_KB'])
    
    return df

def engineer_features(df):
    """Create additional features for better model performance"""
    
    # Log transform CPU frequency to handle wide range
    df['log_cpu_freq'] = np.log1p(df['CPU_Freq_MHz'])
    
    # Create memory constraint indicator
    df['is_low_memory'] = (df['RAM_KB'] < 1.0).astype(int)
    
    # Frequency per KB RAM ratio (computational capability per memory)
    df['freq_per_kb_ram'] = df['CPU_Freq_MHz'] / (df['RAM_KB'] + 0.001)  # avoid division by zero
    
    # Key to data size ratio
    df['key_data_ratio'] = df['Key_Size_Bits'] / (df['Data_Size_Bytes'] * 8)
    
    # Platform type flags (if not already present)
    if 'is_8bit' not in df.columns:
        df['is_8bit'] = df['CPU_Type'].str.contains('8-bit|ATmega', case=False, na=False).astype(int)
    if 'is_32bit' not in df.columns:
        df['is_32bit'] = df['CPU_Type'].str.contains('32-bit|ARM|Cortex', case=False, na=False).astype(int)
    if 'is_fpga' not in df.columns:
        df['is_fpga'] = df['Device_Platform'].str.contains('FPGA', case=False, na=False).astype(int)
    if 'is_asic' not in df.columns:
        df['is_asic'] = df['Device_Platform'].str.contains('ASIC', case=False, na=False).astype(int)
    
    return df

def prepare_ml_data(df):
    """Prepare data for machine learning"""
    
    # Select features for ML model
    feature_columns = [
        'CPU_Freq_MHz', 'RAM_KB', 'Key_Size_Bits', 'Data_Size_Bytes',
        'log_cpu_freq', 'is_low_memory', 'freq_per_kb_ram', 'key_data_ratio',
        'is_8bit', 'is_32bit', 'is_fpga', 'is_asic'
    ]
    
    # Remove any features that don't exist
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    X = df[feature_columns]
    y = df['Algorithm_Clean']
    
    # Print class distribution
    print("\nClass distribution:")
    print(y.value_counts())
    print(f"\nTotal samples: {len(y)}")
    
    return X, y, feature_columns

def split_and_save_data(X, y, df, test_size=0.25, random_state=42):
    """Split data into train/test sets and save"""
    
    # Stratified split to maintain class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Get indices for saving full rows
    train_indices = X_train.index
    test_indices = X_test.index
    
    # Save processed dataset
    df.to_csv('data/processed/dataset_97_processed.csv', index=False)
    
    # Save train and test sets with all columns
    df.loc[train_indices].to_csv('data/processed/train_set.csv', index=False)
    df.loc[test_indices].to_csv('data/processed/test_set.csv', index=False)
    
    print(f"\nTrain set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main execution function"""
    
    # Load raw data
    print("Loading raw dataset...")
    df = load_and_clean_data('data/raw/final_core4_dataset.csv')
    
    # Engineer features
    print("\nEngineering features...")
    df = engineer_features(df)
    
    # Prepare ML data
    print("\nPreparing ML data...")
    X, y, feature_columns = prepare_ml_data(df)
    
    # Split and save data
    print("\nSplitting and saving data...")
    X_train, X_test, y_train, y_test = split_and_save_data(X, y, df)
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Features used: {len(feature_columns)}")
    print(f"Feature names: {feature_columns}")
    print("\nAlgorithm distribution:")
    for algo in y.unique():
        count = (y == algo).sum()
        percentage = (count / len(y)) * 100
        print(f"  {algo}: {count} samples ({percentage:.1f}%)")

if __name__ == "__main__":
    main()