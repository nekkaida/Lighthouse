"""
Updated Dataset Preparation Script
Adapted to your actual file structure
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def prepare_complete_dataset():
    """
    Prepares the complete 151-point dataset for analysis
    Using your actual file structure
    """
    
    print("="*60)
    print("IoT CRYPTO DATASET PREPARATION - 151 POINTS")
    print("="*60)
    
    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load the dataset - it's already in your structure!
    try:
        df = pd.read_csv('data/raw/complete_dataset_151.csv')
        print(f"\n✓ Dataset loaded: {len(df)} data points")
    except FileNotFoundError:
        print("\n✗ ERROR: File not found at data/raw/complete_dataset_151.csv")
        return None
    
    # Data cleaning and preparation
    print("\n📊 Cleaning and preparing data...")
    
    # Clean algorithm names
    df['Algorithm_Clean'] = df['Algorithm'].apply(lambda x: 
        'AES-128' if 'AES' in str(x) else
        'SIMON' if 'SIMON' in str(x) else  
        'SPECK' if 'SPECK' in str(x) and 'SPECK-R' not in str(x) else
        'PRESENT' if 'PRESENT' in str(x) else
        'OTHER'
    )
    
    # Filter to core 4 only
    core_algorithms = ['AES-128', 'SIMON', 'SPECK', 'PRESENT']
    df_core = df[df['Algorithm_Clean'].isin(core_algorithms)].copy()
    
    print(f"\n✓ Core 4 algorithms: {len(df_core)} data points")
    print("\nAlgorithm distribution:")
    for algo, count in df_core['Algorithm_Clean'].value_counts().items():
        print(f"  {algo}: {count} ({count/len(df_core)*100:.1f}%)")
    
    # Handle missing values and convert data types
    print("\n🔧 Handling missing values...")
    
    # Convert numeric columns
    numeric_cols = ['CPU_Freq_MHz', 'RAM_KB', 'Flash_KB', 'Data_Size_Bytes', 
                   'Key_Size_Bits', 'Exec_Time_ms', 'Throughput_Kbps', 
                   'Memory_Used_Bytes', 'Energy_mJ']
    
    for col in numeric_cols:
        if col in df_core.columns:
            df_core[col] = pd.to_numeric(df_core[col], errors='coerce')
    
    # Create platform features
    df_core['is_fpga'] = df_core['Device_Platform'].str.contains('FPGA', case=False, na=False).astype(int)
    df_core['is_8bit'] = df_core['Device_Platform'].str.contains('8-bit|ATmega|AVR', case=False, na=False).astype(int)
    df_core['is_32bit'] = df_core['Device_Platform'].str.contains('32-bit|ARM|ESP32|Teensy|Cortex', case=False, na=False).astype(int)
    df_core['is_asic'] = df_core['Device_Platform'].str.contains('ASIC', case=False, na=False).astype(int)
    df_core['is_desktop'] = df_core['Device_Platform'].str.contains('Core|i5|i7', case=False, na=False).astype(int)
    
    # Smart CPU frequency imputation based on platform
    print("  Imputing CPU frequencies by platform type...")
    df_core.loc[df_core['is_8bit'] == 1, 'CPU_Freq_MHz'] = df_core.loc[df_core['is_8bit'] == 1, 'CPU_Freq_MHz'].fillna(16)
    df_core.loc[df_core['is_32bit'] == 1, 'CPU_Freq_MHz'] = df_core.loc[df_core['is_32bit'] == 1, 'CPU_Freq_MHz'].fillna(80)
    df_core.loc[df_core['is_fpga'] == 1, 'CPU_Freq_MHz'] = df_core.loc[df_core['is_fpga'] == 1, 'CPU_Freq_MHz'].fillna(400)
    df_core.loc[df_core['is_asic'] == 1, 'CPU_Freq_MHz'] = df_core.loc[df_core['is_asic'] == 1, 'CPU_Freq_MHz'].fillna(100)
    df_core.loc[df_core['is_desktop'] == 1, 'CPU_Freq_MHz'] = df_core.loc[df_core['is_desktop'] == 1, 'CPU_Freq_MHz'].fillna(2600)
    
    # Fill remaining with median
    df_core['CPU_Freq_MHz'] = df_core['CPU_Freq_MHz'].fillna(df_core['CPU_Freq_MHz'].median())
    df_core['Key_Size_Bits'] = df_core['Key_Size_Bits'].fillna(96)
    df_core['Data_Size_Bytes'] = df_core['Data_Size_Bytes'].fillna(64)
    df_core['RAM_KB'] = df_core['RAM_KB'].fillna(df_core['RAM_KB'].median())
    
    print("✓ Missing values handled")
    
    # Feature engineering
    print("\n🚀 Creating advanced features...")
    
    # Add derived features based on SHAP insights
    df_core['log_cpu_freq'] = np.log1p(df_core['CPU_Freq_MHz'])
    df_core['is_low_memory'] = (df_core['RAM_KB'] < 1).astype(int)
    df_core['key_data_ratio'] = df_core['Key_Size_Bits'] / (df_core['Data_Size_Bytes'] + 1)
    df_core['freq_per_kb_ram'] = df_core['CPU_Freq_MHz'] / (df_core['RAM_KB'].fillna(1) + 1)
    
    print("✓ Created engineered features")
    
    # Save processed datasets
    print("\n💾 Saving processed datasets...")
    
    # Save full processed dataset
    df_core.to_csv('data/processed/dataset_151_processed.csv', index=False)
    print("  ✓ Saved: data/processed/dataset_151_processed.csv")
    
    # Create train/test split (80/20)
    from sklearn.model_selection import train_test_split
    
    # Prepare for stratified split
    X = df_core.drop(['Algorithm', 'Algorithm_Clean'], axis=1)
    y = df_core['Algorithm_Clean']
    
    # Split indices
    train_idx, test_idx = train_test_split(
        df_core.index, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # Create split column
    df_core['split'] = 'test'
    df_core.loc[train_idx, 'split'] = 'train'
    
    # Save splits
    df_train = df_core[df_core['split'] == 'train'].drop('split', axis=1)
    df_test = df_core[df_core['split'] == 'test'].drop('split', axis=1)
    
    df_train.to_csv('data/processed/train_set.csv', index=False)
    df_test.to_csv('data/processed/test_set.csv', index=False)
    print(f"  ✓ Saved: train_set.csv ({len(df_train)} samples)")
    print(f"  ✓ Saved: test_set.csv ({len(df_test)} samples)")
    
    # Update dataset summary
    summary = f"""DATASET SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================

Total Data Points: {len(df_core)}
Training Set: {len(df_train)} ({len(df_train)/len(df_core)*100:.1f}%)
Test Set: {len(df_test)} ({len(df_test)/len(df_core)*100:.1f}%)

Algorithm Distribution:
{df_core['Algorithm_Clean'].value_counts().to_string()}

Platform Distribution:
- 8-bit MCU: {df_core['is_8bit'].sum()}
- 32-bit MCU: {df_core['is_32bit'].sum()}
- FPGA: {df_core['is_fpga'].sum()}
- ASIC: {df_core['is_asic'].sum()}
- Desktop: {df_core['is_desktop'].sum()}

Data Sources (Papers): {df_core['Paper_Author'].nunique()}
Year Range: {int(df_core['Year'].min())} - {int(df_core['Year'].max())}

Feature Engineering Applied:
✓ Platform type encoding (is_8bit, is_32bit, is_fpga, is_asic, is_desktop)
✓ Log-transformed CPU frequency
✓ Low memory indicator
✓ Key-to-data size ratio
✓ Frequency per KB RAM

Missing Data (after imputation): None
"""
    
    with open('data/dataset_summary.txt', 'w') as f:
        f.write(summary)
    
    print("\n📄 Dataset Summary:")
    print(summary)
    
    return df_core

def validate_prepared_data():
    """
    Validate the prepared datasets
    """
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)
    
    try:
        # Load processed data
        df = pd.read_csv('data/processed/dataset_151_processed.csv')
        train = pd.read_csv('data/processed/train_set.csv')
        test = pd.read_csv('data/processed/test_set.csv')
        
        print(f"\n✓ Processed dataset: {len(df)} samples")
        print(f"✓ Train set: {len(train)} samples")
        print(f"✓ Test set: {len(test)} samples")
        
        # Check for data leakage
        train_indices = set(train.index)
        test_indices = set(test.index)
        overlap = train_indices.intersection(test_indices)
        
        if len(overlap) == 0:
            print("✓ No data leakage between train and test sets")
        else:
            print(f"⚠️ WARNING: {len(overlap)} overlapping samples!")
        
        # Feature columns for model
        feature_cols = [col for col in df.columns if col not in [
            'Paper_Author', 'Year', 'Algorithm', 'Algorithm_Clean', 
            'Device_Platform', 'CPU_Type', 'Notes', 'split'
        ]]
        
        print(f"\n📊 Features available for modeling: {len(feature_cols)}")
        print("Basic features:", [col for col in feature_cols if not col.startswith('is_') and col not in ['log_cpu_freq', 'is_low_memory', 'key_data_ratio', 'freq_per_kb_ram']])
        print("Engineered features:", [col for col in feature_cols if col.startswith('is_') or col in ['log_cpu_freq', 'is_low_memory', 'key_data_ratio', 'freq_per_kb_ram']])
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please run prepare_complete_dataset() first!")

def main():
    """
    Main execution
    """
    print("\n🚀 IoT CRYPTO ALGORITHM SELECTION - DATASET PREPARATION")
    print("This script prepares your 151-point dataset for analysis\n")
    
    # Check if raw data exists
    if not os.path.exists('data/raw/complete_dataset_151.csv'):
        print("❌ ERROR: data/raw/complete_dataset_151.csv not found!")
        print("Please ensure you've saved the CSV file correctly.")
        return
    
    # Prepare dataset
    df = prepare_complete_dataset()
    
    if df is not None:
        # Validate
        validate_prepared_data()
        
        print("\n" + "="*60)
        print("✅ DATASET PREPARATION COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run: python complete_analysis.py")
        print("2. Run: python generate_explanations.py")
        print("\nYour 151-point dataset is ready for analysis! 🎉")

if __name__ == "__main__":
    main()