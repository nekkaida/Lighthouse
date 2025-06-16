"""
Setup Checker - Verify your environment before running analysis
"""

import os
import sys
import pandas as pd

def check_setup():
    """
    Check if everything is set up correctly
    """
    print("="*60)
    print("🔍 IoT CRYPTO PROJECT - SETUP CHECKER")
    print("="*60)
    
    issues = []
    warnings = []
    
    # Check Python version
    print("\n1️⃣ Checking Python environment...")
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 7:
        print(f"  ✓ Python {python_version.major}.{python_version.minor} - OK")
    else:
        issues.append("Python 3.7+ required")
    
    # Check required libraries
    print("\n2️⃣ Checking required libraries...")
    required_libs = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical plots',
        'shap': 'Model explanations'
    }
    
    for lib, purpose in required_libs.items():
        try:
            __import__(lib)
            print(f"  ✓ {lib} - {purpose}")
        except ImportError:
            issues.append(f"Missing library: {lib} (needed for {purpose})")
    
    # Check directory structure
    print("\n3️⃣ Checking directory structure...")
    required_dirs = ['data', 'data/raw', 'data/processed', 'models', 'results', 'visualizations']
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}/ exists")
        else:
            warnings.append(f"Missing directory: {dir_path} (will be created)")
    
    # Check data files
    print("\n4️⃣ Checking data files...")
    
    # Check raw data
    if os.path.exists('data/raw/complete_dataset_151.csv'):
        try:
            df = pd.read_csv('data/raw/complete_dataset_151.csv')
            print(f"  ✓ complete_dataset_151.csv found ({len(df)} rows)")
            
            # Quick data validation
            if len(df) >= 150:
                print(f"    ✓ Dataset size OK: {len(df)} points")
            else:
                warnings.append(f"Dataset has only {len(df)} points (expected 151)")
                
        except Exception as e:
            issues.append(f"Error reading dataset: {e}")
    else:
        issues.append("Missing: data/raw/complete_dataset_151.csv")
    
    # Check if processed data exists
    if os.path.exists('data/processed/dataset_151_processed.csv'):
        print("  ✓ Processed data exists")
    else:
        print("  ⚠️ No processed data yet (run dataset_preparation.py)")
    
    # Check models
    print("\n5️⃣ Checking models...")
    if os.path.exists('models/svm_model.pkl'):
        print("  ⚠️ Found old model (61 points) - will be replaced")
    
    if os.path.exists('models/final_model_151points.pkl'):
        print("  ✓ New model (151 points) exists")
    else:
        print("  ⚠️ No new model yet (run complete_analysis.py)")
    
    # Check Python scripts
    print("\n6️⃣ Checking required scripts...")
    required_scripts = [
        'dataset_preparation.py',
        'complete_analysis.py',
        'generate_explanations.py'
    ]
    
    for script in required_scripts:
        if os.path.exists(script):
            print(f"  ✓ {script}")
        else:
            issues.append(f"Missing script: {script}")
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    if not issues:
        print("\n✅ All critical checks passed! You're ready to proceed.")
        print("\n🚀 Next steps:")
        print("1. Run: python dataset_preparation.py")
        print("2. Run: python complete_analysis.py")
        print("3. Run: python generate_explanations.py")
    else:
        print("\n❌ Critical issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before proceeding.")
    
    if warnings:
        print("\n⚠️ Warnings (non-critical):")
        for warning in warnings:
            print(f"  - {warning}")
    
    return len(issues) == 0

def quick_data_preview():
    """
    Show a quick preview of the data
    """
    print("\n" + "="*60)
    print("📊 QUICK DATA PREVIEW")
    print("="*60)
    
    try:
        df = pd.read_csv('data/raw/complete_dataset_151.csv')
        
        # Clean algorithm names
        df['Algorithm_Clean'] = df['Algorithm'].apply(lambda x: 
            'AES-128' if 'AES' in str(x) else
            'SIMON' if 'SIMON' in str(x) else  
            'SPECK' if 'SPECK' in str(x) and 'SPECK-R' not in str(x) else
            'PRESENT' if 'PRESENT' in str(x) else
            'OTHER'
        )
        
        # Filter to core 4
        df_core = df[df['Algorithm_Clean'].isin(['AES-128', 'SIMON', 'SPECK', 'PRESENT'])]
        
        print(f"\nTotal rows: {len(df)}")
        print(f"Core 4 algorithms: {len(df_core)}")
        
        print("\nAlgorithm distribution:")
        print(df_core['Algorithm_Clean'].value_counts())
        
        print("\nPapers included:")
        papers = df_core['Paper_Author'].unique()
        for i, paper in enumerate(papers[:5], 1):
            print(f"  {i}. {paper}")
        if len(papers) > 5:
            print(f"  ... and {len(papers)-5} more")
        
        print(f"\nYear range: {int(df_core['Year'].min())} - {int(df_core['Year'].max())}")
        
    except Exception as e:
        print(f"Could not preview data: {e}")

if __name__ == "__main__":
    # Run setup check
    is_ready = check_setup()
    
    # Show data preview if ready
    if is_ready:
        quick_data_preview()
    
    print("\n" + "="*60)
    print("Setup check complete!")
    print("="*60)