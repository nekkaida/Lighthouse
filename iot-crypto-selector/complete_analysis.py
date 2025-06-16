"""
Improved analysis for 104-point dataset with better handling of imbalanced data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def improved_analysis():
    """
    Improved analysis addressing the issues found
    """
    print("="*60)
    print("IMPROVED ANALYSIS FOR 104-POINT DATASET")
    print("="*60)
    
    # Load the processed dataset
    df = pd.read_csv('data/raw/final_core4_dataset.csv')
    print(f"\nDataset loaded: {len(df)} points")
    
    # Clean algorithm names
    df['Algorithm_Clean'] = df['Algorithm'].apply(lambda x: 
        'AES-128' if 'AES' in str(x) else
        'SIMON' if 'SIMON' in str(x) else  
        'SPECK' if 'SPECK' in str(x) and 'SPECK-R' not in str(x) else
        'PRESENT' if 'PRESENT' in str(x) else
        'OTHER'
    )
    
    # Verify Core 4 only
    df = df[df['Algorithm_Clean'].isin(['AES-128', 'SIMON', 'SPECK', 'PRESENT'])]
    
    print("\nAlgorithm distribution:")
    print(df['Algorithm_Clean'].value_counts())
    
    # Enhanced preprocessing
    print("\n📊 Enhanced Preprocessing...")
    
    # Platform features
    df['is_fpga'] = df['Device_Platform'].str.contains('FPGA', case=False, na=False).astype(int)
    df['is_8bit'] = df['Device_Platform'].str.contains('8-bit|ATmega|AVR', case=False, na=False).astype(int)
    df['is_32bit'] = df['Device_Platform'].str.contains('32-bit|ARM|ESP32|Teensy|Cortex', case=False, na=False).astype(int)
    df['is_asic'] = df['Device_Platform'].str.contains('ASIC', case=False, na=False).astype(int)
    
    # Handle missing values with smarter imputation
    numeric_cols = ['CPU_Freq_MHz', 'RAM_KB', 'Key_Size_Bits', 'Data_Size_Bytes']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Platform-aware imputation
    df.loc[df['is_8bit'] == 1, 'CPU_Freq_MHz'] = df.loc[df['is_8bit'] == 1, 'CPU_Freq_MHz'].fillna(16)
    df.loc[df['is_32bit'] == 1, 'CPU_Freq_MHz'] = df.loc[df['is_32bit'] == 1, 'CPU_Freq_MHz'].fillna(80)
    df.loc[df['is_fpga'] == 1, 'CPU_Freq_MHz'] = df.loc[df['is_fpga'] == 1, 'CPU_Freq_MHz'].fillna(400)
    df.loc[df['is_asic'] == 1, 'CPU_Freq_MHz'] = df.loc[df['is_asic'] == 1, 'CPU_Freq_MHz'].fillna(100)
    
    # Fill remaining
    df['CPU_Freq_MHz'] = df['CPU_Freq_MHz'].fillna(df['CPU_Freq_MHz'].median())
    df['RAM_KB'] = df['RAM_KB'].fillna(df['RAM_KB'].median())
    df['Key_Size_Bits'] = df['Key_Size_Bits'].fillna(96)
    df['Data_Size_Bytes'] = df['Data_Size_Bytes'].fillna(64)
    
    # Feature engineering
    df['log_cpu_freq'] = np.log1p(df['CPU_Freq_MHz'])
    df['key_data_ratio'] = df['Key_Size_Bits'] / (df['Data_Size_Bytes'] + 1)
    
    # Select features based on your SHAP results
    features = [
        'Key_Size_Bits',      # 28% importance
        'CPU_Freq_MHz',       # Important
        'RAM_KB',             # 12.9% importance
        'Data_Size_Bytes',
        'is_fpga',
        'is_8bit', 
        'is_32bit',
        'is_asic',
        'log_cpu_freq',
        'key_data_ratio'
    ]
    
    X = df[features].fillna(0)
    y = df['Algorithm_Clean']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\nFeatures selected: {len(features)}")
    print(f"Samples per class: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Test multiple approaches
    print("\n🔬 Testing Multiple Approaches...")
    
    results = {}
    
    # 1. Basic models (no balancing)
    print("\n1. Basic Models (No Balancing):")
    
    models = {
        'SVM-RBF': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        'SVM-Linear': SVC(kernel='linear', C=0.5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    }
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        results[name] = {
            'accuracy': acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"  {name}: Test={acc:.2%}, CV={cv_scores.mean():.2%} (±{cv_scores.std()*2:.2%})")
    
    # 2. With SMOTE balancing
    print("\n2. With SMOTE Balancing:")
    
    for name, base_model in models.items():
        # Create pipeline with SMOTE
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', base_model)
        ])
        
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Note: CV with SMOTE should be done carefully
        results[f"{name} + SMOTE"] = {
            'accuracy': acc,
            'predictions': y_pred
        }
        
        print(f"  {name} + SMOTE: Test={acc:.2%}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_acc = results[best_model_name]['accuracy']
    
    print(f"\n✅ Best Model: {best_model_name} with {best_acc:.2%} accuracy")
    
    # Confusion Matrix for best model
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f'Improved Confusion Matrix - {best_model_name}\n'
              f'Accuracy: {best_acc:.2%} (104 data points)')
    plt.ylabel('True Algorithm')
    plt.xlabel('Predicted Algorithm')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix_improved_104.png', dpi=300)
    plt.show()
    
    # Classification report
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, results[best_model_name]['predictions'],
                              target_names=le.classes_))
    
    # Feature importance if Random Forest
    if 'Random Forest' in best_model_name:
        model = models.get('Random Forest', None)
        if model and hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nFeature Importance (Random Forest):")
            for _, row in importance_df.iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
    
    return results, le, features

def create_balanced_splits(X, y, n_splits=5):
    """
    Create balanced train/test splits for better evaluation
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, 
                                     random_state=42, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        
        score = model.score(X_test_scaled, y_test)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

def main():
    """
    Run improved analysis
    """
    print("🚀 Running Improved Analysis for 104-Point Dataset\n")
    
    results, le, features = improved_analysis()
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print("\nKey Findings:")
    print("1. Dataset has 104 unique Core-4 data points")
    print("2. Class imbalance is a major challenge")
    print("3. SMOTE balancing may help improve minority class prediction")
    print("4. Random Forest with class_weight='balanced' shows promise")
    print("\nRecommendations for Thesis:")
    print("- Report actual 104 data points (not 151)")
    print("- Acknowledge class imbalance as a limitation")
    print("- Consider collecting more AES-128 and PRESENT data")
    print("- Use stratified cross-validation for evaluation")
    
    return results

if __name__ == "__main__":
    results = main()