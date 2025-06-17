#!/usr/bin/env python3
"""
complete_analysis.py
Complete analysis pipeline for IoT cryptographic algorithm selection
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import shap
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_analyze_dataset(filepath='data/processed/dataset_97_processed.csv'):
    """Load dataset and perform exploratory data analysis"""
    
    df = pd.read_csv(filepath)
    
    print("=== Dataset Overview ===")
    print(f"Total samples: {len(df)}")
    print(f"Features: {df.shape[1]}")
    print(f"Algorithms: {df['Algorithm_Clean'].unique()}")
    
    # Algorithm distribution
    algo_counts = df['Algorithm_Clean'].value_counts()
    print("\n=== Algorithm Distribution ===")
    for algo, count in algo_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{algo}: {count} samples ({percentage:.1f}%)")
    
    # Platform distribution
    print("\n=== Platform Distribution ===")
    platform_counts = df['Device_Platform'].value_counts()
    print(platform_counts.head(10))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Algorithm distribution
    axes[0, 0].bar(algo_counts.index, algo_counts.values)
    axes[0, 0].set_title('Algorithm Distribution')
    axes[0, 0].set_xlabel('Algorithm')
    axes[0, 0].set_ylabel('Count')
    
    # CPU Frequency distribution by algorithm
    for algo in df['Algorithm_Clean'].unique():
        data = df[df['Algorithm_Clean'] == algo]['CPU_Freq_MHz']
        axes[0, 1].hist(np.log1p(data), alpha=0.5, label=algo, bins=20)
    axes[0, 1].set_title('Log CPU Frequency Distribution')
    axes[0, 1].set_xlabel('Log(CPU Freq + 1)')
    axes[0, 1].legend()
    
    # RAM distribution by algorithm
    for algo in df['Algorithm_Clean'].unique():
        data = df[df['Algorithm_Clean'] == algo]['RAM_KB']
        axes[1, 0].hist(np.log1p(data), alpha=0.5, label=algo, bins=20)
    axes[1, 0].set_title('Log RAM Distribution')
    axes[1, 0].set_xlabel('Log(RAM KB + 1)')
    axes[1, 0].legend()
    
    # Key size distribution
    key_sizes = df.groupby(['Algorithm_Clean', 'Key_Size_Bits']).size().unstack(fill_value=0)
    key_sizes.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Key Size Distribution by Algorithm')
    axes[1, 1].set_xlabel('Algorithm')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('results/dataset_analysis.png', dpi=300)
    plt.close()
    
    return df

def prepare_features(df):
    """Prepare features for machine learning"""
    
    feature_columns = [
        'CPU_Freq_MHz', 'RAM_KB', 'Key_Size_Bits', 'Data_Size_Bytes',
        'log_cpu_freq', 'is_low_memory', 'freq_per_kb_ram', 'key_data_ratio',
        'is_8bit', 'is_32bit', 'is_fpga', 'is_asic'
    ]
    
    # Remove any features that don't exist
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    X = df[feature_columns]
    y = df['Algorithm_Clean']
    
    return X, y, feature_columns

def train_and_evaluate_model(X, y, feature_names):
    """Train model and perform comprehensive evaluation"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\n=== Data Split ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model with class balancing
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== Model Performance ===")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Statistical significance test
    n_classes = len(np.unique(y))
    baseline_accuracy = 1.0 / n_classes
    n_test = len(y_test)
    
    # Binomial test
    successes = int(test_accuracy * n_test)
    p_value = stats.binom_test(successes, n_test, baseline_accuracy, alternative='greater')
    
    # Confidence interval (Wilson score interval)
    z = 1.96  # 95% confidence
    center = (successes + z**2/2) / (n_test + z**2)
    margin = z * np.sqrt(center * (1 - center) / (n_test + z**2))
    ci_lower = center - margin
    ci_upper = center + margin
    
    print(f"\n=== Statistical Analysis ===")
    print(f"Baseline accuracy (random): {baseline_accuracy:.4f} ({baseline_accuracy*100:.1f}%)")
    print(f"Improvement factor: {test_accuracy/baseline_accuracy:.2f}x")
    print(f"Binomial test p-value: {p_value:.6f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\n=== Cross-Validation ===")
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    importances = model.feature_importances_
    feature_importance = dict(zip(feature_names, importances))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=== Feature Importance ===")
    for feature, importance in sorted_features:
        print(f"{feature}: {importance:.4f} ({importance*100:.1f}%)")
    
    return model, X_train, X_test, y_test, y_pred, test_accuracy

def create_visualizations(model, X_test, y_test, y_pred, feature_names):
    """Create all visualization plots"""
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(np.unique(y_test)),
                yticklabels=sorted(np.unique(y_test)))
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Algorithm')
    plt.xlabel('Predicted Algorithm')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix_final.png', dpi=300)
    plt.close()
    
    # 2. Feature Importance from Random Forest
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance - Random Forest")
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), 
               [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('results/feature_importance_rf.png', dpi=300)
    plt.close()
    
    # 3. SHAP Analysis
    print("\n=== Generating SHAP Explanations ===")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      class_names=model.classes_, show=False)
    plt.tight_layout()
    plt.savefig('results/shap_summary_all_classes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP importance plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type="bar", class_names=model.classes_, show=False)
    plt.tight_layout()
    plt.savefig('results/shap_importance_15points.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return explainer, shap_values

def generate_example_predictions(model, X_test, feature_names, n_examples=3):
    """Generate example predictions with explanations"""
    
    print("\n=== Example Predictions with Explanations ===")
    
    # Get a few test samples
    sample_indices = np.random.choice(X_test.index, n_examples, replace=False)
    
    explainer = shap.TreeExplainer(model)
    
    for idx in sample_indices:
        sample = X_test.loc[[idx]]
        prediction = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]
        
        print(f"\n--- Example {idx} ---")
        print(f"Predicted Algorithm: {prediction}")
        print(f"Confidence: {max(probabilities)*100:.1f}%")
        print("\nDevice Characteristics:")
        print(f"  CPU Frequency: {sample['CPU_Freq_MHz'].values[0]:.1f} MHz")
        print(f"  RAM: {sample['RAM_KB'].values[0]:.3f} KB")
        print(f"  Key Size: {sample['Key_Size_Bits'].values[0]} bits")
        
        # Get SHAP values for this prediction
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            # Multi-class: get values for predicted class
            class_idx = list(model.classes_).index(prediction)
            shap_vals = shap_values[class_idx][0]
        else:
            shap_vals = shap_values[0]
        
        # Top contributing features
        feature_impacts = list(zip(feature_names, shap_vals))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("\nTop Contributing Features:")
        for feat, impact in feature_impacts[:3]:
            direction = "increases" if impact > 0 else "decreases"
            print(f"  {feat}: {direction} likelihood (impact: {impact:.3f})")

def main():
    """Run complete analysis pipeline"""
    
    print("=== IoT Cryptographic Algorithm Selection Analysis ===\n")
    
    # Load and analyze dataset
    df = load_and_analyze_dataset()
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Train and evaluate model
    model, X_train, X_test, y_test, y_pred, test_accuracy = train_and_evaluate_model(X, y, feature_names)
    
    # Create visualizations
    explainer, shap_values = create_visualizations(model, X_test, y_test, y_pred, feature_names)
    
    # Generate example predictions
    generate_example_predictions(model, X_test, feature_names)
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/final_model_{timestamp}.pkl'
    joblib.dump(model, model_path)
    print(f"\n=== Model saved to: {model_path} ===")
    
    # Save results summary
    results_summary = {
        'timestamp': timestamp,
        'total_samples': len(df),
        'test_accuracy': test_accuracy,
        'baseline_accuracy': 0.25,
        'improvement_factor': test_accuracy / 0.25,
        'algorithm_distribution': df['Algorithm_Clean'].value_counts().to_dict()
    }
    
    pd.DataFrame([results_summary]).to_csv(f'results/analysis_summary_{timestamp}.csv', index=False)
    
    print("\n=== Analysis Complete ===")
    print(f"Final model accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

if __name__ == "__main__":
    main()