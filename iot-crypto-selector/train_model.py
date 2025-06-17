#!/usr/bin/env python3
"""
train_model.py
Train Random Forest model for IoT cryptographic algorithm selection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shap

def load_prepared_data():
    """Load the prepared train and test sets"""
    train_df = pd.read_csv('data/processed/train_set.csv')
    test_df = pd.read_csv('data/processed/test_set.csv')
    
    # Define feature columns
    feature_columns = [
        'CPU_Freq_MHz', 'RAM_KB', 'Key_Size_Bits', 'Data_Size_Bytes',
        'log_cpu_freq', 'is_low_memory', 'freq_per_kb_ram', 'key_data_ratio',
        'is_8bit', 'is_32bit', 'is_fpga', 'is_asic'
    ]
    
    # Remove any features that don't exist
    feature_columns = [col for col in feature_columns if col in train_df.columns]
    
    X_train = train_df[feature_columns]
    y_train = train_df['Algorithm_Clean']
    X_test = test_df[feature_columns]
    y_test = test_df['Algorithm_Clean']
    
    return X_train, X_test, y_train, y_test, feature_columns

def train_random_forest(X_train, y_train, optimize_hyperparameters=True):
    """Train Random Forest model with optional hyperparameter optimization"""
    
    if optimize_hyperparameters:
        print("Performing hyperparameter optimization...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
        
        # Initialize base model
        rf_base = RandomForestClassifier(random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=5, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
    else:
        # Use default parameters with class balancing
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance"""
    
    # Training accuracy
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    # Test accuracy
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    
    return test_acc, test_pred, cm

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_feature_importance(model, feature_names, save_path):
    """Analyze and plot feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature importance
    print("\nFeature Importance:")
    for i in range(len(feature_names)):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return dict(zip(feature_names, importances))

def generate_shap_explanations(model, X_train, X_test, feature_names):
    """Generate SHAP explanations for model interpretability"""
    print("\nGenerating SHAP explanations...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for test set
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                      class_names=model.classes_, show=False)
    plt.tight_layout()
    plt.savefig('results/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type="bar", class_names=model.classes_, show=False)
    plt.tight_layout()
    plt.savefig('results/shap_importance_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return explainer, shap_values

def save_model_and_results(model, test_acc, feature_importance, timestamp):
    """Save trained model and results"""
    
    # Save model
    model_path = f'models/final_model_97points_{timestamp}.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save results summary
    results = {
        'timestamp': timestamp,
        'test_accuracy': test_acc,
        'feature_importance': feature_importance,
        'model_params': model.get_params()
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(f'results/model_results_{timestamp}.csv', index=False)

def main():
    """Main execution function"""
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load data
    print("Loading prepared data...")
    X_train, X_test, y_train, y_test, feature_names = load_prepared_data()
    
    # Train model
    print("\nTraining Random Forest model...")
    model = train_random_forest(X_train, y_train, optimize_hyperparameters=False)
    
    # Evaluate model
    print("\nEvaluating model...")
    test_acc, test_pred, cm = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Statistical significance test
    baseline_acc = 1.0 / len(np.unique(y_test))  # Random guessing
    improvement = test_acc / baseline_acc
    print(f"\nBaseline accuracy (random guessing): {baseline_acc:.4f}")
    print(f"Improvement over baseline: {improvement:.2f}x")
    
    # Plot confusion matrix
    class_names = sorted(y_test.unique())
    plot_confusion_matrix(cm, class_names, f'results/confusion_matrix_{timestamp}.png')
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(
        model, feature_names, f'results/feature_importance_{timestamp}.png'
    )
    
    # Generate SHAP explanations
    explainer, shap_values = generate_shap_explanations(
        model, X_train, X_test, feature_names
    )
    
    # Save model and results
    save_model_and_results(model, test_acc, feature_importance, timestamp)
    
    print("\n=== Training Complete ===")
    print(f"Final test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

if __name__ == "__main__":
    main()