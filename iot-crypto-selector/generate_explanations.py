import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

def generate_shap_explanations():
    # Load model and data
    with open('models/svm_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    feature_names = model_data['feature_names']
    
    # Load test data
    df = pd.read_csv('data/processed_dataset.csv')
    X = df[feature_names]
    X_scaled = scaler.transform(X)
    
    print("Generating SHAP explanations...")
    
    # Create SHAP explainer
    # Use a subset for faster computation
    background = shap.kmeans(X_scaled, 10)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    
    # Calculate SHAP values for a subset
    sample_size = min(20, len(X_scaled))
    X_sample = X_scaled[:sample_size]
    shap_values = explainer.shap_values(X_sample)
    
    # Get the class names
    class_names = label_encoder.classes_
    
    # 1. Summary plot for all classes
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, 
                     feature_names=feature_names,
                     class_names=class_names,
                     show=False)
    plt.title('SHAP Feature Importance for Algorithm Selection')
    plt.tight_layout()
    plt.savefig('visualizations/shap_summary_all_classes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Feature importance bar plot
    plt.figure(figsize=(10, 6))
    # Average absolute SHAP values across all classes
    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Overall Feature Importance for Algorithm Selection')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance_bar.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Individual prediction explanation
    # Find an interesting case (e.g., 8-bit processor)
    eight_bit_idx = df[df['is_8bit'] == 1].index[0]
    
    for i, class_name in enumerate(class_names):
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(values=shap_values[i][0], 
                           base_values=explainer.expected_value[i],
                           data=X_sample[0],
                           feature_names=feature_names),
            show=False
        )
        plt.title(f'SHAP Explanation for {class_name} Algorithm\n' + 
                 f'(Sample: 8-bit processor, {df.iloc[eight_bit_idx]["CPU_Freq_MHz"]:.0f} MHz)')
        plt.tight_layout()
        plt.savefig(f'visualizations/shap_waterfall_{class_name.replace("-", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        break  # Just show one for demonstration
    
    # 4. Summary statistics
    print("\nSHAP Analysis Complete!")
    print(f"Most important features overall:")
    print(feature_importance.head())
    
    return shap_values, explainer

if __name__ == "__main__":
    shap_values, explainer = generate_shap_explanations()