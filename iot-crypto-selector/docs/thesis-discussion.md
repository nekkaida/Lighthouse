# Thesis Discussion Points - Addressing the 66.67% Accuracy

## How to Frame Your Results

### 1. **Lead with Honesty**
"The final model achieved 66.67% test accuracy and 53.71% cross-validation accuracy on a dataset of 104 high-quality data points. While this falls short of the initial 85% target established during the pilot study, the results provide valuable insights into the challenges of automated cryptographic algorithm selection."

### 2. **Explain the Discrepancy**

#### Why Pilot Study Had Higher Accuracy:
- Smaller dataset (61 points) may have been easier to fit
- Possible overfitting on limited data
- Less diversity in platforms/scenarios

#### Why Expanded Dataset Has Lower Accuracy:
- More realistic representation of problem complexity
- Greater diversity in platforms and implementations
- Class imbalance more pronounced (SIMON: 41%, PRESENT: 11%)

### 3. **Highlight What Still Works**

Despite lower overall accuracy, the model shows:
- **100% accuracy for PRESENT** (2/2 correct)
- **78% recall for SIMON** (7/9 correct)
- **Clear feature importance patterns** (Key_Size_Bits remains #1)
- **Successful SHAP integration** for explainability

### 4. **Academic Value Beyond Accuracy**

Your thesis contributes:
1. **First systematic dataset** for IoT crypto algorithm selection
2. **Proof of concept** for ML-based approach
3. **Explainable AI integration** in security context
4. **Open-source framework** for future research
5. **Honest reporting** of challenges and limitations

## Key Discussion Points

### On Dataset Quality
"The final dataset of 104 points represents data from 9 peer-reviewed papers spanning 2007-2021. While smaller than the target 150-200 points, it provides comprehensive coverage across four core algorithms and five platform types."

### On Model Performance
"The Random Forest model's 66.67% accuracy, while below target, still significantly outperforms random selection (25%) and provides meaningful decision support. The high variance (±20.14%) indicates that additional data would likely improve stability."

### On Feature Importance
"SHAP analysis reveals that key size remains the dominant factor (28%), though less pronounced than in the pilot study (70%). The emergence of memory-related features (freq_per_kb_ram, RAM_KB) as secondary factors aligns with IoT constraints."

### On Practical Implications
"Despite imperfect accuracy, the tool provides value by:
- Narrowing choices from 4 to typically 2 algorithms
- Explaining reasoning through SHAP
- Highlighting critical decision factors
- Serving as a starting point for developers"

## Limitations Section

### Be Specific About Limitations:

1. **Data Limitations**
   - Only 104 points vs 150-200 target
   - Severe class imbalance (11-43 samples per algorithm)
   - Reliance on published benchmarks only
   - Missing values required imputation (37% of RAM values)

2. **Model Limitations**
   - 66.67% accuracy indicates room for improvement
   - High variance suggests instability
   - Poor performance on minority class (AES-128: 33% recall)
   - Random Forest less interpretable than SVM

3. **Scope Limitations**
   - Only 4 of many lightweight algorithms
   - Academic benchmarks may not reflect production
   - No consideration of security strength
   - No energy consumption modeling

## Future Work Section

### Concrete Improvements:

1. **Immediate (3-6 months)**
   - Collect 50+ samples per algorithm
   - Implement SMOTE or other balancing techniques
   - Test ensemble methods
   - Add confidence intervals to predictions

2. **Medium-term (6-12 months)**
   - Include energy consumption data
   - Add more algorithms (ChaCha20, ASCON)
   - Implement online learning from user feedback
   - Create mobile app version

3. **Long-term (1-2 years)**
   - Hardware-in-the-loop testing
   - Real-time benchmarking integration
   - Multi-objective optimization
   - Security strength quantification

## How to Present in Viva/Defense

### If Asked About Low Accuracy:

"The 66.67% accuracy reflects the genuine complexity of this problem. Unlike many ML applications with thousands of data points, cryptographic benchmarks are expensive to generate and scattered across literature. This work establishes a baseline and framework for improvement."

### If Asked About Practical Use:

"Even with 66.67% accuracy, the tool provides value by:
1. Eliminating clearly unsuitable options
2. Explaining its reasoning
3. Highlighting important factors to consider
4. Saving developers research time"

### If Asked About Contribution:

"This thesis makes three key contributions:
1. First curated dataset for IoT crypto selection (104 points)
2. Integration of explainable AI in security decisions
3. Open framework for community improvement"

## Conclusion Paragraph

"This thesis demonstrates both the potential and challenges of applying machine learning to cryptographic algorithm selection for IoT devices. While the achieved accuracy of 66.67% indicates significant room for improvement, the work establishes important foundations: a curated dataset, an explainable ML framework, and clear identification of key decision factors. The honest reporting of these results, including limitations and failures to meet initial targets, provides valuable lessons for future research in automated security decision support systems."