# Preliminary Results Summary

## Proof-of-Concept Implementation

### Dataset Characteristics
- **Total Data Points**: 104 (from 9 peer-reviewed papers)
- **Algorithms Covered**: 
  - SIMON: 43 samples (41.3%)
  - SPECK: 35 samples (33.7%)
  - AES-128: 15 samples (14.4%)
  - PRESENT: 11 samples (10.6%)
- **Platform Types**: 8-bit MCUs, 32-bit MCUs, FPGAs, ASICs, Desktop
- **Year Range**: 2007-2021

### Model Performance

#### Accuracy Results
- **Test Accuracy**: 66.67% (14/21 correct predictions)
- **Cross-Validation**: 53.71% ± 20.14%
- **Baseline (Random)**: 25%
- **Improvement over Baseline**: 2.67×

#### Per-Algorithm Performance
| Algorithm | Precision | Recall | F1-Score | Correct/Total |
|-----------|-----------|--------|----------|---------------|
| PRESENT   | 100%      | 100%   | 1.00     | 2/2           |
| SIMON     | 58%       | 78%    | 0.67     | 7/9           |
| SPECK     | 67%       | 57%    | 0.62     | 4/7           |
| AES-128   | 100%      | 33%    | 0.50     | 1/3           |

### Key Findings

#### Feature Importance (SHAP Analysis)
1. **Key_Size_Bits**: 28.0% (most important)
2. **freq_per_kb_ram**: 14.9%
3. **RAM_KB**: 12.9%
4. **log_cpu_freq**: 11.5%
5. **CPU_Freq_MHz**: 8.8%

#### Algorithm Selection Patterns
- **8-bit MCUs** → Prefer SIMON (memory efficient)
- **32-bit MCUs** → Prefer SPECK (speed optimized)
- **FPGAs** → Mixed preferences based on requirements
- **Low memory** → Avoid AES-128

### Technical Implementation

#### Technologies Used
- **ML Framework**: scikit-learn 1.3+
- **Explainability**: SHAP 0.42+
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

#### Code Structure
```
models/
├── Random Forest (primary)
├── SVM (comparison)
└── Ensemble methods (tested)

features/
├── Basic (device specs)
├── Engineered (ratios, logs)
└── Platform indicators
```

### Challenges Identified

1. **Class Imbalance**: 3.9:1 ratio (SIMON:PRESENT)
2. **Missing Data**: 37% of RAM values required imputation
3. **Small Dataset**: Average 26 samples per class
4. **High Variance**: ±20.14% in cross-validation

### Validation of Approach

Despite limitations, the proof-of-concept demonstrates:
- ✅ ML can learn algorithm selection patterns
- ✅ SHAP provides meaningful explanations
- ✅ 66.67% accuracy is useful for decision support
- ✅ Web integration is technically feasible

### Next Steps

1. **Immediate**: Address class imbalance with SMOTE
2. **Short-term**: Collect 20+ more samples for minority classes
3. **Long-term**: Implement confidence intervals for predictions

This preliminary implementation validates the feasibility of the proposed approach and provides a solid foundation for the full project development.