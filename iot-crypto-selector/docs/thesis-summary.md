# IoT Crypto Algorithm Selection Tool - Updated Results Summary

## Executive Summary

This thesis presents an explainable decision support tool for IoT cryptographic algorithm selection, achieving 66.67% accuracy with a dataset of 104 data points across four lightweight algorithms.

## Key Results

### Dataset Statistics
- **Total Data Points**: 104 (after deduplication and Core-4 filtering)
- **Original Goal**: 150-200 points
- **Achievement**: 52-69% of target

### Algorithm Distribution
- SIMON: 43 points (41.3%)
- SPECK: 35 points (33.7%)
- AES-128: 15 points (14.4%)
- PRESENT: 11 points (10.6%)

### Model Performance
- **Best Model**: Random Forest with enhanced features
- **Test Accuracy**: 66.67%
- **Cross-Validation**: 53.71% (±20.14%)
- **Original Target**: 85%
- **Achievement**: 78.4% of target accuracy

### Feature Importance (SHAP Analysis)
1. **Key_Size_Bits**: 28.0% (down from 70% in pilot study)
2. **freq_per_kb_ram**: 14.9% (new engineered feature)
3. **RAM_KB**: 12.9%
4. **log_cpu_freq**: 11.5%
5. **CPU_Freq_MHz**: 8.8%

### Platform Coverage
- 8-bit MCUs: ~40%
- 32-bit MCUs: ~25%
- FPGA: ~20%
- ASIC: ~10%
- Desktop: ~5%

## Comparison: Pilot Study vs Final Results

| Metric | Pilot (61 points) | Final (104 points) | Change |
|--------|-------------------|-------------------|---------|
| Dataset Size | 61 | 104 | +70.5% |
| Accuracy | 85.71% | 66.67% | -22.3% |
| CV Score | Not reported | 53.71% | N/A |
| Best Model | SVM | Random Forest | Changed |
| Key_Size importance | 70% | 28% | -60% |

## Key Findings

### 1. **Algorithm Selection Patterns**
- **SIMON**: Preferred for 8-bit MCUs with limited memory
- **SPECK**: Optimal for 32-bit processors and FPGAs requiring speed
- **AES-128**: Selected for high-security requirements despite overhead
- **PRESENT**: Chosen for ultra-low resource environments

### 2. **Decision Factors**
- Key size remains the most important single factor
- Memory constraints (freq_per_kb_ram) emerged as critical
- Platform type has moderate influence
- Multiple factors interact in complex ways

### 3. **Model Insights**
- Random Forest handles non-linear relationships better than SVM
- Class imbalance significantly impacts performance
- High variance suggests need for more data

## Limitations

1. **Dataset Limitations**
   - Significant class imbalance (SIMON/SPECK dominate)
   - Limited AES-128 (15) and PRESENT (11) samples
   - Missing data required imputation

2. **Model Limitations**
   - Lower accuracy than target (66.67% vs 85%)
   - High variance in cross-validation (±20.14%)
   - Struggles with minority classes (AES-128: 33% recall)

3. **Scope Limitations**
   - Only 4 algorithms (excluded LED, TWINE, others)
   - Limited to published benchmark data
   - May not reflect real-world implementation variations

## Achievements

Despite limitations, the project successfully:

1. ✅ Created a functional web-based decision support tool
2. ✅ Integrated explainable AI (SHAP) for transparency
3. ✅ Identified key decision factors for algorithm selection
4. ✅ Demonstrated feasibility of ML-based approach
5. ✅ Collected and cleaned 104 high-quality data points
6. ✅ Provided open-source implementation

## Future Work

1. **Data Collection**
   - Target 50+ samples per algorithm
   - Include more diverse platforms
   - Add real-world performance measurements

2. **Model Improvements**
   - Ensemble methods with better balancing
   - Deep learning with data augmentation
   - Multi-objective optimization

3. **Tool Enhancements**
   - Real-time benchmarking integration
   - Hardware-in-the-loop testing
   - Energy consumption predictions

## Conclusion

While the final accuracy (66.67%) falls short of the initial target (85%), this thesis demonstrates the viability of using machine learning with explainable AI for cryptographic algorithm selection in IoT contexts. The tool provides valuable insights into decision factors and serves as a foundation for future improvements in automated security decision support systems.

The honest reporting of actual results (104 points, 66.67% accuracy) rather than inflated claims strengthens the academic integrity of this work and provides a realistic baseline for future research.