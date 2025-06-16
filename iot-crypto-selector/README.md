# IoT Cryptographic Algorithm Selection Tool

An explainable decision support system for selecting lightweight cryptographic algorithms in IoT applications.

## 📊 Project Overview

This project develops a machine learning-based tool that helps IoT developers choose appropriate lightweight cryptographic algorithms based on their device constraints and requirements. The tool provides transparent explanations for its recommendations using SHAP (SHapley Additive exPlanations).

## 🎯 Results Summary

- **Dataset**: 104 high-quality data points from 9 research papers
- **Algorithms**: AES-128, SIMON, SPECK, PRESENT
- **Best Model**: Random Forest
- **Accuracy**: 66.67% (test), 53.71% ± 20.14% (cross-validation)
- **Key Finding**: Key size is the most important factor (28% importance)

## 🔧 Features

- **Algorithm Recommendation**: Suggests optimal cryptographic algorithm based on device specifications
- **Explainable AI**: SHAP-based explanations for transparency
- **Web Interface**: User-friendly input form and visualization
- **Multi-Platform Support**: Covers 8-bit MCUs, 32-bit MCUs, FPGAs, ASICs

## 📁 Repository Structure

```
iot-crypto-selector/
├── data/
│   ├── raw/
│   │   ├── final_core4_dataset.csv    # 104 data points
│   │   └── FINAL_DATASET_SUMMARY.txt   # Dataset statistics
│   └── processed/
│       ├── dataset_151_processed.csv   # Processed with features
│       ├── train_set.csv              # Training data
│       └── test_set.csv               # Test data
├── models/
│   └── final_model_151points_*.pkl    # Trained Random Forest model
├── results/
│   ├── confusion_matrix_*.png         # Model performance
│   └── shap_importance_*.png          # Feature importance
├── visualizations/                     # Additional plots
└── *.py                               # Python scripts
```

## 🚀 Quick Start

1. **Setup Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Run Analysis**
```bash
python dataset_preparation.py    # Prepare data
python complete_analysis.py      # Train model
python generate_explanations.py  # Generate SHAP explanations
```

3. **Analyze Results**
```bash
python results_analysis_script.py  # Understand the dataset
```

## 📈 Key Results

### Algorithm Distribution
- SIMON: 43 samples (41.3%)
- SPECK: 35 samples (33.7%)
- AES-128: 15 samples (14.4%)
- PRESENT: 11 samples (10.6%)

### Performance by Algorithm
| Algorithm | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| AES-128   | 1.00      | 0.33   | 0.50     | 3       |
| PRESENT   | 1.00      | 1.00   | 1.00     | 2       |
| SIMON     | 0.58      | 0.78   | 0.67     | 9       |
| SPECK     | 0.67      | 0.57   | 0.62     | 7       |

### Feature Importance (SHAP)
1. Key_Size_Bits: 28.0%
2. freq_per_kb_ram: 14.9%
3. RAM_KB: 12.9%
4. log_cpu_freq: 11.5%
5. CPU_Freq_MHz: 8.8%

## 🎓 Academic Context

This project is part of an undergraduate thesis in Computer Science. While the achieved accuracy (66.67%) is below the initial target (85%), it demonstrates:

- The feasibility of ML-based cryptographic algorithm selection
- The importance of explainable AI in security decisions
- The challenges of working with limited, imbalanced datasets
- The value of honest academic reporting

## ⚠️ Limitations

1. **Dataset Size**: 104 points (target was 150-200)
2. **Class Imbalance**: SIMON has 3.9x more samples than PRESENT
3. **Accuracy**: 66.67% indicates room for improvement
4. **Scope**: Only 4 algorithms covered

## 🔮 Future Work

1. Collect more balanced data (50+ samples per algorithm)
2. Implement advanced balancing techniques (SMOTE)
3. Include energy consumption metrics
4. Expand to more algorithms (ChaCha20, ASCON)
5. Deploy as web service

## 📚 Data Sources

Data extracted from 9 peer-reviewed papers:
- Diehl et al. (2017)
- Sleem & Couturier (2021)
- Nithya & Kumar (2016)
- Bogdanov et al. (2007)
- Beaulieu et al. (2013)
- Dinu et al. (2015)
- [Your original 3 papers]

## 🤝 Contributing

This is an academic project. Contributions in the form of:
- Additional benchmark data
- Improved ML models
- Bug fixes
- Documentation improvements

are welcome via pull requests.

## 📄 License

This project is released under the MIT License for academic and research purposes.

## 🙏 Acknowledgments

- Thesis supervisor for guidance
- Authors of the benchmark papers
- SHAP library developers
- scikit-learn community

---

**Note**: This tool is a proof-of-concept for academic purposes. For production use, additional validation and higher accuracy would be required.