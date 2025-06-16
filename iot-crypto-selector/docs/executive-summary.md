# Executive Summary

## Explainable Decision Support Tool for IoT Cryptographic Algorithm Selection

**Student**: [Your Name]  
**Supervisor**: [Supervisor Name]  
**Program**: Bachelor of Computer Science  
**Date**: [Current Date]

### Problem Statement
IoT developers face significant challenges selecting appropriate lightweight cryptographic algorithms for resource-constrained devices. With performance variations up to 4× between algorithms on identical hardware, wrong choices can compromise security or exceed device capabilities. Current selection processes require manual analysis of complex benchmarks across multiple research papers.

### Proposed Solution
This project develops a web-based decision support tool using machine learning and explainable AI (SHAP) to recommend optimal cryptographic algorithms based on device specifications. The tool focuses on four widely-adopted algorithms: AES-128, SIMON, SPECK, and PRESENT.

### Preliminary Results
A proof-of-concept implementation demonstrates feasibility:
- **Dataset**: 104 high-quality data points from 9 peer-reviewed papers
- **Accuracy**: 66.67% (significantly better than 25% random baseline)
- **Key Finding**: Key size is the primary decision factor (28% importance)
- **Model**: Random Forest with SHAP integration successfully implemented

### Methodology
1. **Data Collection**: Extract benchmarks from literature (target: 100-150 points)
2. **ML Development**: Optimize models addressing class imbalance
3. **Web Application**: React frontend with Flask API backend
4. **Evaluation**: User testing with 15 IT students

### Expected Outcomes
- Functional web tool with >65% accuracy (already achieved)
- Open-source dataset and codebase
- Clear identification of algorithm selection factors
- Reduced selection time from hours to minutes

### Significance
This tool democratizes secure IoT development by making expert knowledge accessible through automated, explainable recommendations. It addresses a critical gap in IoT security while contributing to academic understanding of explainable AI in security contexts.

### Timeline
15-week development schedule with key milestones:
- Weeks 1-3: Data collection and validation
- Weeks 4-6: Model optimization
- Weeks 7-9: Web application development
- Weeks 10-11: System integration
- Weeks 12-13: User evaluation
- Weeks 14-15: Documentation and submission