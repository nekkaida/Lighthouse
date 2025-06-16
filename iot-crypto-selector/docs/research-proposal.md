# An Explainable Decision Support Tool for IoT Cryptographic Algorithm Selection

## Introduction/Problem Statement

The Internet of Things (IoT) has experienced rapid growth across various domains including smart homes, healthcare, and industrial systems. However, IoT devices face significant security challenges due to their resource-constrained nature. As Al-Husainy et al. (2021) identify, "Such devices have limited processing capabilities in terms of speed, storage, and memory," which fundamentally constrains the cryptographic options available for securing IoT communications.

The primary challenge facing IoT developers is selecting appropriate lightweight cryptographic algorithms for their specific applications. This challenge is compounded by dramatic performance variations across different algorithms and platforms. For example, Radhakrishnan et al. (2024) demonstrate that SPECK achieves 824,041 bits/s encryption throughput while ASCON achieves only 188,349 bits/s on identical hardware—a 4× performance difference.

Currently, IoT developers must manually analyze complex performance data from multiple research studies to make algorithm selection decisions. This process is time-consuming, error-prone, and often results in suboptimal choices that either compromise security or exceed device capabilities. While machine learning approaches have shown promise for cryptographic evaluation, as demonstrated by Puthiyavan and Anandan (2025), these systems operate as "black boxes" without explaining their selection rationale.

This project addresses this gap by developing an explainable decision support tool that provides IoT developers with algorithm recommendations along with clear explanations of why specific algorithms are suitable for their deployment context. The tool will focus on four widely-adopted lightweight algorithms: AES-128 (the standard), SIMON (NSA-designed for hardware efficiency), SPECK (optimized for software implementation), and PRESENT (ultra-lightweight for severely constrained devices).

## Aims and Objectives

**Primary Aim**: To develop a web-based explainable decision support tool that helps IoT developers select appropriate lightweight cryptographic algorithms with transparent selection rationale.

**Specific Objectives**:
1. Create a comprehensive dataset by extracting performance benchmarks from peer-reviewed studies, covering 4 lightweight cryptographic algorithms across multiple IoT platforms (target: 100-150 data points based on availability)
2. Implement machine learning models to predict optimal algorithm selection based on device specifications, achieving at least 65% accuracy with explainable predictions
3. Integrate SHAP (SHapley Additive exPlanations) to provide clear explanations for algorithm recommendations, identifying key decision factors
4. Develop a user-friendly web interface that allows developers to input device constraints and understand selection rationale through visualizations
5. Evaluate the tool's effectiveness through user testing with IT students from diverse specializations, measuring both prediction accuracy and explanation comprehensibility

**Algorithm Selection Approach**: The tool will recommend the algorithm that provides the best performance (execution time) while fitting within the device's memory constraints, based on patterns learned from benchmark data.

## Background Study/Literature Review

The resource-constrained nature of IoT devices creates fundamental challenges for cryptographic implementation. Al-Husainy et al. (2021) identify that conventional encryption techniques like AES, RSA, and ECC "are only suited the systems that have a reasonable capabilities in terms of power, memory, and processing compared to the IoT devices." This limitation has driven extensive research into lightweight cryptography specifically designed for resource-constrained environments.

Performance variations among lightweight cryptographic algorithms are substantial and context-dependent. Radhakrishnan et al. (2024) provide comprehensive hardware evaluation showing SPECK achieving superior throughput (824,041 bits/s) compared to ASCON (188,349 bits/s) on Arduino platforms. Similarly, memory requirements vary significantly, with different algorithms demanding vastly different resources. Through extensive benchmarking, Radhakrishnan et al. (2024) conclude that "SPECK exhibits better performance in resource-constrained IoT devices," highlighting how algorithm choice significantly impacts system performance.

The diversity of IoT applications creates complex selection challenges that extend beyond simple performance comparison. Thakor et al. (2021) provide comprehensive analysis of 41 lightweight cryptography algorithms across seven performance metrics, concluding that "none of the LWC algorithms fulfils all the criteria of hardware and software performance metrics." This finding highlights the fundamental challenge facing IoT developers: no universal solution exists, making systematic selection guidance essential.

Machine learning has emerged as a promising approach for cryptographic algorithm evaluation. Puthiyavan Udayakumar and Anandan (2025) conducted comparative studies using machine learning models on specialized hardware in medical IoT environments. Their findings show that "SVM consistently maintains high accuracy levels across all algorithms and file sizes," achieving up to 99.5% accuracy with certain configurations. This demonstrates the viability of ML-based approaches for algorithm selection tasks.

The broader field of automatic algorithm selection has established comprehensive methodologies. Simpson et al. (2016) demonstrate that machine learning provides "a promising alternative for automatic algorithm selection by easing the design process and overhead while also attaining high accuracy in selection." Their work achieved 86% accuracy in selecting optimal algorithms, validating the approach.

However, explainability remains crucial for security-critical decisions. Miller (2023) demonstrates that traditional "recommendation-driven" AI systems often lead to "over-reliance and under-reliance" problems. Miller proposes "evaluative AI" - a hypothesis-driven decision support framework where systems provide "evidence for and against decisions made by people, rather than provide recommendations to accept or reject." This approach is particularly relevant for cryptographic selection where developers need to understand and trust the rationale behind recommendations.

Lahav et al. (2019) further demonstrate that interpretability requires both comprehensibility and trustworthiness. Their research reveals a fundamental gap between what system designers believe users need and what actually builds user trust. This insight informs our approach to explanation design, ensuring that our tool provides meaningful, actionable explanations rather than just technical metrics.

Despite extensive research on lightweight cryptography performance and proven methodologies in machine learning for algorithm selection, no systematic tools exist for explainable cryptographic algorithm selection in IoT contexts. Current approaches either provide comprehensive performance analysis without selection guidance or focus on non-cryptographic domains. This gap represents the opportunity that our explainable decision support tool addresses.

## Research Methodology

The project employs a systematic approach combining data-driven machine learning with explainable AI techniques. A preliminary proof-of-concept implementation using 104 real-world data points extracted from peer-reviewed literature has demonstrated technical feasibility, achieving 66.67% test accuracy with Random Forest classification and successfully integrating SHAP for explainability.

**Phase 1: Data Collection and Dataset Creation**
The project will build upon the current dataset of 104 samples by extracting additional performance metrics from peer-reviewed papers where available. Data includes algorithm type, device specifications (CPU frequency, RAM, architecture), and performance metrics (execution time, memory usage, throughput). The preliminary dataset shows concentration in certain algorithm-platform combinations (SIMON: 41%, SPECK: 34%, AES-128: 14%, PRESENT: 11%), which will be addressed by targeted extraction focusing on underrepresented combinations.

**Phase 2: Machine Learning Model Development**
Building on the proof-of-concept implementation, the project will optimize model performance through hyperparameter tuning and exploration of ensemble methods. The preliminary analysis has identified key decision factors: key size (28% importance), frequency-per-KB-RAM ratio (15%), and RAM constraints (13%). Algorithm selection will be based on the fastest execution time that fits within the device's memory constraints. Class imbalance will be addressed through techniques such as SMOTE or class-weighted models.

**Phase 3: Web Application Development**
A three-tier architecture will be implemented: React.js frontend for user interaction, Flask REST API for model serving, and integrated SHAP explainer for real-time interpretation. Users will input their device specifications and constraints, and receive recommendations with visual explanations of the decision factors.

**Phase 4: Evaluation**
User testing with 15 students from various IT disciplines will assess the tool's effectiveness. Participants will be given three scenarios: selecting algorithms for different IoT devices with varying constraints. Success will be measured by comparing their selections (with and without the tool) against the performance data from literature. The primary metrics will be task completion time, selection accuracy, and explanation comprehensibility. Ethics approval will be sought from the institutional review board prior to conducting user studies.

## Potential Project Significance

This project addresses a critical gap in IoT security by providing developers with an evidence-based tool for cryptographic algorithm selection. The preliminary implementation has demonstrated that even with 66.67% accuracy, the tool provides significant value over random selection (25% baseline) and successfully identifies key decision factors through explainable AI.

The tool targets specific resource-constrained platforms including ATmega328P (Arduino Uno/Nano), ARM Cortex-M4 (Teensy 3.6), ESP32, and FPGA implementations, as represented in our dataset. The integration of explainable AI ensures transparency in security-critical decisions, building developer confidence and preventing suboptimal implementations that could compromise system security or performance.

By making expert knowledge accessible through automated recommendations with clear explanations, the tool democratizes secure IoT development. The open-source dataset and methodology will also contribute to the research community, enabling further advances in lightweight cryptography evaluation. The preliminary results validate the feasibility of this approach and provide a foundation for continued improvement.

## Expected Outcomes and/or Concluding Remarks

The project will deliver a functional web-based tool achieving >65% accuracy in algorithm selection (already demonstrated at 66.67% in proof-of-concept). The tool will provide transparent, SHAP-based explanations helping developers understand why specific algorithms suit their constraints. Key deliverables include an open-source dataset of IoT cryptographic performance benchmarks (currently 104 points, expandable), a trained machine learning model with explainability features, and comprehensive evaluation results from user studies.

The preliminary proof-of-concept implementation validates the technical approach, with key size identified as the dominant selection factor (28% importance) and clear platform-algorithm correlations established (e.g., 8-bit MCUs preferring SIMON, 32-bit processors favoring SPECK). This foundation ensures the project can deliver practical value to IoT developers while contributing to academic understanding of explainable AI in security contexts. The tool will reduce algorithm selection time from hours to minutes while improving decision quality through evidence-based recommendations.

**Limitations and Mitigation**: The current dataset of 104 points shows class imbalance (SIMON: 43, PRESENT: 11), which impacts model performance. This will be addressed through targeted data collection and balancing techniques. The achieved accuracy of 66.67%, while below ideal targets, still provides meaningful decision support and will be improved through model optimization. The tool focuses on execution time and memory usage as primary factors, acknowledging that specific security requirements or hardware acceleration capabilities may require additional consideration in practice.

## Key References

Al-Husainy, M. A. F., Al-Shargabi, B., & Aljawarneh, S. (2021). A flexible lightweight encryption system for IoT devices. *Computers and Electrical Engineering*, 95, 107418.

Lahav, O., Mastronarde, N., & van der Schaar, M. (2019). What is interpretable? Using machine learning to design interpretable decision-support systems. In *Machine Learning for Health (ML4H) Workshop at NeurIPS 2018*. arXiv preprint arXiv:1811.10799.

Miller, T. (2023). Explainable AI is dead, long live explainable AI! Hypothesis-driven decision support. *arXiv preprint* arXiv:2302.12389.

Puthiyavan Udayakumar, R., & Anandan, R. (2025). Comparative study of lightweight encryption algorithms leveraging neural processing unit for artificial internet of medical things. *International Journal of Computational and Experimental Science and Engineering*, 11(1), 1452-1469.

Radhakrishnan, I., Jadon, S., & Honnavalli, P. B. (2024). Efficiency and security evaluation of lightweight cryptographic algorithms for resource-constrained IoT devices. *Sensors*, 24(12), 4008.

Simpson, M. C., Yi, Q., & Kalita, J. (2016). Automatic algorithm selection in computational software using machine learning. In *2016 15th IEEE International Conference on Machine Learning and Applications (ICMLA)* (pp. 355-360). IEEE.

Thakor, V. A., Razzaque, M. A., & Khandaker, M. R. A. (2021). Lightweight cryptography algorithms for resource-constrained IoT devices: A review, comparison and research opportunities. *IEEE Access*, 9, 28177-28193.