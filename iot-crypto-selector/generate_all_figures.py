#!/usr/bin/env python3
"""
generate_all_figures.py
Generate all figures required for the thesis proposal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_figure_1_literature_comparison():
    """Figure 1: Literature Review Summary - Comparison table"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Data for comparison table
    data = [
        ['Approach', 'Time Required', 'Accuracy', 'Explanations', 'Accessibility'],
        ['Manual Search', '2-3 hours', '~35%', 'None', 'Requires expertise'],
        ['Research Papers', '1-2 hours', 'Variable', 'Technical only', 'Very technical'],
        ['Manufacturer Docs', '30-60 min', 'Biased', 'Limited', 'Product-specific'],
        ['Online Forums', '1-2 hours', 'Inconsistent', 'Anecdotal', 'Variable quality'],
        ['Our Tool', '<5 minutes', '56%', 'AI-powered with CI', 'User-friendly']
    ]
    
    # Create table
    table = ax.table(cellText=data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.15, 0.15, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight our tool row
    for i in range(5):
        table[(5, i)].set_facecolor('#90EE90')
    
    plt.title('Figure 1: Comparison of Existing Approaches vs. Our Solution', 
              fontsize=14, weight='bold', pad=20)
    plt.savefig('figures/figure_1_literature_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_figure_2_methodology_flowchart():
    """Figure 2: Methodology Flowchart"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define boxes
    boxes = [
        # Phase 1
        {'xy': (2, 8), 'width': 3, 'height': 1.2, 'text': 'Phase 1: Data Collection\n450 papers → 38 usable\n97 data points', 'color': '#e8f5e9'},
        {'xy': (2, 6.5), 'width': 3, 'height': 0.8, 'text': 'Inter-rater reliability\nκ = 0.87', 'color': '#e3f2fd'},
        
        # Phase 2
        {'xy': (6, 8), 'width': 3, 'height': 1.2, 'text': 'Phase 2: ML Development\n5 algorithms tested\nRF selected: 56%', 'color': '#fff3e0'},
        {'xy': (6, 6.5), 'width': 3, 'height': 0.8, 'text': 'Hyperparameter tuning\n51% → 56%', 'color': '#fce4ec'},
        
        # Phase 3
        {'xy': (2, 4), 'width': 3, 'height': 1.2, 'text': 'Phase 3: Web App\nFlask + React\nSHAP explanations', 'color': '#f3e5f5'},
        {'xy': (2, 2.5), 'width': 3, 'height': 0.8, 'text': 'Confidence intervals\nWilson score method', 'color': '#e8eaf6'},
        
        # Phase 4
        {'xy': (6, 4), 'width': 3, 'height': 1.2, 'text': 'Phase 4: Evaluation\n15 IT students\n3 scenarios', 'color': '#e0f2f1'},
        {'xy': (6, 2.5), 'width': 3, 'height': 0.8, 'text': 'Baseline: 112.5 min\nTool: 3.5 min', 'color': '#fbe9e7'},
        
        # Central result
        {'xy': (4, 0.5), 'width': 4, 'height': 1, 'text': 'Result: 2.24× improvement\np < 0.001', 'color': '#c8e6c9'}
    ]
    
    # Draw boxes
    for box in boxes:
        rect = plt.Rectangle(box['xy'], box['width'], box['height'], 
                           facecolor=box['color'], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(box['xy'][0] + box['width']/2, box['xy'][1] + box['height']/2, 
                box['text'], ha='center', va='center', fontsize=11, weight='bold')
    
    # Draw arrows
    arrows = [
        # Horizontal arrows
        ((5, 8.6), (6, 8.6)),
        ((5, 7.1), (6, 7.1)),
        # Vertical arrows
        ((3.5, 6.5), (3.5, 5.2)),
        ((7.5, 6.5), (7.5, 5.2)),
        # To result
        ((3.5, 2.5), (5, 1.5)),
        ((7.5, 2.5), (7, 1.5))
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkgray'))
    
    plt.title('Figure 2: Research Methodology Flowchart', fontsize=16, weight='bold', y=0.98)
    plt.savefig('figures/figure_2_methodology_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_figure_3_confusion_matrix():
    """Figure 3: Results Summary - Confusion Matrix"""
    
    # Actual confusion matrix from results
    y_true = ['SIMON'] * 10 + ['SPECK'] * 8 + ['AES-128'] * 4 + ['PRESENT'] * 3
    y_pred = ['SIMON'] * 5 + ['SPECK'] * 3 + ['AES-128'] * 1 + ['PRESENT'] * 1 + \
             ['SPECK'] * 3 + ['SIMON'] * 3 + ['AES-128'] * 1 + ['PRESENT'] * 1 + \
             ['AES-128'] * 3 + ['SIMON'] * 1 + \
             ['PRESENT'] * 3
    
    cm = confusion_matrix(y_true, y_pred, labels=['AES-128', 'PRESENT', 'SIMON', 'SPECK'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['AES-128', 'PRESENT', 'SIMON', 'SPECK'],
                yticklabels=['AES-128', 'PRESENT', 'SIMON', 'SPECK'],
                cbar_kws={'label': 'Count'})
    
    # Add percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j+0.5, i+0.7, f'({cm_percent[i,j]:.0%})', 
                       ha='center', va='center', fontsize=10, color='gray')
    
    plt.title('Figure 3: Confusion Matrix - Model Performance on Test Set\nOverall Accuracy: 56% (14/25 correct predictions)', 
              fontsize=14, weight='bold', pad=20)
    plt.ylabel('True Algorithm', fontsize=12)
    plt.xlabel('Predicted Algorithm', fontsize=12)
    
    # Add performance summary box
    textstr = 'Performance by Algorithm:\n' \
              'PRESENT: 100% recall (3/3)\n' \
              'AES-128: 75% recall (3/4)\n' \
              'SIMON: 50% recall (5/10)\n' \
              'SPECK: 38% recall (3/8)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.02, 0.5, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=props)
    
    plt.tight_layout()
    plt.savefig('figures/figure_3_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_figure_4_feature_importance():
    """Figure 4: Feature Importance Chart"""
    
    # Feature importance data from results
    features = ['Key_Size_Bits', 'RAM_KB', 'freq_per_kb_ram', 'key_data_ratio', 
                'CPU_Freq_MHz', 'Data_Size_Bytes', 'log_cpu_freq', 'is_8bit', 
                'is_asic', 'is_low_memory', 'is_fpga', 'is_32bit']
    importance = [0.3142, 0.1475, 0.1462, 0.1165, 0.0800, 0.0723, 0.0721, 
                  0.0160, 0.0138, 0.0076, 0.0074, 0.0063]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Bar chart
    y_pos = np.arange(len(features))
    bars = ax1.barh(y_pos, importance, color=plt.cm.viridis(np.array(importance)/max(importance)))
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importance)):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.1%}', ha='left', va='center')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features)
    ax1.set_xlabel('Importance Score', fontsize=12)
    ax1.set_title('Feature Importance from Random Forest', fontsize=14, weight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Right plot: Pie chart of top 5
    top_5_features = features[:5]
    top_5_importance = importance[:5]
    other_importance = sum(importance[5:])
    
    pie_labels = top_5_features + ['Others']
    pie_sizes = top_5_importance + [other_importance]
    colors = plt.cm.Set3(np.arange(len(pie_labels)))
    
    wedges, texts, autotexts = ax2.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%',
                                        colors=colors, startangle=90)
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_weight('bold')
        autotext.set_color('white')
    
    ax2.set_title('Top 5 Features Contribution', fontsize=14, weight='bold')
    
    # Main title
    fig.suptitle('Figure 4: Feature Importance Analysis - Key Decision Factors', 
                 fontsize=16, weight='bold', y=0.98)
    
    # Add interpretation box
    textstr = 'Key Insights:\n' \
              '• Key size dominates (31.4%) - different algorithms support different key sizes\n' \
              '• Memory constraints crucial (RAM: 14.8%, freq/RAM ratio: 14.6%)\n' \
              '• Device architecture has minimal impact (<2% each for is_8bit, is_fpga, etc.)'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    fig.text(0.5, 0.02, textstr, ha='center', fontsize=11, bbox=props)
    
    plt.tight_layout()
    plt.savefig('figures/figure_4_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_figure_5_shap_example():
    """Figure 5: SHAP Explanation Example (Mock-up)"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top panel: Device specs and recommendation
    ax1.axis('off')
    
    # Device specs box
    device_text = "Device Specifications:\n" \
                  "• Arduino Uno (ATmega328P)\n" \
                  "• CPU: 16 MHz\n" \
                  "• RAM: 2 KB\n" \
                  "• Key Size: 128-bit\n" \
                  "• Architecture: 8-bit"
    
    recommendation_text = "Recommendation: SIMON\n" \
                         "Confidence: 65.0%\n" \
                         "95% CI: [52.1%, 76.3%]"
    
    ax1.text(0.25, 0.5, device_text, transform=ax1.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            ha='center', va='center')
    
    ax1.text(0.75, 0.5, recommendation_text, transform=ax1.transAxes, fontsize=14,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            ha='center', va='center', weight='bold')
    
    ax1.set_title('Example Device Analysis', fontsize=16, weight='bold', pad=20)
    
    # Bottom panel: SHAP visualization
    features = ['Key_Size_Bits = 128', 'RAM_KB = 2.0', 'freq_per_kb_ram = 8.0',
                'is_8bit = 1', 'CPU_Freq_MHz = 16', 'is_low_memory = 0']
    shap_values = [0.15, 0.12, -0.08, 0.05, -0.03, 0.02]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(features))
    colors = ['red' if x < 0 else 'blue' for x in shap_values]
    
    bars = ax2.barh(y_pos, shap_values, color=colors, alpha=0.7)
    
    # Add value labels
    for bar, val in zip(bars, shap_values):
        x_pos = val + (0.01 if val > 0 else -0.01)
        ha = 'left' if val > 0 else 'right'
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', ha=ha, va='center', fontsize=10)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(features)
    ax2.set_xlabel('SHAP Value (impact on SIMON prediction)', fontsize=12)
    ax2.set_title('Feature Contributions to Prediction', fontsize=14, weight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add explanation text
    explanation = "Explanation: SIMON is recommended because your device has limited memory (2 KB) " \
                  "and SIMON has the smallest memory footprint. The 128-bit key size and low RAM " \
                  "are the main factors supporting this choice."
    
    ax2.text(0.5, -0.15, explanation, transform=ax2.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            ha='center', va='top', wrap=True)
    
    # Main title
    fig.suptitle('Figure 5: SHAP Explanation Example - How the Model Makes Decisions', 
                 fontsize=16, weight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('figures/figure_5_shap_example.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_all_figures():
    """Generate all required figures"""
    
    print("Generating Figure 1: Literature Review Comparison...")
    create_figure_1_literature_comparison()
    
    print("Generating Figure 2: Methodology Flowchart...")
    create_figure_2_methodology_flowchart()
    
    print("Generating Figure 3: Confusion Matrix...")
    create_figure_3_confusion_matrix()
    
    print("Generating Figure 4: Feature Importance...")
    create_figure_4_feature_importance()
    
    print("Generating Figure 5: SHAP Example...")
    create_figure_5_shap_example()
    
    print("\nAll figures generated successfully in 'figures/' directory!")
    
    # Create additional summary figure
    create_summary_infographic()

def create_summary_infographic():
    """Create a summary infographic for presentations"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Title
    fig.suptitle('IoT Cryptographic Algorithm Selection Tool - Project Summary', 
                 fontsize=20, weight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Problem statement
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axis('off')
    problem_text = "PROBLEM\n\n" \
                   "• IoT devices need encryption\n" \
                   "• Limited resources (2KB RAM vs 8GB)\n" \
                   "• Manual selection takes 2-3 hours\n" \
                   "• Requires crypto expertise"
    ax1.text(0.5, 0.5, problem_text, transform=ax1.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='#ffcdd2', alpha=0.8),
            ha='center', va='center')
    
    # Solution
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.axis('off')
    solution_text = "SOLUTION\n\n" \
                    "• ML-powered recommendations\n" \
                    "• Explains decisions with SHAP\n" \
                    "• <5 minutes to get results\n" \
                    "• User-friendly web interface"
    ax2.text(0.5, 0.5, solution_text, transform=ax2.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='#c8e6c9', alpha=0.8),
            ha='center', va='center')
    
    # Key metrics
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    metrics = [
        {'value': '56%', 'label': 'Accuracy', 'detail': '2.24× baseline'},
        {'value': '97', 'label': 'Samples', 'detail': '4 algorithms'},
        {'value': '96.9%', 'label': 'Time Saved', 'detail': '112.5→3.5 min'},
        {'value': 'p<0.001', 'label': 'Significant', 'detail': '78% power'}
    ]
    
    for i, metric in enumerate(metrics):
        x = 0.125 + i * 0.25
        # Value circle
        circle = plt.Circle((x, 0.6), 0.08, color='#2196f3', alpha=0.7)
        ax3.add_patch(circle)
        ax3.text(x, 0.6, metric['value'], ha='center', va='center', 
                fontsize=16, weight='bold', color='white')
        # Label
        ax3.text(x, 0.35, metric['label'], ha='center', va='center', 
                fontsize=12, weight='bold')
        # Detail
        ax3.text(x, 0.2, metric['detail'], ha='center', va='center', 
                fontsize=10, style='italic')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # Technical details
    ax4 = fig.add_subplot(gs[2, :2])
    ax4.axis('off')
    tech_text = "TECHNICAL APPROACH\n\n" \
                "• Random Forest (best of 5 tested)\n" \
                "• Wilson confidence intervals\n" \
                "• Inter-rater reliability κ=0.87\n" \
                "• 10-fold cross-validation"
    ax4.text(0.5, 0.5, tech_text, transform=ax4.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#e1bee7', alpha=0.8),
            ha='center', va='center')
    
    # Impact
    ax5 = fig.add_subplot(gs[2, 2:])
    ax5.axis('off')
    impact_text = "PROJECT IMPACT\n\n" \
                  "• First curated IoT crypto dataset\n" \
                  "• Open-source tool for developers\n" \
                  "• Demonstrates XAI for security\n" \
                  "• Foundation for future research"
    ax5.text(0.5, 0.5, impact_text, transform=ax5.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#fff9c4', alpha=0.8),
            ha='center', va='center')
    
    plt.savefig('figures/project_summary_infographic.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create figures directory if it doesn't exist
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Generate all figures
    create_all_figures()