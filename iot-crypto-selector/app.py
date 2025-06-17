#!/usr/bin/env python3
"""
app.py
Flask web application for IoT Cryptographic Algorithm Selection Tool
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import shap
import json

app = Flask(__name__)

# Load the trained model - use the actual filename from your models directory
import glob
import os

# Find the most recent model file
model_files = glob.glob('models/final_model_*.pkl')
if model_files:
    # Sort by modification time and get the most recent
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Loading model: {latest_model}")
    model = joblib.load(latest_model)
else:
    # Fallback to a specific filename if needed
    model = joblib.load('models/final_model_20250617_105537.pkl')

explainer = shap.TreeExplainer(model)

# Feature names
feature_names = [
    'CPU_Freq_MHz', 'RAM_KB', 'Key_Size_Bits', 'Data_Size_Bytes',
    'log_cpu_freq', 'is_low_memory', 'freq_per_kb_ram', 'key_data_ratio',
    'is_8bit', 'is_32bit', 'is_fpga', 'is_asic'
]

def prepare_input(form_data):
    """Prepare user input for model prediction"""
    
    # Extract basic features
    cpu_freq = float(form_data['cpu_freq'])
    ram_kb = float(form_data['ram_kb'])
    key_size = int(form_data['key_size'])
    data_size = int(form_data.get('data_size', 128))  # Default 128 bytes
    device_type = form_data['device_type']
    
    # Engineer features
    log_cpu_freq = np.log1p(cpu_freq)
    is_low_memory = 1 if ram_kb < 1.0 else 0
    freq_per_kb_ram = cpu_freq / (ram_kb + 0.001)
    key_data_ratio = key_size / (data_size * 8)
    
    # Device type flags
    is_8bit = 1 if device_type == '8bit' else 0
    is_32bit = 1 if device_type == '32bit' else 0
    is_fpga = 1 if device_type == 'fpga' else 0
    is_asic = 1 if device_type == 'asic' else 0
    
    # Create feature vector
    features = pd.DataFrame([[
        cpu_freq, ram_kb, key_size, data_size,
        log_cpu_freq, is_low_memory, freq_per_kb_ram, key_data_ratio,
        is_8bit, is_32bit, is_fpga, is_asic
    ]], columns=feature_names)
    
    return features

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction and return results with explanations"""
    
    try:
        # Get form data
        form_data = request.json
        
        # Prepare input
        features = prepare_input(form_data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get algorithm probabilities
        algorithms = model.classes_
        algo_probs = {algo: float(prob) for algo, prob in zip(algorithms, probabilities)}
        
        # Sort by probability
        sorted_algos = sorted(algo_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Get SHAP values
        shap_values = explainer.shap_values(features)
        
        # Prepare explanation
        if isinstance(shap_values, list):
            # Multi-class: get values for predicted class
            class_idx = list(algorithms).index(prediction)
            shap_vals = shap_values[class_idx][0]
        else:
            shap_vals = shap_values[0]
        
        # Get top contributing features
        feature_impacts = []
        for i, (feat, val) in enumerate(zip(feature_names, shap_vals)):
            if abs(val) > 0.01:  # Only significant impacts
                feature_impacts.append({
                    'feature': feat,
                    'impact': float(val),
                    'value': float(features[feat].iloc[0])
                })
        
        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        # Prepare response
        response = {
            'success': True,
            'recommendation': prediction,
            'confidence': float(max(probabilities)),
            'algorithms': sorted_algos,
            'explanation': {
                'top_features': feature_impacts[:5],
                'reasoning': generate_explanation(prediction, feature_impacts, form_data)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def generate_explanation(algorithm, feature_impacts, form_data):
    """Generate human-readable explanation"""
    
    cpu_freq = float(form_data['cpu_freq'])
    ram_kb = float(form_data['ram_kb'])
    
    explanations = {
        'SPECK': f"SPECK is recommended because your device has {'limited' if ram_kb < 10 else 'sufficient'} memory ({ram_kb:.1f} KB) and SPECK provides the fastest encryption speed for these constraints.",
        'SIMON': f"SIMON is recommended because your device has very limited memory ({ram_kb:.1f} KB) and SIMON has the smallest memory footprint while maintaining good security.",
        'PRESENT': f"PRESENT is recommended for your extremely constrained device ({ram_kb:.1f} KB RAM, {cpu_freq:.1f} MHz) as it's specifically designed for ultra-lightweight applications.",
        'AES-128': f"AES-128 is recommended because your device has sufficient resources ({ram_kb:.1f} KB RAM, {cpu_freq:.1f} MHz) to run the industry standard algorithm efficiently."
    }
    
    base_explanation = explanations.get(algorithm, "Algorithm recommended based on device constraints.")
    
    # Add top feature impacts
    if feature_impacts:
        top_feature = feature_impacts[0]
        if top_feature['feature'] == 'Key_Size_Bits':
            base_explanation += f" The {top_feature['value']}-bit key size was a major factor in this recommendation."
        elif top_feature['feature'] == 'freq_per_kb_ram':
            base_explanation += f" The balance between processing speed and memory was crucial for this selection."
    
    return base_explanation

@app.route('/api/device_profiles')
def device_profiles():
    """Return common device profiles for quick selection"""
    
    profiles = {
        'arduino_uno': {
            'name': 'Arduino Uno',
            'cpu_freq': 16,
            'ram_kb': 2,
            'device_type': '8bit'
        },
        'esp8266': {
            'name': 'ESP8266',
            'cpu_freq': 80,
            'ram_kb': 80,
            'device_type': '32bit'
        },
        'esp32': {
            'name': 'ESP32',
            'cpu_freq': 240,
            'ram_kb': 320,
            'device_type': '32bit'
        },
        'arm_cortex_m4': {
            'name': 'ARM Cortex-M4',
            'cpu_freq': 180,
            'ram_kb': 256,
            'device_type': '32bit'
        }
    }
    
    return jsonify(profiles)

if __name__ == '__main__':
    app.run(debug=True)

# === HTML Template (save as templates/index.html) ===
"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IoT Cryptographic Algorithm Selection Tool</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-sizing: border-box;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
        }
        button:hover {
            background-color: #45a049;
        }
        .results {
            margin-top: 30px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
            display: none;
        }
        .recommendation {
            font-size: 24px;
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .confidence {
            font-size: 18px;
            color: #666;
            margin-bottom: 20px;
        }
        .explanation {
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f4fd;
            border-left: 4px solid #2196F3;
            border-radius: 5px;
        }
        .algorithm-list {
            margin-top: 20px;
        }
        .algorithm-item {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            margin-bottom: 5px;
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .progress-bar {
            width: 200px;
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
        }
        .progress {
            height: 100%;
            background-color: #4CAF50;
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>IoT Cryptographic Algorithm Selection Tool</h1>
        
        <form id="deviceForm">
            <div class="form-group">
                <label for="device_profile">Quick Select Device Profile:</label>
                <select id="device_profile">
                    <option value="">Custom Device</option>
                    <option value="arduino_uno">Arduino Uno</option>
                    <option value="esp8266">ESP8266</option>
                    <option value="esp32">ESP32</option>
                    <option value="arm_cortex_m4">ARM Cortex-M4</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="cpu_freq">CPU Frequency (MHz):</label>
                <input type="number" id="cpu_freq" name="cpu_freq" required step="0.1" value="16">
            </div>
            
            <div class="form-group">
                <label for="ram_kb">RAM (KB):</label>
                <input type="number" id="ram_kb" name="ram_kb" required step="0.001" value="2">
            </div>
            
            <div class="form-group">
                <label for="key_size">Encryption Key Size (bits):</label>
                <select id="key_size" name="key_size" required>
                    <option value="64">64-bit (Low Security)</option>
                    <option value="80">80-bit (Moderate Security)</option>
                    <option value="96">96-bit (Good Security)</option>
                    <option value="128" selected>128-bit (High Security)</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="device_type">Device Architecture:</label>
                <select id="device_type" name="device_type" required>
                    <option value="8bit">8-bit Microcontroller</option>
                    <option value="32bit">32-bit Processor</option>
                    <option value="fpga">FPGA</option>
                    <option value="asic">ASIC</option>
                </select>
            </div>
            
            <button type="submit">Get Recommendation</button>
        </form>
        
        <div id="results" class="results">
            <div class="recommendation" id="recommendation"></div>
            <div class="confidence" id="confidence"></div>
            <div class="explanation" id="explanation"></div>
            <div class="algorithm-list" id="algorithmList"></div>
        </div>
    </div>
    
    <script>
        // Load device profiles
        fetch('/api/device_profiles')
            .then(response => response.json())
            .then(profiles => {
                document.getElementById('device_profile').addEventListener('change', function() {
                    const profile = profiles[this.value];
                    if (profile) {
                        document.getElementById('cpu_freq').value = profile.cpu_freq;
                        document.getElementById('ram_kb').value = profile.ram_kb;
                        document.getElementById('device_type').value = profile.device_type;
                    }
                });
            });
        
        // Handle form submission
        document.getElementById('deviceForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = {
                cpu_freq: document.getElementById('cpu_freq').value,
                ram_kb: document.getElementById('ram_kb').value,
                key_size: document.getElementById('key_size').value,
                device_type: document.getElementById('device_type').value
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Show results
                    document.getElementById('results').style.display = 'block';
                    document.getElementById('recommendation').textContent = 
                        `Recommended Algorithm: ${result.recommendation}`;
                    document.getElementById('confidence').textContent = 
                        `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
                    document.getElementById('explanation').textContent = 
                        result.explanation.reasoning;
                    
                    // Show all algorithms with probabilities
                    const algorithmList = document.getElementById('algorithmList');
                    algorithmList.innerHTML = '<h3>All Algorithm Scores:</h3>';
                    
                    result.algorithms.forEach(([algo, prob]) => {
                        const item = document.createElement('div');
                        item.className = 'algorithm-item';
                        item.innerHTML = `
                            <span>${algo}</span>
                            <div class="progress-bar">
                                <div class="progress" style="width: ${prob * 100}%"></div>
                            </div>
                            <span>${(prob * 100).toFixed(1)}%</span>
                        `;
                        algorithmList.appendChild(item);
                    });
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error making prediction: ' + error);
            }
        });
    </script>
</body>
</html>
"""