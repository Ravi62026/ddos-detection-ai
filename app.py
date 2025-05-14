"""
Streamlit App for DDoS Detection using XGBoost Model

This app allows users to input network flowbenigm label and ddos 2 things

on whether the traffic is BENIGN or a DDoS attack.
"""

import streamlit as st
import numpy as np
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64

# Set page config
st.set_page_config(
    page_title="DDoS Detection App",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to make the app more professional
def add_custom_css():
    st.markdown("""
    <style>
    /* Main page background and text */
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Arial', sans-serif;
        color: #0f5997;
    }
    
    h1 {
        font-weight: 700;
        border-bottom: 2px solid #0f5997;
        padding-bottom: 10px;
        margin-bottom: 30px;
    }
    
    /* Cards for sections */
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #0f5997;
        color: white;
        font-weight: 500;
        border-radius: 5px;
        border: none;
        padding: 10px 15px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #0c4e83;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        border-radius: 5px;
        border: 1px solid #ced4da;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #0f5997;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        font-size: 14px;
        color: #6c757d;
        border-top: 1px solid #e9ecef;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #e9ecef;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0f5997 !important;
        color: white !important;
    }
    
    /* Prediction result section */
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    
    .benign-prediction {
        background-color: rgba(40, 167, 69, 0.2);
        border: 2px solid #28a745;
    }
    
    .ddos-prediction {
        background-color: rgba(220, 53, 69, 0.2);
        border: 2px solid #dc3545;
    }
    
    /* Progress bars */
    .custom-progress-bar {
        height: 25px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Add the custom CSS
add_custom_css()

# Function to generate a professional header with logo
def add_header():
    # Create columns for logo and title
    col1, col2 = st.columns([1, 5])
    
    with col1:
        # Add a shield icon for security
        st.markdown('<i class="fas fa-shield-alt fa-4x" style="color: #0f5997;"></i>', unsafe_allow_html=True)
        
    with col2:
        st.title("🛡️ DDoS Attack Detection System")
        st.markdown("""
        <p style="font-size: 18px; color: #495057;">
        An advanced machine learning solution for real-time detection of DDoS attacks using XGBoost
        </p>
        """, unsafe_allow_html=True)
    
    # Add a professional banner/separator
    st.markdown("""
    <div style="background-color: #0f5997; height: 5px; margin-bottom: 30px;"></div>
    """, unsafe_allow_html=True)

# Define explanations for each field
FIELD_EXPLANATIONS = {
    " Flow Duration": "The duration of the flow in microseconds. Longer durations may indicate different types of traffic.",
    " Total Fwd Packets": "Total number of packets in the forward direction. Can indicate the volume of requests.",
    " Total Backward Packets": "Total number of packets in the backward direction. Can indicate response volume.",
    "Total Length of Fwd Packets": "Sum of the length of all packets in forward direction in bytes.",
    " Total Length of Bwd Packets": "Sum of the length of all packets in backward direction in bytes.",
    " Fwd Packet Length Max": "Maximum size of packet in forward direction in bytes.",
    " Fwd Packet Length Min": "Minimum size of packet in forward direction in bytes.",
    " Fwd Packet Length Mean": "Average size of packet in forward direction in bytes.",
    " Fwd Packet Length Std": "Standard deviation of packet size in forward direction.",
    "Bwd Packet Length Max": "Maximum size of packet in backward direction in bytes.",
    " Bwd Packet Length Min": "Minimum size of packet in backward direction in bytes.",
    " Bwd Packet Length Mean": "Average size of packet in backward direction in bytes.",
    " Bwd Packet Length Std": "Standard deviation of packet size in backward direction.",
    " Flow IAT Mean": "Average Inter-Arrival Time between packets in the flow (all directions).",
    " Flow IAT Std": "Standard deviation of Inter-Arrival Time between packets in the flow.",
    " Flow IAT Max": "Maximum Inter-Arrival Time between packets in the flow.",
    " Flow IAT Min": "Minimum Inter-Arrival Time between packets in the flow.",
    "Fwd IAT Total": "Total time between two packets sent in the forward direction.",
    " Fwd IAT Mean": "Mean time between two packets sent in the forward direction.",
    " Fwd IAT Std": "Standard deviation time between two packets sent in the forward direction.",
    " Fwd IAT Max": "Maximum time between two packets sent in the forward direction.",
    " Fwd IAT Min": "Minimum time between two packets sent in the forward direction.",
    "Bwd IAT Total": "Total time between two packets sent in the backward direction.",
    " Bwd IAT Mean": "Mean time between two packets sent in the backward direction.",
    " Bwd IAT Std": "Standard deviation of time between two packets sent in the backward direction.",
    " Bwd IAT Max": "Maximum time between two packets sent in the backward direction.",
    " Bwd IAT Min": "Minimum time between two packets sent in the backward direction.",
    " SYN Flag Count": "Number of packets with SYN flag set. SYN flags are used to initiate TCP connections.",
    " ACK Flag Count": "Number of packets with ACK flag set. ACK flags acknowledge receipt of previous packets.",
    " PSH Flag Count": "Number of packets with PSH flag set. PSH flags ask TCP to pass the data to the application ASAP."
}

# Function to load the model and encoder
@st.cache_resource
def load_model():
    try:
        with open('xgboost_model.pkl', 'rb') as file:
            model_data = pickle.load(file)
        
        model = model_data['model']
        encoder = model_data['label_encoder']
        return model, encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to make prediction
def predict(features, model, encoder):
    # Convert to numpy array
    features_array = np.array(features).reshape(1, -1)
    
    # Make prediction
    numeric_prediction = model.predict(features_array)[0]
    probabilities = model.predict_proba(features_array)[0]
    
    # Convert back to original label
    prediction = encoder.inverse_transform([numeric_prediction])[0]
    
    # Get probability values for each class
    prob_dict = {
        encoder.inverse_transform([i])[0]: float(prob) 
        for i, prob in enumerate(probabilities)
    }
    
    return prediction, prob_dict

# Function to load sample data
def load_sample_data():
    try:
        with open('dataset.json', 'r') as file:
            data = json.load(file)
            # Get first few samples of each class
            benign_sample = None
            ddos_sample = None
            
            for item in data:
                if item['label'] == 'BENIGN' and benign_sample is None:
                    benign_sample = item
                elif item['label'] == 'DDoS' and ddos_sample is None:
                    ddos_sample = item
                    
                if benign_sample and ddos_sample:
                    break
                    
            return benign_sample, ddos_sample
    except FileNotFoundError:
        st.warning("Sample data file 'dataset.json' not found in the current directory. Sample data tab will be disabled.")
        return None, None
    except json.JSONDecodeError:
        st.warning("Error parsing the dataset.json file. The file may be corrupted or not in valid JSON format.")
        return None, None
    except Exception as e:
        st.warning(f"Unexpected error loading sample data: {str(e)}")
        return None, None

# Main function
def main():
    # Load model
    model, encoder = load_model()
    
    # Load sample data
    benign_sample, ddos_sample = load_sample_data()
    
    # Add the professional header
    add_header()
    
    # Add social sharing buttons
    st.markdown("""
    <div style="display: flex; justify-content: flex-end; gap: 10px; margin-bottom: 20px;">
        <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://ddos-detection-app.streamlit.app" target="_blank" style="text-decoration: none;">
            <button style="background-color: #0077B5; color: white; border: none; padding: 8px 12px; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 5px;">
                <i class="fab fa-linkedin"></i> Share on LinkedIn
            </button>
        </a>
        <a href="https://github.com/yourusername/ddos-detection" target="_blank" style="text-decoration: none;">
            <button style="background-color: #333; color: white; border: none; padding: 8px 12px; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 5px;">
                <i class="fab fa-github"></i> View on GitHub
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # Main app container with professional styling
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Add explanations about BENIGN and DDoS
    with st.expander("❓ What are BENIGN and DDoS?", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color: rgba(40, 167, 69, 0.1); border-radius: 10px; padding: 20px; height: 100%;">
                <h3 style="color: #28a745; border-bottom: 2px solid #28a745; padding-bottom: 10px;">
                    <i class="fas fa-check-circle"></i> BENIGN Traffic
                </h3>
                <p>BENIGN traffic refers to normal, legitimate network activity that doesn't pose any security threat.</p>
                <h4 style="color: #28a745; margin-top: 15px;">Characteristics:</h4>
                <ul>
                    <li>Regular communication patterns</li>
                    <li>Normal packet sizes and intervals</li>
                    <li>Expected protocol behavior</li>
                    <li>Legitimate source and destination</li>
                </ul>
                <p style="font-style: italic; margin-top: 15px;">When the model classifies traffic as BENIGN, it indicates that the network flow appears to be normal user activity.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style="background-color: rgba(220, 53, 69, 0.1); border-radius: 10px; padding: 20px; height: 100%;">
                <h3 style="color: #dc3545; border-bottom: 2px solid #dc3545; padding-bottom: 10px;">
                    <i class="fas fa-exclamation-triangle"></i> DDoS Attack Traffic
                </h3>
                <p>DDoS (Distributed Denial of Service) is an attack where multiple systems flood a target with traffic to make it unavailable.</p>
                <h4 style="color: #dc3545; margin-top: 15px;">Characteristics:</h4>
                <ul>
                    <li>Abnormally high volume of packets</li>
                    <li>Unusual packet patterns</li>
                    <li>Traffic spikes from multiple sources</li>
                    <li>Irregular protocol behavior</li>
                </ul>
                <p style="font-style: italic; margin-top: 15px;">DDoS attacks aim to overwhelm servers, making them unavailable to legitimate users.</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f8f9fa; border-left: 4px solid #0f5997; padding: 15px; margin: 20px 0;">
        <h3 style="margin-top: 0;">How to Use This Tool</h3>
        <p>This application uses a trained XGBoost model to predict whether network traffic is benign or a DDoS attack.
        You can test the model using any of the following methods:</p>
        <ul>
            <li><strong>Manual Input:</strong> Enter feature values manually</li>
            <li><strong>JSON Dictionary:</strong> Enter JSON data as a dictionary</li>
            <li><strong>Sample Data:</strong> Load pre-configured sample data</li>
            <li><strong>Upload JSON:</strong> Upload your own JSON file</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create professionally styled tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Manual Input", 
        "🔠 JSON Dictionary", 
        "📊 Sample Data", 
        "📤 Upload JSON"
    ])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Enter Network Flow Features")
        st.markdown("""
        <p style="margin-bottom: 20px;">
            Enter the values for each feature to make a prediction. 
            Hover over each label to understand what each metric means for network traffic analysis.
        </p>
        """, unsafe_allow_html=True)
        
        # Add explanation section for feature categories
        with st.expander("📚 Feature Categories Explained"):
            st.markdown("""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div style="background-color: rgba(15, 89, 151, 0.1); border-radius: 10px; padding: 15px;">
                    <h4 style="color: #0f5997; border-bottom: 1px solid #0f5997; padding-bottom: 5px;">Flow Features</h4>
                    <p>These features describe the overall characteristics of the network connection:</p>
                    <ul>
                        <li><strong>Flow Duration:</strong> How long the connection lasted</li>
                        <li><strong>Packet Counts:</strong> How many packets were sent in each direction</li>
                    </ul>
                </div>
                
                <div style="background-color: rgba(15, 89, 151, 0.1); border-radius: 10px; padding: 15px;">
                    <h4 style="color: #0f5997; border-bottom: 1px solid #0f5997; padding-bottom: 5px;">Packet Length Features</h4>
                    <p>These features analyze the size of packets in the connection:</p>
                    <ul>
                        <li><strong>Length Statistics:</strong> Total, maximum, minimum, mean, and standard deviation of packet sizes</li>
                        <li>Separate statistics for forward (client to server) and backward (server to client) directions</li>
                    </ul>
                </div>
                
                <div style="background-color: rgba(15, 89, 151, 0.1); border-radius: 10px; padding: 15px;">
                    <h4 style="color: #0f5997; border-bottom: 1px solid #0f5997; padding-bottom: 5px;">Timing Features (IAT)</h4>
                    <p>These features analyze the timing between packets:</p>
                    <ul>
                        <li><strong>IAT Statistics:</strong> Mean, standard deviation, max, min of time between packets</li>
                        <li>Separate statistics for the whole flow and for each direction</li>
                    </ul>
                </div>
                
                <div style="background-color: rgba(15, 89, 151, 0.1); border-radius: 10px; padding: 15px;">
                    <h4 style="color: #0f5997; border-bottom: 1px solid #0f5997; padding-bottom: 5px;">TCP Flag Features</h4>
                    <p>These features count specific TCP flags in the connection:</p>
                    <ul>
                        <li><strong>SYN:</strong> Used to establish connections</li>
                        <li><strong>ACK:</strong> Used to acknowledge received data</li>
                        <li><strong>PSH:</strong> Used to push data to the application layer immediately</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Create columns to organize inputs
        col1, col2, col3 = st.columns(3)
        
        feature_values = {}
        
        # First column - Flow and Packet Count Features
        with col1:
            st.markdown('<h4 style="color: #0f5997;">Flow & Packet Features</h4>', unsafe_allow_html=True)
            for field in [" Flow Duration", " Total Fwd Packets", " Total Backward Packets", 
                         "Total Length of Fwd Packets", " Total Length of Bwd Packets"]:
                feature_values[field] = st.number_input(
                    field, 
                    value=7762328 if field == " Flow Duration" else 
                          1 if field == " Total Fwd Packets" else
                          5 if field == " Total Backward Packets" else
                          6 if field == "Total Length of Fwd Packets" else
                          30,
                    help=FIELD_EXPLANATIONS.get(field, "")
                )
            
            st.markdown('<h4 style="color: #0f5997; margin-top: 20px;">Packet Length Features</h4>', unsafe_allow_html=True)    
            for field in [" Fwd Packet Length Max", " Fwd Packet Length Min", " Fwd Packet Length Mean", 
                         " Fwd Packet Length Std", "Bwd Packet Length Max"]:
                feature_values[field] = st.number_input(
                    field, 
                    value=6 if field != " Fwd Packet Length Std" else 0.0,
                    help=FIELD_EXPLANATIONS.get(field, "")
                )
        
        # Second column - More Packet Length and IAT Features
        with col2:
            st.markdown('<h4 style="color: #0f5997;">More Packet Length Features</h4>', unsafe_allow_html=True)
            for field in [" Bwd Packet Length Min", " Bwd Packet Length Mean", " Bwd Packet Length Std"]:
                feature_values[field] = st.number_input(
                    field, 
                    value=6 if field != " Bwd Packet Length Std" else 0.0,
                    help=FIELD_EXPLANATIONS.get(field, "")
                )
                
            st.markdown('<h4 style="color: #0f5997; margin-top: 20px;">Flow Timing Features</h4>', unsafe_allow_html=True)
            for field in [" Flow IAT Mean", " Flow IAT Std", " Flow IAT Max", " Flow IAT Min"]:
                feature_values[field] = st.number_input(
                    field, 
                    value=1552465.6 if field == " Flow IAT Mean" else
                          3450641.699 if field == " Flow IAT Std" else
                          7725126 if field == " Flow IAT Max" else 1,
                    help=FIELD_EXPLANATIONS.get(field, "")
                )
                
            st.markdown('<h4 style="color: #0f5997; margin-top: 20px;">Forward Timing Features</h4>', unsafe_allow_html=True)    
            for field in ["Fwd IAT Total", " Fwd IAT Mean", " Fwd IAT Std"]:
                feature_values[field] = st.number_input(
                    field, 
                    value=0.0,
                    help=FIELD_EXPLANATIONS.get(field, "")
                )
        
        # Third column - Remaining Timing and Flag Features
        with col3:
            st.markdown('<h4 style="color: #0f5997;">More Forward Timing</h4>', unsafe_allow_html=True)
            for field in [" Fwd IAT Max", " Fwd IAT Min"]:
                feature_values[field] = st.number_input(
                    field, 
                    value=0,
                    help=FIELD_EXPLANATIONS.get(field, "")
                )
                
            st.markdown('<h4 style="color: #0f5997; margin-top: 20px;">Backward Timing Features</h4>', unsafe_allow_html=True)
            for field in ["Bwd IAT Total", " Bwd IAT Mean", " Bwd IAT Std", " Bwd IAT Max", " Bwd IAT Min"]:
                feature_values[field] = st.number_input(
                    field, 
                    value=7733825 if field == "Bwd IAT Total" else
                          1933456.25 if field == " Bwd IAT Mean" else
                          3861115.341 if field == " Bwd IAT Std" else
                          7725126 if field == " Bwd IAT Max" else 1,
                    help=FIELD_EXPLANATIONS.get(field, "")
                )
                
            st.markdown('<h4 style="color: #0f5997; margin-top: 20px;">TCP Flag Features</h4>', unsafe_allow_html=True)
            for field in [" SYN Flag Count", " ACK Flag Count", " PSH Flag Count"]:
                feature_values[field] = st.number_input(
                    field, 
                    value=0.0 if field != " ACK Flag Count" else 1.0,
                    help=FIELD_EXPLANATIONS.get(field, "")
                )
        
        # Make prediction button with improved styling
        st.markdown('<div style="display: flex; justify-content: center; margin-top: 30px;">', unsafe_allow_html=True)
        predict_button = st.button("🔍 Analyze Traffic", key="manual_predict", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if predict_button:
            if model is not None:
                # Extract features in order
                features = list(feature_values.values())
                
                # Get prediction
                prediction, probabilities = predict(features, model, encoder)
                
                # Display result
                display_prediction(prediction, probabilities)
            else:
                st.error("Model could not be loaded. Please check the model file.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Enter JSON Dictionary")
        st.markdown("""
        <p style="margin-bottom: 20px;">
            Paste your JSON data directly in the text area below. The JSON should contain a 'log' object with all 
            the network flow features. You can optionally include a 'label' field for comparison with the prediction.
        </p>
        """, unsafe_allow_html=True)
        
        with st.expander("📝 JSON Format Explanation"):
            st.markdown("""
            <div style="background-color: rgba(15, 89, 151, 0.1); border-radius: 10px; padding: 20px;">
                <h4 style="color: #0f5997; border-bottom: 1px solid #0f5997; padding-bottom: 5px;">Required JSON Format</h4>
                
                <p>Your JSON should follow this structure:</p>
                
                <div style="background-color: #272822; padding: 15px; border-radius: 5px; margin: 15px 0; overflow: auto;">
                <pre style="color: #f8f8f2; margin: 0;"><code>{
    "log": {
        "Feature1": value1,
        "Feature2": value2,
        ...
    },
    "label": "BENIGN" or "DDoS" (optional)
}</code></pre>
                </div>
                
                <p>The 'log' object must contain all 30 features in the correct order. The 'label' field is optional and used only for comparison.</p>
                
                <div style="display: flex; margin-top: 15px;">
                    <div style="background-color: #28a745; color: white; padding: 10px; border-radius: 5px 0 0 5px; width: 80px; text-align: center;">
                        <i class="fas fa-lightbulb"></i><br>Tip
                    </div>
                    <div style="background-color: rgba(40, 167, 69, 0.1); padding: 10px; border-radius: 0 5px 5px 0; flex-grow: 1;">
                        Make sure all feature names match exactly as they appear in the model training data, including any leading spaces.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Default JSON example - Using a DDoS sample instead of BENIGN
        default_json = '''
{
    "log": {
        " Flow Duration": 257921,
        " Total Fwd Packets": 7,
        " Total Backward Packets": 3,
        "Total Length of Fwd Packets": 11607,
        " Total Length of Bwd Packets": 26,
        " Fwd Packet Length Max": 4380,
        " Fwd Packet Length Min": 0,
        " Fwd Packet Length Mean": 1658.142857,
        " Fwd Packet Length Std": 1964.053328,
        "Bwd Packet Length Max": 20,
        " Bwd Packet Length Min": 0,
        " Bwd Packet Length Mean": 8.666666667,
        " Bwd Packet Length Std": 10.26320288,
        " Flow IAT Mean": 28657.88889,
        " Flow IAT Std": 85504.75044,
        " Flow IAT Max": 256670,
        " Flow IAT Min": 2,
        "Fwd IAT Total": 257921,
        " Fwd IAT Mean": 42986.83333,
        " Fwd IAT Std": 104683.5693,
        " Fwd IAT Max": 256670,
        " Fwd IAT Min": 2,
        "Bwd IAT Total": 790,
        " Bwd IAT Mean": 395.0,
        " Bwd IAT Std": 335.1686143,
        " Bwd IAT Max": 632,
        " Bwd IAT Min": 158,
        " SYN Flag Count": 0.0,
        " ACK Flag Count": 1.0,
        " PSH Flag Count": 1.0
    },
    "label": "DDoS"
}
'''
        
        # Text area for JSON input with improved styling
        st.markdown("""
        <div style="background-color: rgba(15, 89, 151, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h4 style="margin-top: 0; color: #0f5997;"><i class="fas fa-code"></i> JSON Input</h4>
        </div>
        """, unsafe_allow_html=True)
        
        json_input = st.text_area("", value=default_json, height=400, label_visibility="collapsed")
        
        # Add styled prediction button
        st.markdown('<div style="display: flex; justify-content: center; margin-top: 30px;">', unsafe_allow_html=True)
        predict_json_button = st.button("🔍 Analyze JSON Data", key="json_predict", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if predict_json_button:
            if model is not None:
                try:
                    # Show a spinner while processing
                    with st.spinner('Analyzing network traffic data...'):
                        # Parse the JSON input
                        data = json.loads(json_input)
                        
                        # Check if the JSON has the expected structure
                        if 'log' in data:
                            # Extract features for prediction
                            features = list(data['log'].values())
                            
                            # Get prediction
                            prediction, probabilities = predict(features, model, encoder)
                            
                            # Display result
                            display_prediction(prediction, probabilities)
                            
                            # Show comparison with label if available
                            if 'label' in data:
                                actual = data['label']
                                if prediction == actual:
                                    st.success(f"✅ Prediction matches label in data: {actual}")
                                else:
                                    st.error(f"❌ Prediction does not match label in data: {actual}")
                        else:
                            st.error("The JSON does not have the expected 'log' field.")
                except json.JSONDecodeError:
                    st.error("Invalid JSON format. Please check your input.")
                except Exception as e:
                    st.error(f"Error processing the input: {e}")
            else:
                st.error("Model could not be loaded. Please check the model file.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Use Sample Data")
        st.markdown("""
        <p style="margin-bottom: 20px;">
            Load pre-defined samples from the dataset to see predictions. 
            This helps understand how the model classifies known BENIGN and DDoS traffic patterns.
        </p>
        """, unsafe_allow_html=True)
        
        # Check if sample data was loaded successfully
        if benign_sample is None or ddos_sample is None:
            st.error("""
            <div style="display: flex; align-items: center; background-color: #f8d7da; padding: 15px; border-radius: 5px;">
                <i class="fas fa-exclamation-triangle" style="color: #842029; font-size: 24px; margin-right: 10px;"></i>
                <div>
                    <strong>Sample data could not be loaded.</strong><br>
                    Make sure the dataset.json file is present in the application directory.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Explanation of sample data
            with st.expander("ℹ️ About the Sample Data"):
                st.markdown("""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div style="background-color: rgba(40, 167, 69, 0.1); border-radius: 10px; padding: 15px;">
                        <h4 style="color: #28a745; border-bottom: 1px solid #28a745; padding-bottom: 5px;">BENIGN Traffic Sample</h4>
                        <p>This is an example of normal network traffic that doesn't show signs of attack.</p>
                        <ul>
                            <li>Regular packet sizes and intervals</li>
                            <li>Normal flow duration</li>
                            <li>Expected packet count ratio</li>
                            <li>Typical TCP flag patterns</li>
                        </ul>
                    </div>
                    
                    <div style="background-color: rgba(220, 53, 69, 0.1); border-radius: 10px; padding: 15px;">
                        <h4 style="color: #dc3545; border-bottom: 1px solid #dc3545; padding-bottom: 5px;">DDoS Traffic Sample</h4>
                        <p>This is an example of traffic that shows characteristics of a DDoS attack.</p>
                        <ul>
                            <li>Unusual packet patterns</li>
                            <li>Atypical timing intervals</li>
                            <li>Suspicious packet size distribution</li>
                            <li>Abnormal TCP flag counts</li>
                        </ul>
                    </div>
                </div>
                <p style="margin-top: 15px;">Examining these samples can help you understand what patterns the model looks for when classifying network traffic.</p>
                """, unsafe_allow_html=True)
            
            # Sample selection with better styling
            st.markdown("""
            <div style="background-color: rgba(15, 89, 151, 0.05); padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h4 style="margin-top: 0; color: #0f5997;"><i class="fas fa-database"></i> Select Sample Type</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Create custom radio buttons with more visual appeal
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                sample_type = st.radio(
                    "Choose a sample type:",
                    options=["BENIGN", "DDoS"],
                    index=1,  # Set DDoS as default selected
                    label_visibility="collapsed",
                    horizontal=True
                )
            
            # Style the container for the button
            st.markdown('<div style="display: flex; justify-content: center; margin-top: 25px;">', unsafe_allow_html=True)
            load_sample_button = st.button("📊 Load Sample and Analyze", key="sample_predict", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if load_sample_button:
                if model is not None:
                    # Show a spinner while processing
                    with st.spinner(f'Loading {sample_type} sample data...'):
                        # Select appropriate sample
                        sample = benign_sample if sample_type == "BENIGN" else ddos_sample
                        
                        if sample:
                            # Create a nice container for the sample data
                            st.markdown(f"""
                            <div style="background-color: rgba(15, 89, 151, 0.05); padding: 20px; border-radius: 10px; margin: 20px 0;">
                                <h4 style="margin-top: 0; color: #0f5997;"><i class="fas fa-file-code"></i> {sample_type} Sample Data</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display sample data in a collapsible section
                            with st.expander("View Raw JSON", expanded=False):
                                st.json(sample)
                            
                            # Extract features for prediction
                            features = list(sample['log'].values())
                            
                            # Get prediction
                            prediction, probabilities = predict(features, model, encoder)
                            
                            # Display result
                            display_prediction(prediction, probabilities)
                            
                            # Show whether prediction matches expected label
                            actual = sample['label']
                            result_color = "#28a745" if prediction == actual else "#dc3545"
                            result_icon = "check-circle" if prediction == actual else "times-circle"
                            result_text = "matches" if prediction == actual else "does not match"
                            
                            st.markdown(f"""
                            <div style="background-color: rgba({15 if prediction == actual else 220}, {167 if prediction == actual else 53}, {69 if prediction == actual else 69}, 0.1); 
                                padding: 15px; border-radius: 10px; margin: 20px 0; text-align: center;">
                                <h4 style="color: {result_color}; margin: 0;">
                                    <i class="fas fa-{result_icon}"></i> Prediction {result_text} actual label: {actual}
                                </h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Add feature analysis with better visualization
                            with st.expander("Feature Analysis", expanded=True):
                                st.markdown("""
                                <h4 style="color: #0f5997; border-bottom: 1px solid #0f5997; padding-bottom: 10px; margin-bottom: 20px;">
                                    Key Features in this Sample
                                </h4>
                                """, unsafe_allow_html=True)
                                
                                feature_names = list(sample['log'].keys())
                                feature_values = list(sample['log'].values())
                                
                                # Create a DataFrame for the features
                                df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Value': feature_values
                                })
                                
                                # Highlight some important features
                                important_features = [" Flow Duration", " Total Fwd Packets", " Total Backward Packets", 
                                                     " Flow IAT Mean", " SYN Flag Count", " ACK Flag Count"]
                                
                                important_df = df[df['Feature'].isin(important_features)]
                                
                                # Create two columns for the visualization
                                feat_col1, feat_col2 = st.columns([1, 1])
                                
                                with feat_col1:
                                    # Create a styled table
                                    st.markdown("""
                                    <div style="background-color: rgba(15, 89, 151, 0.05); padding: 15px; border-radius: 10px;">
                                        <h5 style="color: #0f5997; margin-top: 0;">Most Relevant Features</h5>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.table(important_df.set_index('Feature'))
                                
                                with feat_col2:
                                    # Create a simple bar chart of important features
                                    st.markdown("""
                                    <div style="background-color: rgba(15, 89, 151, 0.05); padding: 15px; border-radius: 10px;">
                                        <h5 style="color: #0f5997; margin-top: 0;">Feature Visualization</h5>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Create a horizontal bar chart
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    
                                    # Use a custom color palette
                                    colors = ["#0f5997" if feature != " Flow Duration" else "#dc3545" 
                                             for feature in important_df['Feature']]
                                    
                                    # Create the bar chart
                                    bars = ax.barh(
                                        important_df['Feature'], 
                                        important_df['Value'],
                                        color=colors
                                    )
                                    
                                    # Add labels and adjust layout
                                    ax.set_xlabel('Value')
                                    ax.set_title('Key Feature Values')
                                    
                                    # Adjust margins
                                    plt.tight_layout()
                                    
                                    # Display the chart
                                    st.pyplot(fig)
                                
                                # Add a brief analysis
                                traffic_color = "#28a745" if actual == "BENIGN" else "#dc3545"
                                st.markdown(f"""
                                <div style="background-color: rgba(15, 89, 151, 0.05); padding: 20px; border-radius: 10px; margin-top: 20px;">
                                    <h5 style="color: #0f5997; margin-top: 0;">Traffic Analysis</h5>
                                    <p><strong>Traffic Type:</strong> <span style="color: {traffic_color};">{actual}</span></p>
                                    <p><strong>Key Observations:</strong></p>
                                    <ul>
                                        <li><strong>Flow Duration:</strong> {sample['log'].get(' Flow Duration')} microseconds</li>
                                        <li><strong>Total Forward Packets:</strong> {sample['log'].get(' Total Fwd Packets')}</li>
                                        <li><strong>Total Backward Packets:</strong> {sample['log'].get(' Total Backward Packets')}</li>
                                        <li><strong>Forward/Backward Ratio:</strong> {sample['log'].get(' Total Fwd Packets')/sample['log'].get(' Total Backward Packets'):.2f if sample['log'].get(' Total Backward Packets') != 0 else 'N/A'}</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.error("Sample data could not be loaded.")
                else:
                    st.error("Model could not be loaded. Please check the model file.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Upload Your Own JSON Data")
        st.markdown("""
        <p style="margin-bottom: 20px;">
            Upload a JSON file with network flow data for prediction. 
            The file should follow the same format as shown in the JSON Dictionary tab.
        </p>
        """, unsafe_allow_html=True)
        
        # Explanation of file format with better styling
        with st.expander("📄 Required File Format"):
            st.markdown("""
            <div style="background-color: rgba(15, 89, 151, 0.1); border-radius: 10px; padding: 20px;">
                <h4 style="color: #0f5997; border-bottom: 1px solid #0f5997; padding-bottom: 5px;">JSON File Format</h4>
                
                <p>Your JSON file should contain either:</p>
                
                <div style="margin-top: 15px;">
                    <h5 style="color: #0f5997; margin-bottom: 10px;">1. Single Object Format:</h5>
                    <div style="background-color: #272822; padding: 15px; border-radius: 5px; overflow: auto;">
                    <pre style="color: #f8f8f2; margin: 0;"><code>{
    "log": { ... feature values ... },
    "label": "BENIGN" or "DDoS" (optional)
}</code></pre>
                    </div>
                </div>
                
                <div style="margin-top: 20px;">
                    <h5 style="color: #0f5997; margin-bottom: 10px;">2. Array Format (for batch prediction):</h5>
                    <div style="background-color: #272822; padding: 15px; border-radius: 5px; overflow: auto;">
                    <pre style="color: #f8f8f2; margin: 0;"><code>[
    {
        "log": { ... feature values ... },
        "label": "BENIGN" or "DDoS" (optional)
    },
    ...
]</code></pre>
                    </div>
                </div>
                
                <div style="display: flex; margin-top: 20px;">
                    <div style="background-color: #0f5997; color: white; padding: 10px; border-radius: 5px 0 0 5px; width: 80px; text-align: center;">
                        <i class="fas fa-info-circle"></i><br>Note
                    </div>
                    <div style="background-color: rgba(15, 89, 151, 0.1); padding: 10px; border-radius: 0 5px 5px 0; flex-grow: 1;">
                        If you upload a file containing an array of objects, the app will process the first object by default.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # File upload section with better styling
        st.markdown("""
        <div style="background-color: rgba(15, 89, 151, 0.05); padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h4 style="margin-top: 0; color: #0f5997;"><i class="fas fa-upload"></i> Upload JSON File</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a centered upload button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_file = st.file_uploader("Choose a JSON file", type="json", label_visibility="collapsed")
        
        if uploaded_file is not None:
            try:
                # Show loading spinner
                with st.spinner('Processing uploaded file...'):
                    # Load the uploaded JSON
                    data = json.load(uploaded_file)
                    
                    # If it's an array, take the first item
                    if isinstance(data, list) and len(data) > 0:
                        st.info(f"""
                        <div style="display: flex; align-items: center; background-color: #cfe2ff; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                            <i class="fas fa-info-circle" style="color: #0d6efd; font-size: 24px; margin-right: 10px;"></i>
                            <div>
                                File contains {len(data)} records. Using the first record for prediction.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        data = data[0]
                    
                    # Display the uploaded data in a collapsible section
                    with st.expander("View Uploaded Data", expanded=False):
                        st.json(data)
                    
                    # Add a nice button to trigger prediction
                    st.markdown('<div style="display: flex; justify-content: center; margin-top: 25px;">', unsafe_allow_html=True)
                    predict_upload_button = st.button("🔍 Analyze Uploaded Data", key="upload_predict", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if predict_upload_button:
                        if model is not None:
                            if 'log' in data:
                                # Extract features for prediction
                                features = list(data['log'].values())
                                
                                # Get prediction
                                prediction, probabilities = predict(features, model, encoder)
                                
                                # Display result
                                display_prediction(prediction, probabilities)
                                
                                # Show comparison with label if available
                                if 'label' in data:
                                    actual = data['label']
                                    result_color = "#28a745" if prediction == actual else "#dc3545"
                                    result_icon = "check-circle" if prediction == actual else "times-circle"
                                    result_text = "matches" if prediction == actual else "does not match"
                                    
                                    st.markdown(f"""
                                    <div style="background-color: rgba({15 if prediction == actual else 220}, {167 if prediction == actual else 53}, {69 if prediction == actual else 69}, 0.1); 
                                         padding: 15px; border-radius: 10px; margin: 20px 0; text-align: center;">
                                        <h4 style="color: {result_color}; margin: 0;">
                                            <i class="fas fa-{result_icon}"></i> Prediction {result_text} label in data: {actual}
                                        </h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.error(f"""
                                <div style="display: flex; align-items: center; background-color: #f8d7da; padding: 15px; border-radius: 5px;">
                                    <i class="fas fa-exclamation-triangle" style="color: #842029; font-size: 24px; margin-right: 10px;"></i>
                                    <div>
                                        The uploaded JSON does not have the expected 'log' field. Please check your file format.
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.error(f"""
                            <div style="display: flex; align-items: center; background-color: #f8d7da; padding: 15px; border-radius: 5px;">
                                <i class="fas fa-exclamation-triangle" style="color: #842029; font-size: 24px; margin-right: 10px;"></i>
                                <div>
                                    Model could not be loaded. Please check the model file.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"""
                <div style="display: flex; align-items: center; background-color: #f8d7da; padding: 15px; border-radius: 5px;">
                    <i class="fas fa-exclamation-triangle" style="color: #842029; font-size: 24px; margin-right: 10px;"></i>
                    <div>
                        Error processing the uploaded file: {e}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Show an empty state with instructions
            st.markdown("""
            <div style="background-color: #f8f9fa; border: 1px dashed #adb5bd; border-radius: 10px; padding: 30px; text-align: center; margin: 30px 0;">
                <i class="fas fa-file-upload" style="font-size: 48px; color: #6c757d; margin-bottom: 20px;"></i>
                <h4 style="color: #6c757d;">No File Uploaded</h4>
                <p style="color: #6c757d;">Drag and drop a JSON file or click 'Browse files' to upload</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Add a professional footer
    st.markdown("""
    <div class="footer">
        <p>© 2023 DDoS Detection System | Created with Streamlit</p>
        <p>Share this tool with your network security team!</p>
        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 10px;">
            <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://ddos-detection-app.streamlit.app" target="_blank" style="color: #0077B5;"><i class="fab fa-linkedin fa-2x"></i></a>
            <a href="https://twitter.com/intent/tweet?url=https://ddos-detection-app.streamlit.app&text=Check%20out%20this%20DDoS%20Detection%20Tool" target="_blank" style="color: #1DA1F2;"><i class="fab fa-twitter fa-2x"></i></a>
            <a href="https://github.com/yourusername/ddos-detection" target="_blank" style="color: #333;"><i class="fab fa-github fa-2x"></i></a>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_prediction(prediction, probabilities):
    # Define colors for each class
    colors = {"BENIGN": "#28a745", "DDoS": "#dc3545"}
    
    # Create a professional looking result card
    st.markdown("""
    <div class="card">
        <h2 style="text-align: center; margin-bottom: 20px;">Analysis Results</h2>
    """, unsafe_allow_html=True)
    
    # Display prediction with appropriate styling
    prediction_class = "benign-prediction" if prediction == "BENIGN" else "ddos-prediction"
    
    st.markdown(f"""
    <div class="prediction-card {prediction_class}">
        <h3 style="margin-bottom: 10px;">Traffic Classification</h3>
        <p style="font-size: 28px; font-weight: 700; color: {colors.get(prediction, '#0f5997')};">
            {prediction}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add explanation of what the prediction means
    if prediction == "BENIGN":
        st.markdown("""
        <div style="background-color: rgba(40, 167, 69, 0.1); border-left: 4px solid #28a745; padding: 15px; margin: 20px 0;">
            <h4 style="color: #28a745; margin-top: 0;">💚 Normal Traffic Detected</h4>
            <p>This network flow appears to be legitimate activity. The pattern matches normal traffic behavior and doesn't show characteristics of a DDoS attack based on the analyzed features.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: rgba(220, 53, 69, 0.1); border-left: 4px solid #dc3545; padding: 15px; margin: 20px 0;">
            <h4 style="color: #dc3545; margin-top: 0;">🚨 DDoS Attack Detected</h4>
            <p>This network flow exhibits characteristics consistent with a DDoS attack. The traffic pattern suggests malicious intent to overwhelm network resources. Immediate investigation is recommended.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create columns for probability bars and visualization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display probabilities with improved styling
        st.markdown("<h3>Confidence Scores</h3>", unsafe_allow_html=True)
        
        for label, prob in probabilities.items():
            color = colors.get(label, "#0f5997")
            st.markdown(
                f"""
                <div style="margin-bottom: 15px;">
                    <p style="margin-bottom: 5px; font-weight: 500;">{label}</p>
                    <div style="display: flex; align-items: center;">
                        <div style="flex-grow: 1; background-color: #e9ecef; border-radius: 5px; height: 25px; overflow: hidden;">
                            <div style="width: {int(prob*100)}%; background-color: {color}; height: 100%;"></div>
                        </div>
                        <div style="width: 60px; text-align: right; margin-left: 10px; font-weight: 600;">{prob:.1%}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with col2:
        # Create a more professional pie chart
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Set a professional style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        labels = list(probabilities.keys())
        sizes = list(probabilities.values())
        color_list = [colors.get(label, '#0f5997') for label in labels]
        
        # Create pie chart with better styling
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=None,  # We'll add a legend instead
            autopct='%1.1f%%', 
            colors=color_list, 
            startangle=90,
            wedgeprops={'width': 0.5, 'edgecolor': 'white', 'linewidth': 2},
            textprops={'fontsize': 14, 'color': '#333333', 'fontweight': 'bold'}
        )
        
        # Make the percentage labels white for better visibility
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Add a white circle at the center to make it a donut chart
        centre_circle = plt.Circle((0, 0), 0.3, fc='white')
        ax.add_patch(centre_circle)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Add a legend with custom styling
        ax.legend(
            wedges, 
            labels, 
            title="Traffic Type",
            loc="center",
            bbox_to_anchor=(0.5, 0),
            fontsize=12
        )
        
        # Add title with custom styling
        plt.title('Prediction Confidence', fontsize=16, fontweight='bold', pad=20)
        
        # Make the plot background transparent
        fig.patch.set_alpha(0.0)
        
        # Display the figure
        st.pyplot(fig)
        
    # Add recommendations section
    st.markdown("""
    <h3 style="margin-top: 30px;">Recommended Actions</h3>
    """, unsafe_allow_html=True)
    
    if prediction == "BENIGN":
        st.markdown("""
        <div class="card" style="border-left: 4px solid #28a745;">
            <h4>✅ For Normal Traffic:</h4>
            <ul>
                <li>Continue regular monitoring of network traffic</li>
                <li>Maintain current security protocols</li>
                <li>Consider periodic security assessments</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="border-left: 4px solid #dc3545;">
            <h4>🛑 For DDoS Attack:</h4>
            <ul>
                <li>Investigate the source IP addresses immediately</li>
                <li>Implement rate limiting for affected services</li>
                <li>Consider activating DDoS mitigation services</li>
                <li>Notify your security team and follow incident response procedures</li>
                <li>Document the attack patterns for future reference</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Close the main card div
    st.markdown("</div>", unsafe_allow_html=True)

# Add Font Awesome for icons
def load_font_awesome():
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    load_font_awesome()
    main() 