"""
XGBoost Model Prediction Example

This script demonstrates how to load the saved XGBoost model and make predictions.
"""

import json
import numpy as np
import pickle

def load_model_and_encoder(model_path='xgboost_model.pkl'):
    """Load saved model and label encoder"""
    with open(model_path, 'rb') as file:
        model_data = pickle.load(file)
    
    model = model_data['model']
    encoder = model_data['label_encoder']
    
    print(f"Model loaded: {type(model).__name__}")
    print(f"Label encoder classes: {encoder.classes_}")
    
    return model, encoder

def predict_single_sample(sample_data, model, encoder):
    """Make prediction for a single sample"""
    # Convert to numpy array and reshape for single sample prediction
    sample = np.array(sample_data).reshape(1, -1)
    
    # Get numeric prediction
    numeric_prediction = model.predict(sample)[0]
    
    # Get probability scores (confidence)
    probabilities = model.predict_proba(sample)[0]
    
    # Convert back to original label
    original_prediction = encoder.inverse_transform([numeric_prediction])[0]
    
    return {
        'prediction': original_prediction,
        'numeric_prediction': int(numeric_prediction),
        'probabilities': {
            encoder.inverse_transform([i])[0]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
    }

def main():
    # 1. Load model and encoder
    model, encoder = load_model_and_encoder()
    
    # 2. Load some test data from dataset.json
    with open('dataset.json', 'r') as file:
        data = json.load(file)
    
    # 3. Get a few samples for testing predictions
    test_samples = []
    sample_labels = []
    
    # Get 5 samples (3 of one class, 2 of another)
    class_counts = {'BENIGN': 0, 'DDoS': 0}
    max_per_class = {'BENIGN': 3, 'DDoS': 2}
    
    for item in data:
        label = item['label']
        if class_counts[label] < max_per_class[label]:
            test_samples.append(list(item['log'].values()))
            sample_labels.append(label)
            class_counts[label] += 1
        
        if class_counts['BENIGN'] >= max_per_class['BENIGN'] and class_counts['DDoS'] >= max_per_class['DDoS']:
            break
    
    # 4. Make predictions
    print("\n--- Making predictions for test samples ---")
    
    for i, (sample, true_label) in enumerate(zip(test_samples, sample_labels)):
        result = predict_single_sample(sample, model, encoder)
        
        print(f"\nSample {i+1}:")
        print(f"True label: {true_label}")
        print(f"Predicted: {result['prediction']}")
        
        # Print confidence scores
        print("Confidence scores:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.4f}")
        
        # Print prediction correctness
        is_correct = result['prediction'] == true_label
        print(f"Prediction {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    
    # 5. Example of how to use this in a real application
    print("\n--- Example of how to use the model for new data ---")
    print("1. Prepare your feature data in the same format as training data")
    print("2. Load the model and encoder")
    print("3. Call the prediction function")
    print("4. Use the prediction result")

if __name__ == "__main__":
    main() 