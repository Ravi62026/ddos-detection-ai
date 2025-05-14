"""
XGBoost Model Builder for Classification

This script loads data from dataset.json and builds an XGBoost classification model.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle

# 1. Load the dataset
print("Loading dataset...")
with open('dataset.json', 'r') as file:
    data = json.load(file)

# 2. Data preprocessing: Extract features and labels
features = []
labels = []

for item in data:
    # Extract feature values from each log entry
    features.append(list(item['log'].values()))
    labels.append(item['label'])

# Convert to numpy arrays for modeling
X = np.array(features)
y = np.array(labels)

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Label distribution: {np.unique(y, return_counts=True)}")

# Convert string labels to numeric values using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Encoded labels: {np.unique(y_encoded, return_counts=True)}")
print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 4. Define XGBoost model parameters
# Adjust these parameters based on your specific dataset and problem

# 5. Train the XGBoost model
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier()
xgb_model.fit(
    X_train, y_train,
    verbose=True
)

# 6. Evaluate the model
print("\nEvaluating model performance...")
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Convert numeric predictions back to original labels for better interpretability
y_test_original = label_encoder.inverse_transform(y_test)
y_pred_original = label_encoder.inverse_transform(y_pred.astype(int))

print("y_test_original",y_test_original)
print("y_pred_original",y_pred_original)

# Display detailed classification metrics
print("\nClassification Report:")
print(classification_report(y_test_original, y_pred_original))

# Show confusion matrix
conf_matrix = confusion_matrix(y_test_original, y_pred_original)
print("\nConfusion Matrix:")
print(conf_matrix)

# 7. Visualize feature importance
print("\nGenerating feature importance plot...")
plt.figure(figsize=(12, 8))
xgb.plot_importance(xgb_model, max_num_features=20)
plt.title('XGBoost Feature Importance')
plt.savefig('feature_importance.png')
plt.close()

# 8. Save the trained model and label encoder
print("Saving model and label encoder...")
model_data = {
    'model': xgb_model,
    'label_encoder': label_encoder
}
with open('xgboost_model.pkl', 'wb') as model_file:
    pickle.dump(model_data, model_file)

print("Model training and evaluation complete!")

# Example of how to load and use the model for prediction
print("\nExample of model usage:")
print("# Load the model and label encoder")
print("with open('xgboost_model.pkl', 'rb') as file:")
print("    model_data = pickle.load(file)")
print("    loaded_model = model_data['model']")
print("    label_encoder = model_data['label_encoder']")
print("\n# Make predictions and convert back to original labels")
print("numeric_predictions = loaded_model.predict(new_data)")
print("predictions = label_encoder.inverse_transform(numeric_predictions.astype(int))") 