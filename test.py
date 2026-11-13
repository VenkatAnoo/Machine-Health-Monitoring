import joblib
import pandas as pd

# Load saved model
print("Loading model...")
model = joblib.load("final_torque_prediction_model.pkl")
print("Model loaded successfully!")

# Create a sample test input row
sample_data = pd.DataFrame({
    'Air temperature [K]': [298.1],
    'Process temperature [K]': [308.6],
    'Rotational speed [rpm]': [1551],
    'Tool wear [min]': [0],
    'Type': ['M'],
})

# Predict
prediction = model.predict(sample_data)
print(f'Test prediction successful! Predicted Torque: {prediction[0]:.2f} Nm')