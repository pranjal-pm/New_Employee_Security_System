import numpy as np
import tensorflow as tf
from utils import load_and_preprocess_data  # For scaler
from tensorflow.keras.models import load_model

# Load the trained model and scaler
model = load_model('models/gait_model.h5')
_, _, scaler = load_and_preprocess_data('data/employees_gait_data.csv')  # Load scaler from training

def preprocess_incoming_data(raw_data, window_size=200):
    """Preprocess data received from smartphone (e.g., via API)."""
    # Assume raw_data is a list or array of [acc_x, acc_y, acc_z] for a sequence
    if len(raw_data) < window_size:
        raise ValueError("Data too short; need at least {} samples.".format(window_size))
    
    # Trim or pad to window_size
    data = raw_data[:window_size]
    data = scaler.transform(data.reshape(-1, 3)).reshape(1, window_size, 3)  # Normalize
    return data

def perform_gait_analysis(incoming_data):
    """Main function: Analyze gait and return employee ID."""
    preprocessed_data = preprocess_incoming_data(incoming_data)
    prediction = model.predict(preprocessed_data)
    predicted_id = np.argmax(prediction) + 1  # Assuming IDs start from 1
    confidence = np.max(prediction)  # Probability score
    
    if confidence > 0.90:  # Threshold for security; adjust as needed
        return predicted_id, confidence
    else:
        return None, confidence  # No match or low confidence

# Example: Simulate incoming data from smartphone
if __name__ == "__main__":
   
    simulated_data = np.random.randn(200, 3)  # 200 samples of [acc_x, acc_y, acc_z]
    employee_id, confidence = perform_gait_analysis(simulated_data)
    
    if employee_id:
        print(f"Access Granted: Employee ID {employee_id} (Confidence: {confidence:.2f})")
        # In a real system: Trigger door unlock via API
    else:
        print(f"Access Denied: No match (Confidence: {confidence:.2f})")
        # Log the attempt for security review
