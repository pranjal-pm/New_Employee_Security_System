# New_Employee_Security_System
This project implements a contactless employee check-in system using smartphone sensor data. Instead of traditional keycards, employees are identified automatically through gait analysis: the system analyzes accelerometer data from the employee’s smartphone to recognize their walking pattern and grants access.  


# Stark Industries Employee Security System (Gait Analysis)

## Project Overview
This project implements a **contactless employee check-in system** using **smartphone accelerometer data**. 
Instead of traditional keycards, employees are identified automatically through **gait analysis**: the system analyzes the walking pattern captured by a smartphone to grant access.

## Key Features
- **Contactless Authentication** – No physical keycards required.
- **Gait Recognition** – Identifies employees based on their unique walking patterns.
- **Machine Learning Model** – Uses a **Convolutional Neural Network (CNN)** trained on accelerometer windows.
- **Real-Time Prediction** – Predicts employees in near real-time.
- **Scalable** – Easily extendable to larger organizations.

## Dataset
- Simulated data for **30 employees** collected using **Physics Toolbox Sensor Suite**.
- Sensor readings include **ax, ay, az accelerometer values** over time.
- Data is preprocessed into fixed-length windows for CNN input.

## Technology Stack
- **Python** – Data preprocessing and model implementation.
- **Pandas / NumPy** – Data handling and numerical computation.
- **TensorFlow / Keras** – CNN model training and inference.
- **Scikit-learn** – Label encoding, train-test split, evaluation.
- **Google Colab** – Notebook execution and testing.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/stark-gait-analysis.git

