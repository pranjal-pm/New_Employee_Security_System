import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report
from utils import load_and_preprocess_data, add_noise
import matplotlib.pyplot as plt

# Load and prepare data
X, y, scaler = load_and_preprocess_data('data/employees_gait_data.csv')

# Add data augmentation for better generalization
X_augmented = add_noise(X)

# Combine original and augmented data
X_combined = np.concatenate([X, X_augmented])
y_combined = np.concatenate([y, y])  # Duplicate labels

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined)

# Build the LSTM model
model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),  # Prevent overfitting
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(np.unique(y).size, activation='softmax')  # Multi-class output (30 classes)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks for best accuracy
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('models/training_history.png')  # Save for GitHub

# Save the model
model.save('models/gait_model.h5')
print("Model saved to models/gait_model.h5")
