# this file is setting up of the rnn_model, and is being utilized in accuracy_testing.py

import numpy as np
import tensorflow as tf
from data_preprocessing import load_processed_data

#for global declaration
rnn_model = None

def train_rnn_model():
    global rnn_model
    # Load the processed data
    X_train, X_test, y_train, y_test = load_processed_data()

    # Reshape your data if necessary (for RNN input, typically requires 3D input)
    # we assume the input data needs to have a third dimension
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Reshape to (samples, timesteps, features)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))  # Similarly reshape the test data

    # Define the RNN rnn_model
    rnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], 1)),  # Use Input() layer for input shape specification
    tf.keras.layers.SimpleRNN(64, return_sequences=True),
    tf.keras.layers.SimpleRNN(32),  # Added another RNN layer for example
    tf.keras.layers.Dense(1, activation='sigmoid')  # Final output layer for binary classification
    ])

    # Compile the rnn_model
    rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the RNN rnn_model
    print("Training the RNN rnn_model...")
    rnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    print("Training complete.")

    # Save the trained model
    rnn_model.save('models/rnn_model.h5')

def predict(transaction_sequence):
    # Predict using RNN rnn_model and return binary classification
    transaction_sequence_reshaped = np.reshape(transaction_sequence, (1, transaction_sequence.shape[0], 1))
    fraud_probability = rnn_model.predict(transaction_sequence_reshaped)
    return 1 if fraud_probability >= 0.5 else 0

if __name__ == "__main__":
    train_rnn_model()  # Call only if this file is executed directly
