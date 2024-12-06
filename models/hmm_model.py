# this file is setting up of the model, and is being utilized in accuracy_testing.py

''' My tentative plan (take note of accuracy, target: 95%): 
preprocessed data will come to both the hmm and rnn models


each will determine independently whether a transaction is considered a fraudulent one, and only when both are in agreement, it will be flagged as such

my reasoning: 
this will allow the transaction to be vetted based on statistical analysis and machine learning, and minimise false positives.

Target:
lower accuracy loss
'''


import numpy as np
from hmmlearn import hmm
import pandas as pd
import subprocess
import pickle
import os
from sklearn.metrics import classification_report
from data_preprocessing import load_processed_data

hmm_model = None

def train_hmm_model():
    global hmm_model
    X_train, X_test, y_train, y_test = load_processed_data()

    # Initialize and train the HMM model
    num_components = 2  # e.g., two states: 'fraud' and 'non-fraud'
    hmm_model = hmm.GaussianHMM(n_components=num_components, covariance_type='full', n_iter=100)

    # Reshape the training data for the HMM
    X_train_hmm = np.column_stack([X_train])

    # Train the model
    print("Training the HMM model...")
    hmm_model.fit(X_train_hmm)
    print("Training complete.")

    # Save the trained HMM model
    with open('models/hmm_model.pkl', 'wb') as f:
        pickle.dump(hmm_model, f)
    

def predict(transaction_sequence):
    # Predict using HMM model and return binary classification
    hidden_states = hmm_model.predict(np.column_stack([transaction_sequence]))
    # Assuming state '1' indicates fraud in the HMM states
    return 1 if 1 in hidden_states else 0

if __name__ == "__main__":
    train_hmm_model()  # Call only if this file is executed directly



