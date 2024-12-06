'''
This file is where rnn_model, hmm_model, accuracy_testing are used to flag fraudulent transacitons.
The output is a pickle .pkl file "flagged_transactions.pkl" that should be opened using pickle for froontend use of the data
'''

import os
import subprocess
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from accuracy_testing import predict_transaction

# Function to load the HMM model
def load_hmm_model():
    with open('hmm_model.pkl', 'rb') as f:
        hmm_model = pickle.load(f)
    return hmm_model

# Load the RNN model from file
def load_rnn_model():
    return load_model('rnn_model.h5')

def load_transaction_data(file_path):
    """Load transaction data from a CSV file."""
    return pd.read_csv(file_path)

def flag_fraudulent_transactions(transactions, hmm_model, rnn_model):
    fraudulent_transactions = []
    for _, transaction in transactions.iterrows():
        # Use the predict_transaction logic assuming it accepts a DataFrame row
        if predict_transaction(transaction.values, hmm_model, rnn_model):
            fraudulent_transactions.append(transaction)  # Store the entire row if fraudulent
    return fraudulent_transactions

def store_flagged_transactions(fraudulent_transactions, filename='flagged_transactions.pkl'):
    """Store flagged transactions in a file for frontend access."""
    with open(filename, 'wb') as f:
        pickle.dump(fraudulent_transactions, f)
    print(f"Fraudulent transactions saved to {filename}")

def preprocess_and_flag_transactions():
    # Run the data_preprocessing.py script to process CSV files
    subprocess.run(['python', 'data_preprocessing.py'])
    
    # Load and analyze transactions from data/processed directory
    processed_dir = 'data/processed/'
    if not os.path.exists(processed_dir):
        print("Processed data directory does not exist. Please check.")
        return

    # Get a list of processed CSV files (e.g., X_train.pkl, etc.)
    processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv') or f.endswith('.pkl')]
    
    if not processed_files:
        print("No processed files found in the data/processed directory.")
        return

    # Load models
    hmm_model = load_hmm_model()
    rnn_model = load_rnn_model()

    # Flag fraudulent transactions for each processed file
    for filename in processed_files:
        file_path = os.path.join(processed_dir, filename)
        transactions = load_transaction_data(file_path)

        # Flag transactions
        flagged_transactions = flag_fraudulent_transactions(transactions, hmm_model, rnn_model)

        # Store flagged transactions
        store_flagged_transactions(flagged_transactions, filename=f'flagged_{filename}.pkl')

def main():
    preprocess_and_flag_transactions()

if __name__ == "__main__":
    main()
