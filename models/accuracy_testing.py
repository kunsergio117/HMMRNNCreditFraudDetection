'''
This file needs to be run once only, accuracy will be output, and after which the models will be trained and saved 
on the available datasets under data directory. Then you can utilized the flag_transactions.py file without having to go
through retraining(which is time-consuming) too many times
'''

import subprocess
from hmm_model import predict as hmm_predict
from hmm_model import train_hmm_model
from rnn_model import predict as rnn_predict
from rnn_model import train_rnn_model
from sklearn.metrics import accuracy_score
from data_preprocessing import load_processed_data

def preprocess_data():
    """Function to run data preprocessing."""
    subprocess.run(['python', 'data_preprocessing.py'])

def predict_transaction(transaction):
    """Function to predict using HMM and RNN, then apply voting mechanism."""
    score = 0

    # Obtain predictions
    hmm_prediction = hmm_predict(transaction)
    rnn_prediction = rnn_predict(transaction)

    # Voting mechanism: hmm and rnn models are given equal weight in this voting system
    score += hmm_prediction
    score += rnn_prediction

    # Return combined result: 1 if both models agree on fraud, else 0
    return 1 if score == 2 else 0

def evaluate_model(transactions, true_labels):
    """Function to evaluate model accuracy using combined predictions."""
    combined_predictions = [predict_transaction(t) for t in transactions]

    # Calculate and print accuracy metrics
    accuracy = accuracy_score(true_labels, combined_predictions)
    print(f"Accuracy: {accuracy}")

def main():
    """Main function to orchestrate loading data, testing, and evaluation."""
    preprocess_data()  # Ensure data is preprocessed
    
    #training: they will save the models after training
    train_hmm_model()
    train_rnn_model()

    X_train, X_test, y_train, y_test = load_processed_data()

    evaluate_model(X_test, y_test)

# Execute main function
if __name__ == "__main__":
    main()
