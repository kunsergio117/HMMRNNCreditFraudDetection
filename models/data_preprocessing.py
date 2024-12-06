import pandas as pd
import os
import glob
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(directory_path='data/raw'):
    """
    Load CSV data from the given directory.
    
    return Data loaded into a DataFrame from all CSV files found.
    """
    # Construct the full path by using os.path.abspath
    full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', directory_path))
    print(f"Looking for CSV files in: {full_path}")

    # Find all CSV files in the directory
    file_paths = glob.glob(os.path.join(full_path, '*.csv'))

    # Ensure there are CSV files in the directory
    if not file_paths:
        raise FileNotFoundError(f"No CSV files found in directory {full_path}.")

    dataframes = [pd.read_csv(file) for file in file_paths]
    data = pd.concat(dataframes, ignore_index=True)
    print(f"Data loaded successfully from {len(file_paths)} file(s).")
    return data

def preprocess_data(df):
    """
    Preprocess the dataset: clean data and standardize features.
    
    """
    # Separate features and target
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and test sets.
    
    Training and test sets (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print("Data split into training and test sets.")
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, directory='data/processed/'):
    """Save the processed data using pickle."""
    # Create an absolute path for the specified directory
    absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', directory))
    
    os.makedirs(absolute_path, exist_ok=True)  # Ensure the directory exists
    try:
        with open(os.path.join(absolute_path, 'X_train.pkl'), 'wb') as f:
            pickle.dump(X_train, f)
        with open(os.path.join(absolute_path, 'X_test.pkl'), 'wb') as f:
            pickle.dump(X_test, f)
        with open(os.path.join(absolute_path, 'y_train.pkl'), 'wb') as f:
            pickle.dump(y_train, f)
        with open(os.path.join(absolute_path, 'y_test.pkl'), 'wb') as f:
            pickle.dump(y_test, f)
        
        print("Processed data saved successfully under data/processed")
    except Exception as e:
        print(f"Error saving processed data: {e}")

# Loading the processed data
def load_processed_data(directory='data/processed/'):
    """Load preprocessed data from the specified directory using absolute paths."""
    # Construct the absolute path to the data directory
    absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', directory))
    
    try:
        with open(os.path.join(absolute_path, 'X_train.pkl'), 'rb') as f:
            X_train = pickle.load(f)
        with open(os.path.join(absolute_path, 'X_test.pkl'), 'rb') as f:
            X_test = pickle.load(f)
        with open(os.path.join(absolute_path, 'y_train.pkl'), 'rb') as f:
            y_train = pickle.load(f)
        with open(os.path.join(absolute_path, 'y_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)
        
        print("Processed data loaded successfully from:", absolute_path)
        
        return X_train, X_test, y_train, y_test
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure the processed data files exist in the 'data/processed/' directory.")
        raise

if __name__ == "__main__":
    data = load_data()
    X, y = preprocess_data(data)
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_processed_data(X_train, X_test, y_train, y_test)