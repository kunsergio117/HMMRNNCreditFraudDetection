import pickle
import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import subprocess
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'data/raw'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set valid credentials
VALID_USERNAME = "user"
VALID_PASSWORD = "password"

# Load flagged transactions data for the dashboard
def load_flagged_transactions():
    try:
        with open("flagged_transactions.pkl", "rb") as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            error = "Invalid username or password. Please try again."
            return render_template('login.html', error=error)
    
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Load the flagged transactions for display
    flagged_data = load_flagged_transactions()
    if flagged_data:
        flagged_transactions = flagged_data.get('flagged_transactions', [])
        accuracy = flagged_data.get('accuracy', 'N/A')
    else:
        flagged_transactions = []
        accuracy = 'N/A'

    # Handle threshold alert setting
    if request.method == 'POST':
        threshold = request.form.get('threshold')
        flash(f"Alert threshold set to: {threshold}")
    
    return render_template('dashboard.html', transactions=flagged_transactions, accuracy=accuracy)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Run data preprocessing and flagging after upload
            subprocess.run(['python', 'data_preprocessing.py'])
            subprocess.run(['python', 'flag_transactions.py'])

            flash(f"File {filename} uploaded and processed successfully.")
            return redirect(url_for('dashboard'))
    
    return render_template('upload.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
