CreditFraudDetectionHMM/
│
├── data/
│   ├── raw/                  # Directory for raw CSV data
│   └── processed/            # Directory for processed data
│
├── models/
│   ├── __init__.py           # Make this directory a package
|   ├── data_preprocessing.py # Scripts for data preprocessing
│   ├── hmm_model.py          # Implementations related to the HMM
│   └── rnn_model.py          # Implementations related to the RNN
│   
│   
│
├── notebooks/
│   ├── exploration.ipynb     # Jupyter notebooks for exploratory data analysis
│   └── evaluation.ipynb      # For model evaluation and comparison
│
├── interface/
│   ├── dashboard.py          # Code for creating the user interface (e.g., with Streamlit or Flask)
│   └── templates/            # HTML templates if using a web framework like Flask
│
├── utils/
│   ├── common.py             # Utility functions that can be used across the project
│   └── visualization.py      # Functions for plotting and visualizations
│

│
├── requirements.txt          # List of required Python packages
├── README.md                 # Documentation for your project
└── .gitignore                # Git ignore file

To Joseph:
under interface directory, we will be using Flask as a framework for presenting the dashboard or alerts to the user for credit fraud,
if you feel more comfortable with some other framework like Django you can use it, I just thought that Flask was the simplest and easiest to integrate with this existing
structure.

If you are using Flask, the interface directory will look like this:
├── interface/
│   ├── app.py                 # Main Flask application
│   ├── templates/             # HTML templates for rendering views
│   │   ├── layout.html        # Base template
│   │   └── index.html         # Home page template
│   └── static/                # Static files (CSS, JavaScript, images)

To-do for Joseph:

User authentication

The "accuracy_testing.py" file needs to be run only once (takes awhile to run give it about 10mins), after which
the "flag_transactions.py" file can be run where rnn_model, hmm_model, accuracy_testing are used to flag fraudulent transacitons. 
The output is a pickle .pkl file "flagged_transactions.pkl" that should be opened using pickle for froontend use of the data
This file is the only file you need to concern with for frontend development, where the data will be used for the following:

Alerts

Dashboard

also take note that the "flag_transactions.py" searches for files in data/raw, meaning the .csv files that are uploaded to the application are stored there,
then runs "data_preprocessing.py" on them hence placing them into data/processed, and then operates on them to output the final pickle file containing
the fraudulent transactions that you need to display on the dashboard along with maybe the accuracy of the mprogram? up to you 