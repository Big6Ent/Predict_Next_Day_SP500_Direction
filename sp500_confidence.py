import gspread
import pandas as pd
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from flask import Flask, render_template
from datetime import datetime, timedelta

def run_script():
    # Authenticate and open the Google Sheet
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('sp500-mes-06736c615696.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open('SP500').sheet1

    # Get the data from the sheet
    data = sheet.get_all_records()

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data)

    # Convert the 'Date' column to a datetime object
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate the date 60 days before today
    start_date = datetime.now() - timedelta(days=60)

    # Filter the DataFrame to include only the last 60 days of data
    df = df[df['Date'] >= start_date]

    # Replace '.' with NaN
    df = df.replace('.', np.nan)
    df = df.dropna()

    # Calculate daily returns
    df['Return'] = df['SP500'].pct_change()

    # Define a function to label the market direction
    def label_market_direction(return_value):
        if return_value > 0.001:
            return 1
        elif return_value < -0.001:
            return -1
        else:
            return 0

    # Create a new column with the market direction labels
    df['Direction'] = df['Return'].apply(label_market_direction)

    # Shift the 'Direction' column up by one to predict the next day's direction
    df['Direction'] = df['Direction'].shift(-1)

    # Drop rows with missing values
    df = df.dropna()

    # Split the data into features (X) and target (y) variables
    X = df[['SP500', 'Return']]
    y = df['Direction']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict the market direction on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy, precision, recall, and F1-score of the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Compute the confusion matrix
    confusion = confusion_matrix(y_test, y_pred)
    confusion_list = list(zip(*confusion))


    # Predict the market direction for the last data point
    last_data_point = X.iloc[-1].values.reshape(1, -1)
    last_direction_prediction = model.predict(last_data_point)

    # Get the class probabilities for the last data point
    confidence_values = model.predict_proba(last_data_point)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": confusion_list,
        "confidence_values": confidence_values
    }

app = Flask(__name__)

@app.route('/')
def home():
    results = run_script()
    accuracy = "{:.2%}".format(results["accuracy"])
    precision = "{:.2%}".format(results["precision"])
    recall = "{:.2%}".format(results["recall"])
    f1 = "{:.2%}".format(results["f1_score"])
    cm = results["confusion_matrix"]
    confidence_values = results["confidence_values"]
    
    now = datetime.now()
    today = now.strftime("%B %d, %Y")
    
    return render_template(
        'index.html',
        title=f'SP500 Prediction for next day, as of {today}',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm,
        confidence_values=confidence_values 
    )

if __name__ == '__main__':
    app.run(debug=True)
