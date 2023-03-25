SP500 Prediction Model
This is a Python Flask web application that predicts the market direction of the S&P 500 for the next day using a RandomForestClassifier. The application retrieves data from a Google Sheet, preprocesses it, trains the model, and displays the prediction results on a web page.

Dependencies
gspread
pandas
numpy
oauth2client
scikit-learn
Flask
Setup
Install the required dependencies using pip:
bash
Copy code
pip install -r requirements.txt
Create a Google API project and enable the Google Sheets API.

Create a service account for the Google API project and download the JSON key file.

Replace the file 'sp500-mes-06736c615696.json' with your downloaded JSON key file.

Share your Google Sheet with the email address of the service account you created.

Running the Application
Run the Flask web application:
bash
Copy code
python app.py
Open a web browser and navigate to http://localhost:5000 to view the predictions.
Application Structure
app.py: The main Python script that runs the Flask web application.
index.html: The HTML template for the home page that displays the prediction results.
daily.html: The HTML template for the daily metrics page that displays the accuracy, precision, recall, F1-score, and other metrics for the RandomForestClassifier.
How the Application Works
The application authenticates with the Google Sheets API using a service account and retrieves data from a Google Sheet.
The data is converted to a pandas DataFrame, and the 'Date' column is converted to a datetime object.
The daily returns of the S&P 500 are calculated, and a new 'Direction' column is created based on the daily returns.
The 'Direction' column is shifted up by one to predict the next day's market direction.
The DataFrame is split into training and testing sets, and a RandomForestClassifier is trained on the data.
The trained model is used to predict the market direction for the next day, and various performance metrics are calculated.
The prediction results are displayed on a web page.
Note: This application is for educational purposes only and should not be used for making actual financial or trading decisions.
