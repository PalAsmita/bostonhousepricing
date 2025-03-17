import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Parse the incoming JSON data
    data = request.json['data']
    print("Received data:", data)

    # Ensure that data is a list and access the first dictionary if it's a list
    if isinstance(data, list) and isinstance(data[0], dict):
        input_data = np.array(list(data[0].values())).reshape(1, -1)
    else:
        return jsonify({"error": "Invalid data format"}), 400

    print("Reshaped data for prediction:", input_data)

    # Scale the data and make a prediction
    new_data = scaler.transform(input_data)
    output = regmodel.predict(new_data)

    print("Prediction result:", output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)
