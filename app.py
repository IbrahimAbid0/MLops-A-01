from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd  # Assuming you're using pandas for data preprocessing

# Load your trained ML model (replace 'your_model.pkl' with the actual filename)
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

# Set the template folder explicitly to ensure correct path
app.template_folder = '.'  # Use the current directory for templates

@app.route('/')
def index():
    return render_template('index.html')  # Assuming the file is in the same directory

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        weight = float(request.form['weight'])
        volume = float(request.form['volume'])

        # Preprocess input data if necessary (e.g., scaling, normalization)
        # ... (replace with your preprocessing steps)

        # Make prediction using your model
        prediction = model.predict(np.array([[weight, volume]]))

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)