from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load(r'C:\Users\HC\Downloads\heart disease presiction\rfc_model.pkl')

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    """Render the homepage where users input data."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submissions and make predictions."""
    try:
        # Extract input data from the form
        features = [
            float(request.form['male']),
            float(request.form['age']),
            float(request.form['education']),
            float(request.form['cigsPerDay']),
            float(request.form['prevalentStroke']),
            float(request.form['prevalentHyp']),
            float(request.form['sysBP']),
            float(request.form['diaBP']),
            float(request.form['BMI']),
            float(request.form['heartRate']),
            float(request.form['glucose'])
        ]

        # Convert features into a 2D array for model prediction
        features_array = np.array(features).reshape(1, -1)

        # Make the prediction
        prediction = model.predict(features_array)[0]  # 0 or 1
        probability = model.predict_proba(features_array)[0, 1]  # Probability of CHD (1)

        # Return the results to the result page
        return render_template('result.html', prediction=prediction, probability=probability)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

