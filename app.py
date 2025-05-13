from flask import Flask, request, render_template
import numpy as np
import joblib
import pickle
# Load model
app = Flask(__name__)
model = joblib.load('files_for_training_model/models/model.joblib')

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        int_features = [float(x) for x in request.form.values()]
        features = [np.array(int_features)]
        
        # Make prediction
        prediction = model.predict(features)[0]
        output = round(prediction, 2)
        
        return render_template('index.html', prediction_text=f'Predicted Heart Disease %: {output}')
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run()
