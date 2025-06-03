from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and feature list
model = joblib.load('logreg_model.pkl')
selected_features = joblib.load('selected_features.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Ensure all required features are present
    features = [data[feature] for feature in selected_features]
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
