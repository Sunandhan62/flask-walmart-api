from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model and features
model = joblib.load("optimized_price_model.pkl")
model_features = joblib.load("model_features.pkl")

# Initialize the Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return "Welcome to the Walmart Price Prediction API!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert input data to DataFrame
        df_input = pd.DataFrame([data])

        # Align columns with model
        df_input = df_input.reindex(columns=model_features, fill_value=0)

        # Make prediction
        predicted_price = model.predict(df_input)

        # Return result
        return jsonify({'predicted_price': round(float(predicted_price[0]), 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
