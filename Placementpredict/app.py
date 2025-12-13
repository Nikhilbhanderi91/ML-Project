from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# ---------------------------------------------------
# Load trained model, scaler & feature order
# ---------------------------------------------------
model = joblib.load("model/placement_model.pkl")
scaler = joblib.load("model/scaler.pkl")
FEATURE_COLUMNS = joblib.load("model/feature_columns.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read input values in EXACT feature order
        input_data = []
        for col in FEATURE_COLUMNS:
            value = float(request.form[col])
            input_data.append(value)

        # Convert to numpy array
        input_array = np.array(input_data).reshape(1, -1)

        # Scale input
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Convert prediction to readable output
        result = "SELECTED ✔" if prediction == 1 else "NOT SELECTED ❌"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)