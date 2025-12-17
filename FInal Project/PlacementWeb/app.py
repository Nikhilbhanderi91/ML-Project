from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)


model = joblib.load("model/placement_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = []

        # Read inputs IN SAME ORDER AS TRAINING
        for feature in feature_columns:
            value = float(request.form[feature])
            input_data.append(value)

        # Convert to numpy array
        arr = np.array(input_data).reshape(1, -1)

        # Scale input
        arr_scaled = scaler.transform(arr)

        # Predict
        prediction = model.predict(arr_scaled)[0]

        # Output label
        if prediction == 1:
            result = "PLACED "
        else:
            result = "NOT PLACED "

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error occurred: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)