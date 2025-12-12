from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model & scaler
model = joblib.load("model/placement_prediction_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # List of all input fields in correct order
        fields = [
            "tenth_percentage","twelfth_percentage","cgpa","basic_aptitude",
            "icp_grade","oops_grade","dsa_grade","daa_grade","dbms_grade","os_grade",
            "cn_grade","iwt_grade","cpmad_grade","aj_grade","awt_grade",
            "communication_skill","aptitude_training","coding_training",
            "tcs_google_score","amcat_score","hackathon_level","leetcode_solved",
            "github_activity","no_projects","no_internships","research_papers",
            "attendance","mock_interview","certifications_score","personality_score"
        ]

        # Read values from HTML form
        input_data = []
        for field in fields:
            value = float(request.form[field])
            input_data.append(value)

        # Convert to array
        arr = np.array(input_data).reshape(1, -1)

        # Scale
        arr_scaled = scaler.transform(arr)

        # Predict
        pred = model.predict(arr_scaled)[0]

        # Convert output to label
        result = "SELECTED ✔" if pred == 1 else "NOT SELECTED ❌"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)