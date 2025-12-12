# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model & scaler (make sure path is correct)
model = joblib.load("model/placement_prediction_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Example: list the expected fields in correct order
FIELDS = [
    "tenth_percentage","twelfth_percentage","cgpa","basic_aptitude",
    "icp_grade","oops_grade","dsa_grade","daa_grade","dbms_grade","os_grade",
    "cn_grade","iwt_grade","cpmad_grade","aj_grade","awt_grade",
    "communication_skill","aptitude_training","coding_training",
    "tcs_google_score","amcat_score","hackathon_level","leetcode_solved",
    "github_activity","no_projects","no_internships","research_papers",
    "attendance","mock_interview","certifications_score","personality_score"
]

@app.route("/")
def home():
    return "Placement Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # expects JSON
        # Validate
        if not data:
            return jsonify({"error":"No JSON payload received"}), 400

        # Build input vector in correct order
        try:
            x = [float(data[field]) for field in FIELDS]
        except KeyError as e:
            return jsonify({"error": f"Missing field: {e}"}), 400
        except ValueError:
            return jsonify({"error":"Invalid numeric value in input"}), 400

        x = np.array(x).reshape(1, -1)
        x_scaled = scaler.transform(x)
        pred = model.predict(x_scaled)   # assume output is 0 or 1

        # Optionally return prediction probability if model supports it
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x_scaled).tolist()

        result = {"prediction": int(pred[0]), "probability": prob}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)