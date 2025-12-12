import tkinter as tk
from tkinter import *
import joblib
import numpy as np

# Load Model & Scaler
model = joblib.load("placement_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

# GUI Window
root = tk.Tk()
root.title("Student Placement Prediction")
root.geometry("500x600")

# Labels
Label(root, text="Student Placement Predictor", font=("Arial", 16, "bold")).pack(pady=10)

entries = []
fields = [
    "10th Percentage",
    "12th Percentage",
    "CGPA",
    "Basic Aptitude Score",
    "ICP Grade",
    "OOPS Grade",
    "DSA Grade",
    "DAA Grade",
    "DBMS Grade",
    "OS Grade",
    "CN Grade",
    "IWT Grade",
    "CPMAD Grade",
    "AJ Grade",
    "AWT Grade",
    "Communication Skill",
    "Aptitude Training Performance",
    "Coding Training Performance",
    "TCS/Google Test Score",
    "AMCAT Score",
    "Hackathon Level",
    "LeetCode Problems Solved",
    "GitHub Activity Score",
    "No. of Projects",
    "No. of Internships",
    "Research Papers",
    "Attendance %",
    "Mock Interview Score",
    "Certifications Score",
    "Personality Score"
]

# Create Input Boxes
for field in fields:
    label = Label(root, text=field, font=("Arial", 10))
    label.pack()
    entry = Entry(root)
    entry.pack()
    entries.append(entry)

# Prediction Function
def predict():
    data = []

    for entry in entries:
        data.append(float(entry.get()))

    data = np.array(data).reshape(1, -1)
    scaled_data = scaler.transform(data)

    result = model.predict(scaled_data)

    if result[0] == 1:
        output_label.config(text="Prediction: SELECTED ✔", fg="green")
    else:
        output_label.config(text="Prediction: NOT SELECTED ❌", fg="red")

# Predict Button
Button(root, text="Predict Placement", command=predict, bg="blue", fg="white", font=("Arial", 12)).pack(pady=20)

# Output
output_label = Label(root, text="", font=("Arial", 14, "bold"))
output_label.pack()

root.mainloop()