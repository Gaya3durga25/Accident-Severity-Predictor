# 🚦 Accident Severity Predictor (RTA)

The **Accident Severity Predictor** is a machine learning-based web application that predicts the severity level of a road traffic accident based on various input factors such as weather, light conditions, vehicle type, and road surface.  
This project leverages data-driven insights to support better road safety analysis and help authorities and individuals make informed decisions.

Built using **Flask** and **Scikit-learn**, the application provides an intuitive web interface for users to input accident details and instantly get predictions on the potential severity of an incident.

---

## ✨ Features

- **Accident Severity Prediction:**  
  Input road and environmental factors to predict whether an accident is *Slight*, *Serious*, or *Fatal*.

- **User-Friendly Interface:**  
  Simple web form built with HTML and CSS for easy interaction.

- **Real-World Dataset:**  
  Trained on the **RTA (Road Traffic Accident)** dataset with preprocessed and encoded features.

- **Efficient Model Performance:**  
  Developed and trained using the **Extra Trees Classifier** algorithm for reliable and interpretable results.

- **Educational Purpose:**  
  Demonstrates an end-to-end machine learning pipeline — from data preprocessing and model training to web deployment.

---

## 🚀 Technologies Used

**Frontend:** HTML, CSS  
**Backend:** Flask  
**Machine Learning:** Scikit-learn, Pandas, NumPy  
**Model:** Extra Trees Classifier  
**Development Tools:** Jupyter Notebook, Python 3  

---

Navigate to the project directory:

cd Accident-Severity-Predictor


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Install dependencies:

pip install -r requirements.txt


Run the Flask application:

python app.py


Open your browser and navigate to:

http://127.0.0.1:5000
