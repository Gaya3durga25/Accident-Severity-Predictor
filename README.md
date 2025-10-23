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

## 🛠️ Installation Guide

**Step 0: Clone the repository**  
git clone https://github.com/Gaya3durga25/Accident-Severity-Predictor.git
**Step 1: Navigate to the project directory**  
cd Accident_Severity_Predictor

**Step 2: Create and activate a virtual environment**  
python -m venv venv  
## Linux/Mac  
source venv/bin/activate  
## Windows  
venv\Scripts\activate

**Step 3: Install dependencies**  
pip install -r requirements.txt

**Step 4: Run the Flask application**  
python app1.py

**Step 5: Open your browser and navigate to:**  
http://127.0.0.1:5000

---

## 🖼️ How to Use

**Step 1: Enter Accident Details**  
Fill out the form with weather, light condition, vehicle type, and road surface.

**Step 2: Predict Severity**  
Click the **Predict** button to view the model’s prediction.

**Step 3: Analyze Results**  
The output page shows whether the accident is **Slight**, **Serious**, or **Fatal**.

**Step 4: Experiment Further**  
Change inputs to see how different factors influence accident severity predictions. 

## 📂 Project Structure

```
Accident-Severity-Predictor/
│
├── templates/ # HTML templates for the Flask web app
├── RTA Dataset.csv # Dataset used for model training
├── app.py # Main Flask application
├── app1.py # Optional alternate Flask version
├── extree_model.pkl # Trained ML model
├── model.ipynb # Jupyter Notebook with training pipeline
├── requirements.txt # Dependencies list
├── .gitignore # Ignored files for version control
└── README.md # Project documentation
```











