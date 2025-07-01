from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load(r"D:\RTA\extree_model.pkl")  # Adjust the path as needed

@app.route('/')
def home():
    return render_template('index.html')  # This is the HTML you posted


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        day_of_week = int(request.form['day_of_week'])
        age_band = int(request.form['age_band'])
        light_conditions = int(request.form['light_conditions'])
        vehicles = int(request.form['vehicles'])
        casualties = int(request.form['casualties'])
        hour = int(request.form['hour'])
        minute = int(request.form['minute'])
        junction_type = int(request.form['junction_type'])
        road_surface = int(request.form['road_surface'])
        experience = int(request.form['experience'])

        # Arrange in the correct order for the model
        input_data = np.array([
            day_of_week, age_band, light_conditions, vehicles, casualties,
            hour, minute, junction_type, road_surface, experience
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

       
        probs = model.predict_proba(input_data)[0]  # ✔️ Gives probability array
        classes = model.classes_

        result = {cls: f"{round(prob*100, 2)}%" for cls, prob in zip(classes, probs)}

        most_likely = classes[probs.argmax()]  # ✔️ Get highest probability class

        return jsonify({'prediction': result, 'most_likely': most_likely})

       

        
    
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
