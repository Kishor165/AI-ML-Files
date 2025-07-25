from flask import Flask, request, render_template
import joblib

app = Flask(__name__)  # FIXED: use __name__, not _name_

# Load your trained model
model = joblib.load('movie_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender_str = request.form['gender']

    # Convert gender string to number as per model expectation
    gender_map = {'male': 0, 'female': 1, 'other': 2}  # Adjust if your model only handles male/female
    gender = gender_map.get(gender_str.lower(), 2)  # Default to 2 for "other"

    # Predict
    prediction = model.predict([[age, gender]])

    # Render result page
    return render_template('result.html', age=age, gender=gender_str.capitalize(), prediction=prediction[0])

if __name__ == '__main__':  # FIXED: use __main__
    app.run(debug=True)
