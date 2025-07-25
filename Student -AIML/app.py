from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load your trained model
model_performance = pickle.load(open('model_performance.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance'])
        previous_grade_numeric = float(request.form['previous_grade_numeric'])

        # Prepare data for prediction
        input_features = [[study_hours, attendance, previous_grade_numeric]]

        # Predict
        prediction = model_performance.predict(input_features)[0]

        return render_template('result.html', prediction=prediction)

    except ValueError:
        # Handle invalid inputs
        return render_template('index.html', error="Invalid input. Please enter numeric values.")

if __name__ == "__main__":
    app.run(debug=True)
