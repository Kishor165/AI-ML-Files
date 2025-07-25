from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('taxi_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    try:
        price = float(request.form['Priceperweek'])
        population = float(request.form['Population'])
        income = float(request.form['Monthlyincome'])
        parking = float(request.form['Averageparkingpermonth'])

        # Make prediction
        features = np.array([[price, population, income, parking]])
        prediction = model.predict(features)[0]

        # Round and display
        return render_template('result.html', prediction=int(prediction))
    
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
