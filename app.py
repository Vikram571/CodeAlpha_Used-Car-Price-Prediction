from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)
rf = joblib.load('car_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    present_price = float(request.form['present_price'])
    driven_kms = int(request.form['driven_kms'])
    owner = int(request.form['owner'])
    fuel = request.form['fuel_type']
    selling_type = request.form['selling_type']
    transmission = request.form['transmission']

    input_data = {
        'Year': year,
        'Present_Price': present_price,
        'Driven_kms': driven_kms,
        'Owner': owner,
        'Fuel_Type_Diesel': 1 if fuel == 'Diesel' else 0,
        'Fuel_Type_Petrol': 1 if fuel == 'Petrol' else 0,
        'Selling_type_Individual': 1 if selling_type == 'Individual' else 0,
        'Transmission_Manual': 1 if transmission == 'Manual' else 0
    }

    input_df = pd.DataFrame([input_data])
    predicted_price = rf.predict(input_df)[0]

    return f"<h2>Predicted Selling Price: {round(predicted_price, 2)} lakhs</h2>"

if __name__ == '__main__':
    app.run(debug=True)
