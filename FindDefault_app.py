from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('./models/PredictionofCreditCardFraud_model.pkl')
scaler = joblib.load('./models/PredictionofCreditCardFraud_Scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form input
        input_data = [float(request.form[f'feature{i+1}']) for i in range(30)]
        features = np.array(input_data).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        return render_template('index.html',
                               prediction=int(prediction),
                               probability=round(probability * 100, 2))
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
