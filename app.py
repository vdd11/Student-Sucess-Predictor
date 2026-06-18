from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the serialized model on startup
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve values submitted in the HTML form
        age = int(request.form['age'])
        studytime = int(request.form['studytime'])
        failures = int(request.form['failures'])
        absences = int(request.form['absences'])
        
        # Grade inputs from the HTML form (0-100% scale)
        g1_pct = float(request.form['g1'])
        g2_pct = float(request.form['g2'])
        
        # Convert 0-100% scale to the model's native 0-20 scale
        g1_mapped = (g1_pct / 100) * 20
        g2_mapped = (g2_pct / 100) * 20
        
        schoolsup = int(request.form['schoolsup'])
        famsup = int(request.form['famsup'])
        paid = int(request.form['paid'])
        internet = int(request.form['internet'])
        higher = int(request.form['higher'])
        romantic = int(request.form['romantic'])
        
        # Create DataFrame matching the model's expected features
        input_data = pd.DataFrame([[
            age, studytime, failures, absences, g1_mapped, g2_mapped,
            schoolsup, famsup, paid, internet, higher, romantic
        ]], columns=[
            'age', 'studytime', 'failures', 'absences', 'G1', 'G2', 
            'schoolsup', 'famsup', 'paid', 'internet', 'higher', 'romantic'
        ])
        
        # Run prediction
        prediction = model.predict(input_data)
        pass_status = prediction[0] # 1 or 0
        
        return render_template('predict.html', result=pass_status)

if __name__ == '__main__':
    app.run(debug=True)
