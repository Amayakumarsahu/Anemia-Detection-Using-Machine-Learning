from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('c:\\Users\\Amaya\\OneDrive\\Desktop\\Anemia detection project\\model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
        
        Gender = int(request.form['Gender'])
        Hemoglobin = float(request.form['Hemoglobin'])
        MCH = float(request.form['MCH'])
        MCHC = float(request.form['MCHC'])
        MCV = float(request.form['MCV'])

        features_values = np.array([[Gender,Hemoglobin,MCH,MCHC,MCV]])  # reshape to 2D array
        df = pd.DataFrame(features_values, columns=['Gender', 'Hemoglobin','MCH','MCHC','MCV'])
        print(df)

        prediction = model.predict(df)
        print(prediction[0])
        

        if prediction[0] == 0:
            result = "You don't have any Anemic Disease"
        elif prediction[0] == 1:
            result = "You have anemic disease"

        text = "Hence, based on calculation: "
        return render_template("predict.html", prediction_text=text + str(result))

if __name__ == "__main__":
     app.run(debug=False, port=5000)
