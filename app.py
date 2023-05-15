from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('diabetes.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    #we read all the data required and we fit this data into the pickle we 
    #have made
    if request.method == 'POST':
        num_preg = int(request.form['num_preg'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bp'])
        skinthick = int(request.form['skinthick'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diab_pred = float(request.form['diab_pred'])
        age = int(request.form['age'])

        features = np.array([[num_preg,glucose,bp,skinthick,insulin,bmi,diab_pred,age]])
        util_output = model.predict(features)
        output = util_output[0];
        if output==0:
            return render_template('index.html',prediction_text="Not Diabetic")
        elif output==1:
            return render_template('index.html',prediction_text="Diabetic (81 percent chance)")
        else:
            return render_template('index.html',prediction_text = "Data given cannot determine result")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

