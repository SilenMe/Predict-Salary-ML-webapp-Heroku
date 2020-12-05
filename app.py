import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))#read byte mode

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()] #storing all the inputs in an array
    final_features = [np.array(int_features)] #onverting array to numpy array
    prediction = model.predict(final_features) #saved in model.py

    output = round(prediction[0], 2) #saving the rounded output in variable named "output"

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output)) #rendreing the output in the index template, prediction test is called via jijna in the index.html


if __name__ == "__main__":
    app.run(debug=True)