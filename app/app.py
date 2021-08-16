from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from joblib import load
import uuid
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#import math
import os

app = Flask(__name__)
#run_with_ngrok(app)

@app.route('/', methods=['GET', 'POST'])
#@app.route('/entry', methods=['GET', 'POST'])
def hello():
    request_type_str = request.method
    
    if request_type_str == 'GET':
        return render_template('index.html', href='static/base_pic.svg', href2='')
    else:
        
        
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = "static/" + random_string + ".svg"
        model = load('HousingLinear.joblib')
        np_arr = floats_string_to_np_arr(text)
        predictions = model.predict(np_arr.reshape(-1,1))
           
        #list1 = predictions.tolist()
        #list1_str = ''
        #for x in list1:
        #    list1_str = list1_str + str(math.trunc(x)) + ','

        predictions_to_str = str(predictions)
        
        make_picture('housing_small.csv', model, np_arr, path)
        
        #return predictions_to_str
        return render_template('index.html', href=path, href2='The suitable house for values ('+ text +')' +' is:'+predictions_to_str)


def make_picture(training_data_filename,model,new_input_arr, output_file='predictions_pic.svg'):
    housing_data = pd.read_csv(training_data_filename)
    incomes = housing_data['median_income']
    housevalues = housing_data['median_house_value'] 

    predictions = model.predict(incomes.values.reshape(-1,1))
    plt.plot(incomes, predictions, label = 'Linear Regression', color = 'b', alpha = .7)
    plt.scatter(incomes, housevalues, label = 'Actual Test Data', color ='g', alpha = .7)
    plt.legend
    plt.xlabel('median income(10k)')
    plt.ylabel('median house value')
    
    
    new_pred = model.predict(new_input_arr)
    plt.scatter(new_input_arr, new_pred, label = 'Actual Test Data', color ='r', alpha = .7)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    
    
def floats_string_to_np_arr(floats_str):
  def is_float(s):
    try:
      float(s)
      return True
    except:
      return False
  floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
  return floats.reshape(len(floats), 1)
