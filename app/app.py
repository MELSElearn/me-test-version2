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
        path = "app/static/" + random_string + ".svg"
        model = load('app/HousingLinear.joblib')
        np_arr = floats_string_to_np_arr(text)
        predictions = model.predict(np_arr.reshape(-1,1))
           
        #list1 = predictions.tolist()
        #list1_str = ''
        #for x in list1:
        #    list1_str = list1_str + str(math.trunc(x)) + ','

        predictions_to_str = str(predictions)
        
        make_picture('app/housing_small.csv', model, np_arr, path)
        
        #return predictions_to_str
        return render_template('index.html', href=path, href2='The suitable house for values ('+ text +')' +' is:'+predictions_to_str)


def make_picture(training_data_filename,model,new_input_arr, output_file):
    housing_data = pd.read_csv(training_data_filename)
    incomes = housing_data['median_income']
    housevalues = housing_data['median_house_value'] 

    #predictions = model.predict(incomes.values.reshape(-1,1))
    #plt.plot(incomes, predictions, label = 'Linear Regression', color = 'b', alpha = .7)
    #plt.scatter(incomes, housevalues, label = 'Actual Test Data', color ='g', alpha = .7)
    #plt.legend
    #plt.xlabel('median income(10k)')
    #plt.ylabel('median house value')
    
    
    #new_pred = model.predict(new_input_arr)
    #plt.scatter(new_input_arr, new_pred, label = 'Actual Test Data', color ='r', alpha = .7)
    
    #plt.tight_layout()
    #plt.savefig(output_file)
    #plt.show()
    
    
    preds = model.predict(incomes.values.reshape(-1,1))

    fig = px.scatter(x=incomes, y=housevalues, title="Income vs House value", labels={'x': 'Income (10k)',
                                                                                    'y': 'House value'})

    fig.add_trace(go.Scatter(x=incomes.values.reshape(-1,1), y=preds, mode='lines', name='Model'))

    new_preds = model.predict(new_inp_np_arr)

    fig.add_trace(go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)), y=new_preds, name='New Outputs', mode='markers', marker=dict(color='purple', size=20, line=dict(color='purple', width=2))))

    fig.write_image(output_file, width=800, engine='kaleido')
    fig.show()
    
    
def floats_string_to_np_arr(floats_str):
  def is_float(s):
    try:
      float(s)
      return True
    except:
      return False
  floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
  return floats.reshape(len(floats), 1)
